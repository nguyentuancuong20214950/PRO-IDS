#!/usr/bin/env python3
"""latency_predictor.py - inference-only wrapper around a trained M3 latency model (PRO-IDS).

``LatencyPredictor`` loads one GBM quantile model produced by train_latency.py, its
calibration offset eta and the exact feature-column order, and returns the calibrated
latency quantile  Q_hat = max(0, model.predict(x) + eta)  in milliseconds for M4.

    pred = LatencyPredictor("models/latency/blocked", route="cloud", preset="balanced", variant="no_ui")
    q_hat_ms = pred.predict(s_t, b_i=716.0)

``s_t`` is the dict served by M2 ``GET /state`` (11 fields). The feature vector is built BY NAME
from the ``x_columns_<route>`` list in collect_metadata.json, so a reordering of columns in the
training pipeline cannot silently break serving. Edge models ignore ``n_c``; cloud models use
it and impute 0.0 when it is None (as collect_latency_data.py did).

Variants: ``no_ui`` needs only s_t and b_i. ``with_ui`` additionally needs the 20 flow
features, which must be supplied INSIDE the ``s_t`` dict under their column names (listed in
``info()["columns"]``), with the same preprocessing as in training.

collect_metadata.json is looked up in this order: next to the models; at the artifacts
directory recorded in train_metadata.json (as written, then relative to the models
directory's ancestors). Copy it next to the models when deploying them without the artifacts.

Thread safety: no mutable state after construction; predict() may be called concurrently.
Dependencies: numpy and joblib (scikit-learn is imported by joblib only to unpickle the model;
load it with the scikit-learn version it was trained with).

Self-test:  PYTHONPATH=src python3 -m pro_ids.m3.latency.predictor [models_dir]
            (default models/latency/blocked)
"""

import json
import logging
import re
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np

LOG = logging.getLogger("latency_predictor")

DEFAULT_B_I = 716.0  # len(features) * 8 + 100 with 77 features (constant in the testbed)
ROUTES = ("edge", "cloud")
VARIANTS = ("no_ui", "with_ui")
P_TOLERANCE = 1e-9


def _read_json(path: Path, what: str) -> dict:
    """Load a JSON file or raise FileNotFoundError naming it."""
    if not path.is_file():
        raise FileNotFoundError(f"{what} not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _finite_float(value: object, label: str) -> float:
    """Convert to a finite float or raise ValueError (None, non-numeric, NaN, inf)."""
    if value is None:
        raise ValueError(f"{label} is None; cannot predict")
    try:
        out = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{label} = {value!r} is not numeric") from None
    if not np.isfinite(out):
        raise ValueError(f"{label} = {value!r} is not finite")
    return out


class LatencyPredictor:
    """Load a trained GBM latency model + calibration eta. Predict Q_hat (ms)."""

    def __init__(self, models_dir: str | Path, route: str, preset: str = "balanced",
                 variant: str = "no_ui"):
        """Load model, calibration eta and column order. Raises on any mismatch.

        Raises:
            FileNotFoundError: models dir, calibration/metadata files or the model are missing.
            ValueError: bad route/variant, unknown preset/variant in calibration.json, or the
                model and the metadata disagree (feature count, layout).
        """
        models_dir = Path(models_dir)
        if not models_dir.is_dir():
            raise FileNotFoundError(f"models directory not found: {models_dir}")
        if route not in ROUTES:
            raise ValueError(f"route must be 'edge' or 'cloud', got {route!r}")
        if variant not in VARIANTS:
            raise ValueError(f"variant must be 'no_ui' or 'with_ui', got {variant!r}")
        self.models_dir, self.route, self.preset, self.variant = models_dir, route, preset, variant

        # --- calibration ---
        calib = _read_json(models_dir / "calibration.json", "calibration.json")
        if route not in calib:
            raise ValueError(f"calibration.json has no '{route}' section")
        cfg_key = f"{preset}_{variant}"
        if cfg_key not in calib[route]:
            raise ValueError(f"calibration.json has no '{cfg_key}' for route={route}. "
                             f"Available: {list(calib[route].keys())}")
        self._eta = float(calib[route][cfg_key]["eta_ms"])
        self._n_cal = int(calib[route][cfg_key]["n_cal"])
        self._p = float(calib["p"])
        if not np.isfinite(self._eta) or self._eta < 0:
            raise ValueError(f"invalid eta_ms in calibration.json: {self._eta}")

        # --- training metadata and column order ---
        self._meta = _read_json(models_dir / "train_metadata.json", "train_metadata.json")
        collect_meta, self._collect_meta_path = self._load_collect_meta()
        cfg = collect_meta["configuration"]
        x_columns = list(cfg[f"x_columns_{route}"])
        s_t_fields = list(cfg[f"s_t_fields_{route}"])
        n_st = len(s_t_fields)
        if x_columns[:n_st] != s_t_fields or x_columns[-1] != "b_i":
            raise ValueError(f"unexpected x_columns layout for {route}: expected the s_t fields "
                             f"first and b_i last, got {x_columns[:n_st]} ... {x_columns[-1]}")
        col_idx = (list(range(n_st)) + [len(x_columns) - 1]) if variant == "no_ui" \
            else list(range(len(x_columns)))
        self._used_columns = [x_columns[i] for i in col_idx]
        self._feature_count = len(col_idx)
        # Precomputed for the hot path.
        self._required = [c for c in self._used_columns if c not in ("b_i", "n_c")]
        self._uses_n_c = "n_c" in self._used_columns

        # --- model (version warnings are recorded and logged, never hidden) ---
        self.model_path = models_dir / route / f"{preset}_{variant}.joblib"
        if not self.model_path.is_file():
            raise FileNotFoundError(f"model file not found: {self.model_path}")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self._model = joblib.load(self.model_path)
        # sklearn emits one InconsistentVersionWarning per unpickled tree: keep unique texts only.
        self._load_warnings = list(dict.fromkeys(f"{w.category.__name__}: {w.message}" for w in caught))
        version_warnings = [str(w.message) for w in caught if w.category.__name__ == "InconsistentVersionWarning"]
        if version_warnings:
            found = re.search(r"from version (\S+) when using version (\d[\w.]*?)\.(?:\s|$)", version_warnings[0])
            trained, runtime = found.groups() if found else (
                self._meta.get("environment", {}).get("scikit_learn", "?"), self._runtime_sklearn_version())
            LOG.warning("Model %s: scikit-learn version mismatch (trained with %s, runtime %s; %d unpickle "
                        "warnings); predictions may drift slightly", self.model_path.name, trained, runtime,
                        len(version_warnings))
        for w in caught:
            if w.category.__name__ != "InconsistentVersionWarning":
                LOG.debug("model load warning: %s", w.message)

        # --- consistency checks ---
        n_expected = getattr(self._model, "n_features_in_", None)
        if n_expected is not None and n_expected != self._feature_count:
            raise ValueError(f"Model expects {n_expected} features, wrapper has {self._feature_count}. "
                             f"Model={self.model_path.name}, variant={variant}")
        for row in self._meta.get("results", []):
            if (row.get("route"), row.get("preset"), row.get("variant")) == (route, preset, variant) \
                    and row.get("feature_count") != self._feature_count:
                raise ValueError(f"train_metadata.json says {row.get('feature_count')} features for "
                                 f"{route}/{cfg_key}, wrapper has {self._feature_count}")
        meta_p = self._meta.get("config", {}).get("p")
        if meta_p is not None and abs(float(meta_p) - self._p) > P_TOLERANCE:
            LOG.warning("p mismatch: calibration.json has %.4f but train_metadata.json has %.4f",
                        self._p, float(meta_p))
        alpha = getattr(self._model, "alpha", None)
        if alpha is not None and abs(float(alpha) - self._p) > P_TOLERANCE:
            LOG.warning("model was trained for quantile alpha=%.4f but calibration uses p=%.4f",
                        float(alpha), self._p)

        LOG.info("Loaded %s/%s_%s: %d features, eta=%.4f ms, p=%.2f, n_cal=%d",
                 route, preset, variant, self._feature_count, self._eta, self._p, self._n_cal)

    # ------------------------------------------------------------------ helpers
    def _load_collect_meta(self) -> tuple[dict, Path]:
        """Find collect_metadata.json (see module docstring for the search order)."""
        candidates = [self.models_dir / "collect_metadata.json"]
        art = self._meta.get("config", {}).get("artifacts_dir")
        if art:
            art_path = Path(art)
            candidates.append(art_path / "collect_metadata.json")
            if not art_path.is_absolute():
                for anc in (self.models_dir.resolve(), *self.models_dir.resolve().parents):
                    candidates.append(anc / art_path / "collect_metadata.json")
        for path in candidates:
            if path.is_file():
                return _read_json(path, "collect_metadata.json"), path
        tried = "\n  ".join(str(c) for c in dict.fromkeys(candidates))
        raise FileNotFoundError(
            "collect_metadata.json not found. Tried:\n  " + tried +
            "\nCopy it from the artifacts directory used for training into the models directory.")

    @staticmethod
    def _runtime_sklearn_version() -> str:
        """Runtime scikit-learn version without importing it (joblib has already done so)."""
        return str(getattr(sys.modules.get("sklearn"), "__version__", "unknown"))

    def _build_feature_vector(self, s_t: dict, b_i: float) -> np.ndarray:
        """Build the exact float32 feature vector used in training, ordered by column name."""
        if not isinstance(s_t, dict):
            raise TypeError(f"s_t must be a dict, got {type(s_t).__name__}")
        missing = [c for c in self._required if c not in s_t]
        if missing:
            raise KeyError(f"s_t missing required field(s): {missing}")
        values = {name: _finite_float(s_t[name], f"s_t['{name}']") for name in self._required}
        if self._uses_n_c:  # cloud only: None -> 0.0, as in collect_latency_data.py
            n_c = s_t.get("n_c")
            values["n_c"] = 0.0 if n_c is None else _finite_float(n_c, "s_t['n_c']")
        values["b_i"] = _finite_float(b_i, "b_i")
        vec = np.array([values[c] for c in self._used_columns], dtype=np.float32)
        if not np.isfinite(vec).all():  # e.g. a value too large for float32
            raise ValueError(f"feature vector contains a non-finite value: {vec}")
        return vec

    # ------------------------------------------------------------------ public API
    def predict(self, s_t: dict, b_i: float = DEFAULT_B_I) -> float:
        """Return Q_hat in ms for one request: max(0, model.predict(x) + eta) as a Python float."""
        vec = self._build_feature_vector(s_t, b_i)
        raw = float(self._model.predict(vec.reshape(1, -1))[0])
        return float(max(0.0, raw + self._eta))

    def predict_batch(self, s_t_list: list[dict], b_i: float = DEFAULT_B_I) -> np.ndarray:
        """Vectorized predict. Returns a (N,) float32 array of Q_hat in ms (empty for an empty list)."""
        if not s_t_list:
            return np.empty(0, dtype=np.float32)
        x = np.stack([self._build_feature_vector(s, b_i) for s in s_t_list])
        raw = self._model.predict(x)
        return np.maximum(0.0, raw + self._eta).astype(np.float32)

    def info(self) -> dict:
        """Summary of what was loaded (JSON-serialisable)."""
        return {
            "models_dir": str(self.models_dir),
            "route": self.route,
            "preset": self.preset,
            "variant": self.variant,
            "feature_count": self._feature_count,
            "columns": list(self._used_columns),
            "eta_ms": self._eta,
            "p": self._p,
            "n_cal": self._n_cal,
            "model_path": str(self.model_path),
            "collect_metadata_path": str(self._collect_meta_path),
            "sklearn_version_trained": self._meta.get("environment", {}).get("scikit_learn"),
            "sklearn_version_runtime": self._runtime_sklearn_version(),
            "load_warnings": list(self._load_warnings),
        }


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #
S_T_EXAMPLE = {
    "q_e": 0, "q_c": 1, "u_e": 45.0, "u_c": 30.0,
    "lambda_hat": 120.0, "mu_e_hat": 180.0, "mu_c_hat": 150.0,
    "rtt_ms": 6.5, "tau_hat_bps": 95000.0, "n_c": 1, "h_c_ms": 250.0,
}


def _with_flow_features(pred: LatencyPredictor) -> dict:
    """Example state plus zero flow features (needed by with_ui models)."""
    s = dict(S_T_EXAMPLE)
    for c in pred.info()["columns"]:
        s.setdefault(c, 0.0)
    return s


def _self_test(models_dir: str) -> int:
    """Run the checks against a models directory. Returns the number of failures."""
    failures = 0

    def check(name: str, condition: bool, detail: str = "") -> None:
        nonlocal failures
        print(f"  {'PASS' if condition else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
        failures += 0 if condition else 1

    def raises(exc: type, fn) -> str | None:
        """Message of the expected exception; None if nothing or a different exception was raised."""
        try:
            fn()
        except exc as e:
            return str(e) or exc.__name__
        except Exception:  # noqa: BLE001 - a different exception type counts as failure
            return None
        return None

    preds = {}
    for route in ROUTES:
        print(f"\n=== {route} (balanced/no_ui) ===")
        pred = preds[route] = LatencyPredictor(models_dir, route=route)
        i = pred.info()
        print(f"  info: {i}")
        check("feature_count", i["feature_count"] == (11 if route == "edge" else 12), str(i["feature_count"]))
        check("columns end with b_i", i["columns"][-1] == "b_i")
        check("n_c in columns" if route == "cloud" else "n_c not in columns",
              ("n_c" in i["columns"]) == (route == "cloud"))
        q = pred.predict(S_T_EXAMPLE, b_i=716.0)
        check("predict returns a Python float in [0, 500]", type(q) is float and 0.0 <= q <= 500.0, f"{q:.3f} ms")
        qb = pred.predict_batch([S_T_EXAMPLE] * 5, b_i=716.0)
        check("predict_batch shape/dtype/equal values", qb.shape == (5,) and qb.dtype == np.float32
              and np.allclose(qb, qb[0]) and abs(float(qb[0]) - q) < 1e-3, str(qb))
        vec = pred._build_feature_vector(S_T_EXAMPLE, 716.0)
        raw = float(pred._model.predict(vec.reshape(1, -1))[0])
        check("matches direct model call + eta (1e-5)", abs(q - max(0.0, raw + pred._eta)) < 1e-5)
        check("predict_batch([]) is empty", pred.predict_batch([]).shape == (0,))

    cloud, edge = preds["cloud"], preds["edge"]
    print("\n=== edge cases ===")
    check("cloud n_c=None imputes 0.0",
          cloud.predict({**S_T_EXAMPLE, "n_c": None}) == cloud.predict({**S_T_EXAMPLE, "n_c": 0}))
    check("edge ignores n_c (present, None, absent)",
          edge.predict(S_T_EXAMPLE) == edge.predict({**S_T_EXAMPLE, "n_c": None})
          == edge.predict({k: v for k, v in S_T_EXAMPLE.items() if k != "n_c"}))
    msg = raises(KeyError, lambda: edge.predict({k: v for k, v in S_T_EXAMPLE.items() if k != "rtt_ms"}))
    check("edge missing rtt_ms -> KeyError naming it", msg is not None and "rtt_ms" in msg)
    for name in ("u_e", "q_c", "h_c_ms"):
        check(f"NaN in {name} -> ValueError", raises(ValueError, lambda n=name: edge.predict({**S_T_EXAMPLE, n: float("nan")})) is not None)
    check("inf -> ValueError", raises(ValueError, lambda: edge.predict({**S_T_EXAMPLE, "rtt_ms": float("inf")})) is not None)
    check("None required field -> ValueError", raises(ValueError, lambda: edge.predict({**S_T_EXAMPLE, "u_c": None})) is not None)
    check("non-numeric -> ValueError", raises(ValueError, lambda: edge.predict({**S_T_EXAMPLE, "u_c": "x"})) is not None)
    check("NaN n_c on cloud -> ValueError", raises(ValueError, lambda: cloud.predict({**S_T_EXAMPLE, "n_c": float("nan")})) is not None)
    check("s_t not a dict -> TypeError", raises(TypeError, lambda: edge.predict([1, 2, 3])) is not None)
    check("bad route -> ValueError", raises(ValueError, lambda: LatencyPredictor(models_dir, route="fog")) is not None)
    check("unknown preset -> ValueError listing available",
          "Available" in (raises(ValueError, lambda: LatencyPredictor(models_dir, route="edge", preset="fast")) or ""))
    check("missing models dir -> FileNotFoundError",
          raises(FileNotFoundError, lambda: LatencyPredictor(models_dir + "_does_not_exist", route="edge")) is not None)

    print("\n=== all configurations in calibration.json ===")
    calib = _read_json(Path(models_dir) / "calibration.json", "calibration.json")
    for route in ROUTES:
        for key in calib[route]:
            preset, variant = key.split("_", 1)
            try:
                p = LatencyPredictor(models_dir, route=route, preset=preset, variant=variant)
                q = p.predict(_with_flow_features(p))
                check(f"{route}/{key} loads and predicts", 0.0 <= q < 1e6, f"{p.info()['feature_count']} features, {q:.2f} ms")
            except Exception as exc:  # noqa: BLE001
                check(f"{route}/{key} loads and predicts", False, f"{type(exc).__name__}: {exc}")
    return failures


def main(argv: list | None = None) -> int:
    """CLI: run the self-test; exit code 0 on success, 1 on any failure."""
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    argv = sys.argv[1:] if argv is None else argv
    models_dir = argv[0] if argv else "models/latency/blocked"
    try:
        failures = _self_test(models_dir)
    except Exception as exc:  # noqa: BLE001 - any construction error is a failed self-test
        print(f"FAIL: {type(exc).__name__}: {exc}")
        return 1
    print("\n=== All self-tests passed ===" if failures == 0 else f"\n=== {failures} self-test(s) FAILED ===")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
