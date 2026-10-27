#!/usr/bin/env python3
"""train_latency.py - train and calibrate the M3 latency quantile predictors (PRO-IDS).

For every combination of route (edge, cloud) x preset (balanced, precise) x feature
variant (no_ui, with_ui) it

  1. fits a GradientBoostingRegressor(loss="quantile", alpha=p) on the train split,
  2. calibrates it on the cal split (paper Eq. 26-28):
         s_j = y_j - pred_j,   k = ceil((n+1) p),   eta = max(0, s_(k)),
     so that the predicted p-quantile is  Q_hat = pred + eta,
  3. evaluates Q_hat on the test split (ER = share of test samples with y > Q_hat,
     target 1 - p; coverage = 1 - ER; MAE = mean |y - Q_hat|),
  4. times single-sample inference, saves the model and records everything.

Input: the directory written by collect_latency_data.py (six .npz files + collect_metadata.json).
Column layout and names come from the metadata, never from hard-coded positions.

Feature variants
    no_ui    s_t fields + b_i        (edge 11 columns, cloud 12)
    with_ui  s_t + top-20 flow features u_i + b_i (edge 31, cloud 32)

MAE definition: all three MAE columns (train/cal/test) are computed on the CALIBRATED
quantile Q_hat = pred + eta, so they are comparable with each other. Because Q_hat targets
the p-quantile it sits above the typical y, so MAE is not a point-forecast error.

Outputs under --output-dir: <route>/<preset>_<variant>.joblib, calibration.json,
evaluation.csv, train_metadata.json. Models are plain joblib files of the sklearn
estimator; load them with the same scikit-learn version they were trained with.
"""

import argparse
import csv
import json
import logging
import math
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import GradientBoostingRegressor

LOG = logging.getLogger("train_latency")

PRESETS = {
    "balanced": {"n_estimators": 200, "max_depth": 5, "learning_rate": 0.05, "min_samples_leaf": 10},
    "precise": {"n_estimators": 500, "max_depth": 6, "learning_rate": 0.03, "min_samples_leaf": 5},
}
VARIANTS = ("no_ui", "with_ui")
ROUTES = ("edge", "cloud")
SPLITS = ("train", "cal", "test")
CSV_FIELDS = (
    "route", "preset", "variant", "n_train", "n_cal", "n_test",
    "train_time_sec", "model_size_kb", "feature_count",
    "eta_ms", "p",
    "train_mae_ms", "cal_mae_ms", "test_mae_ms",
    "test_er", "test_coverage",
    "infer_ms_mean", "infer_ms_p50", "infer_ms_p95",
)
UI_ER_TOLERANCE = 0.01          # no_ui vs with_ui should agree within this ER difference
ER_TOLERANCE = 0.05             # accepted |ER - (1 - p)|
SIZE_LIMITS_KB = {("balanced", "no_ui"): 200.0, ("precise", "no_ui"): 700.0}


class DataError(Exception):
    """Unrecoverable input problem (reported without a traceback, exit code 2)."""


# --------------------------------------------------------------------------- #
# Loading and validation
# --------------------------------------------------------------------------- #
def load_metadata(artifacts_dir: Path) -> dict:
    """Read collect_metadata.json and check the fields this script relies on."""
    path = artifacts_dir / "collect_metadata.json"
    if not path.is_file():
        raise DataError(f"collect_metadata.json not found in {artifacts_dir}")
    with path.open("r", encoding="utf-8") as fh:
        meta = json.load(fh)
    cfg = meta.get("configuration", {})
    needed = [f"{k}_{r}" for r in ROUTES for k in ("x_columns", "s_t_fields")]
    missing = [k for k in needed if k not in cfg]
    if missing:
        raise DataError(f"{path} lacks configuration keys {missing}; regenerate the artifacts "
                        "with the current collect_latency_data.py")
    return meta


def get_column_indices(route: str, variant: str, metadata: dict) -> list[int]:
    """Column indices of X for a variant, derived from the metadata column names.

    no_ui = the s_t columns plus the final b_i column; with_ui = every column.
    """
    cfg = metadata["configuration"]
    cols, st = cfg[f"x_columns_{route}"], cfg[f"s_t_fields_{route}"]
    n_st = len(st)
    if cols[:n_st] != st or cols[-1] != "b_i":
        raise DataError(f"{route}: unexpected x_columns layout (expected s_t fields first, b_i last)")
    if variant == "with_ui":
        return list(range(len(cols)))
    if variant == "no_ui":
        return list(range(n_st)) + [len(cols) - 1]
    raise ValueError(f"unknown variant {variant}")


def validate_artifacts(artifacts_dir: Path, metadata: dict) -> dict:
    """Fail fast on missing/ill-formed .npz files; returns {route: {n_train, n_cal, n_test}}."""
    paths = {(r, s): artifacts_dir / f"latency_{r}_{s}.npz" for r in ROUTES for s in SPLITS}
    missing = [str(p) for p in paths.values() if not p.is_file()]
    if missing:
        raise DataError("missing artifact files:\n  " + "\n  ".join(missing))
    info: dict = {r: {} for r in ROUTES}
    for (route, split), path in paths.items():
        with np.load(path) as d:
            if "X" not in d.files or "y" not in d.files:
                raise DataError(f"{path}: expected keys X and y, found {d.files}")
            x, y = d["X"], d["y"]
        n_cols = len(metadata["configuration"][f"x_columns_{route}"])
        if x.ndim != 2 or x.shape[1] != n_cols or y.shape != (x.shape[0],):
            raise DataError(f"{path}: X{x.shape}/y{y.shape} do not match {n_cols} metadata columns")
        if x.shape[0] == 0:
            raise DataError(f"{path}: no samples")
        if not (np.isfinite(x).all() and np.isfinite(y).all()):
            raise DataError(f"{path}: contains NaN or inf")
        if float(y.min()) == float(y.max()):
            raise DataError(f"{path}: all y values are identical ({float(y[0])})")
        info[route][f"n_{split}"] = int(x.shape[0])
    return info


def load_split(artifacts_dir: Path, route: str, split: str, col_idx: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Load X[:, col_idx] (float32, C-contiguous) and y (float64, raw ms) of one split."""
    with np.load(artifacts_dir / f"latency_{route}_{split}.npz") as d:
        x = np.ascontiguousarray(d["X"][:, col_idx])
        y = d["y"].astype(np.float64)
    return x, y


# --------------------------------------------------------------------------- #
# Model, calibration, evaluation
# --------------------------------------------------------------------------- #
def train_model(x_train: np.ndarray, y_train: np.ndarray, preset_config: dict, p: float,
                seed: int) -> GradientBoostingRegressor:
    """Fit a quantile GBM (pinball loss at level p). y stays in raw milliseconds."""
    model = GradientBoostingRegressor(loss="quantile", alpha=p, random_state=seed, **preset_config)
    model.fit(x_train, y_train)
    return model


def calibrate(model: GradientBoostingRegressor, x_cal: np.ndarray, y_cal: np.ndarray, p: float) -> float:
    """Conformal-style upward correction eta (paper Eq. 26-28), in milliseconds.

    k = ceil((n+1) p). The paper requires k <= n; with too few calibration samples k is
    clipped to n here and a warning is logged (the guarantee no longer holds).
    """
    residuals = y_cal - model.predict(x_cal)  # positive = model under-predicted
    n = residuals.size
    k = math.ceil((n + 1) * p)
    if k > n:
        LOG.warning("calibration set too small (n=%d, k=%d > n): clipping k to n", n, k)
        k = n
    return max(0.0, float(np.sort(residuals)[k - 1]))


def evaluate(model: GradientBoostingRegressor, eta: float, x: np.ndarray, y: np.ndarray) -> dict:
    """ER, coverage and MAE of Q_hat = predict + eta on one split."""
    q_hat = model.predict(x) + eta
    er = float(np.mean(y > q_hat))
    return {"er": er, "coverage": 1.0 - er, "mae": float(np.mean(np.abs(y - q_hat)))}


def measure_inference_time(model: GradientBoostingRegressor, x_test: np.ndarray, n_samples: int = 1000) -> dict:
    """Per-sample predict() latency (ms): every call timed individually, after a short warm-up.

    Measured on this host with sklearn's public predict(), i.e. including its input
    validation; it is not the latency on the edge VM.
    """
    n = min(n_samples, x_test.shape[0])
    for i in range(min(20, n)):
        model.predict(x_test[i:i + 1])
    lat = np.empty(n)
    perf = time.perf_counter
    for i in range(n):
        row = x_test[i:i + 1]
        t0 = perf()
        model.predict(row)
        lat[i] = perf() - t0
    ms = lat * 1e3
    return {"infer_ms_mean": float(ms.mean()), "infer_ms_p50": float(np.percentile(ms, 50)),
            "infer_ms_p95": float(np.percentile(ms, 95))}


def save_config_artifacts(model: GradientBoostingRegressor, eta: float, route: str, preset: str,
                          variant: str, output_dir: Path) -> tuple[Path, float]:
    """Save the model as <output_dir>/<route>/<preset>_<variant>.joblib.

    ``eta`` is not stored in the model file: run() records it in calibration.json.
    Returns (path, size in KB).
    """
    path = output_dir / route / f"{preset}_{variant}.joblib"
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path, os.path.getsize(path) / 1024.0


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def sanitize(obj):
    """JSON-safe copy: numpy -> python, non-finite float -> None."""
    if isinstance(obj, dict):
        return {str(k): sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize(v) for v in obj]
    if isinstance(obj, np.generic):
        return sanitize(obj.item())
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


def check_row(row: dict, p: float) -> None:
    """Log [WARNING]s for a miscalibrated or oversized configuration."""
    name = f"{row['route']} {row['preset']}_{row['variant']}"
    er = row["test_er"]
    if er > 0.5 or er < 0.001:
        LOG.warning("%s: test ER %.4f is badly miscalibrated (target %.3f); check the artifacts "
                    "and --p", name, er, 1 - p)
    elif abs(er - (1 - p)) > ER_TOLERANCE:
        LOG.warning("%s: test ER %.4f is outside the tolerance %.3f +/- %.2f", name, er, 1 - p, ER_TOLERANCE)
    limit = SIZE_LIMITS_KB.get((row["preset"], row["variant"]))
    if limit is not None and row["model_size_kb"] > limit:
        LOG.warning("%s: model size %.0f KB exceeds the expected %.0f KB", name, row["model_size_kb"], limit)


def check_ui_effect(rows: list) -> None:
    """Warn when adding the flow features u_i changes the test ER by more than 0.01."""
    er = {(r["route"], r["preset"], r["variant"]): r["test_er"] for r in rows}
    for route in ROUTES:
        for preset in PRESETS:
            a, b = er.get((route, preset, "no_ui")), er.get((route, preset, "with_ui"))
            if a is not None and b is not None and abs(a - b) > UI_ER_TOLERANCE:
                LOG.warning("%s %s: ui features unexpectedly changed ER by >0.01 (no_ui %.4f vs with_ui %.4f)",
                            route, preset, a, b)


def write_outputs(output_dir: Path, args, rows: list, calibration: dict, split_info: dict,
                  metadata: dict, mi_max: dict) -> None:
    """Write calibration.json, evaluation.csv and train_metadata.json."""
    cal = {"p": args.p, "artifacts_dir": str(args.artifacts_dir), **calibration}
    with (output_dir / "calibration.json").open("w", encoding="utf-8") as fh:
        json.dump(sanitize(cal), fh, indent=2, allow_nan=False)
    with (output_dir / "evaluation.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r[k] is None else r[k]) for k in CSV_FIELDS})
    cfg = metadata["configuration"]
    meta = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "config": {"artifacts_dir": str(args.artifacts_dir), "p": args.p, "seed": args.seed,
                   "presets": {k: PRESETS[k] for k in args.presets}, "variants": list(args.variants)},
        "artifacts_provenance": {k: cfg.get(k) for k in ("split_strategy", "edge_split", "cloud_split", "seed")},
        "environment": {"python": platform.python_version(), "scikit_learn": sklearn.__version__,
                        "numpy": np.__version__, "joblib": joblib.__version__},
        "split_info": split_info,
        "max_flow_feature_mi": mi_max,
        "results": rows,
        "notes": ("Comparison of balanced (200 trees, depth 5) vs precise (500 trees, depth 6), and "
                  "no_ui (s_t + b_i only) vs with_ui (s_t + top-20 flow features + b_i). MAE columns use "
                  "the calibrated quantile Q_hat = pred + eta. Inference times are measured on this host."),
    }
    with (output_dir / "train_metadata.json").open("w", encoding="utf-8") as fh:
        json.dump(sanitize(meta), fh, indent=2, allow_nan=False)


def print_summary(rows: list, p: float) -> None:
    """Final table plus the two recommendations per route."""
    print("\n=== Summary ===")
    print(f"{'route':<6} {'preset':<9} {'variant':<8} {'ER_test':>8} {'MAE_ms':>9} {'eta_ms':>9} "
          f"{'size_KB':>9} {'infer_ms':>9}")
    for r in rows:
        print(f"{r['route']:<6} {r['preset']:<9} {r['variant']:<8} {r['test_er']:>8.4f} "
              f"{r['test_mae_ms']:>9.2f} {r['eta_ms']:>9.2f} {r['model_size_kb']:>9.0f} "
              f"{r['infer_ms_mean']:>9.4f}")
    print("\nRecommended:")
    for route in ROUTES:
        sub = [r for r in rows if r["route"] == route]
        if not sub:
            continue
        best = min(sub, key=lambda r: (abs(r["test_er"] - (1 - p)), r["model_size_kb"]))
        small = min(sub, key=lambda r: (r["model_size_kb"], abs(r["test_er"] - (1 - p))))
        print(f"  {route}: ER closest to {1 - p:.2f} -> {best['preset']}_{best['variant']} "
              f"(ER={best['test_er']:.4f}); smallest model -> {small['preset']}_{small['variant']} "
              f"({small['model_size_kb']:.0f} KB, ER={small['test_er']:.4f})")


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run(args: argparse.Namespace) -> int:
    """Train, calibrate, evaluate and save every configuration. Returns the exit code."""
    if not (0.0 < args.p < 1.0):
        raise DataError("--p must be in (0, 1)")
    artifacts_dir, output_dir = Path(args.artifacts_dir), Path(args.output_dir)
    if not artifacts_dir.is_dir():
        raise DataError(f"artifacts directory not found: {artifacts_dir}")
    metadata = load_metadata(artifacts_dir)
    split_info = validate_artifacts(artifacts_dir, metadata)
    mi_max = {r: float(max(metadata["configuration"].get(f"mutual_info_scores_{r}", {"-": float("nan")}).values()))
              for r in ROUTES}
    LOG.info("artifacts=%s split=%s | max flow-feature MI: edge=%.4f cloud=%.4f",
             artifacts_dir, metadata["configuration"].get("split_strategy"), mi_max["edge"], mi_max["cloud"])
    output_dir.mkdir(parents=True, exist_ok=True)

    configs = [(r, pr, v) for r in ROUTES for pr in args.presets for v in args.variants]
    rows: list = []
    calibration: dict = {r: {} for r in ROUTES}
    interrupted = False
    try:
        for i, (route, preset, variant) in enumerate(configs, 1):
            col_idx = get_column_indices(route, variant, metadata)
            x_tr, y_tr = load_split(artifacts_dir, route, "train", col_idx)
            x_ca, y_ca = load_split(artifacts_dir, route, "cal", col_idx)
            x_te, y_te = load_split(artifacts_dir, route, "test", col_idx)

            t0 = time.perf_counter()
            model = train_model(x_tr, y_tr, PRESETS[preset], args.p, args.seed)
            train_time = time.perf_counter() - t0
            eta = calibrate(model, x_ca, y_ca, args.p)
            ev_tr, ev_ca, ev_te = (evaluate(model, eta, x, y)
                                   for x, y in ((x_tr, y_tr), (x_ca, y_ca), (x_te, y_te)))
            timing = measure_inference_time(model, x_te)
            _, size_kb = save_config_artifacts(model, eta, route, preset, variant, output_dir)

            row = {"route": route, "preset": preset, "variant": variant,
                   "n_train": len(y_tr), "n_cal": len(y_ca), "n_test": len(y_te),
                   "train_time_sec": train_time, "model_size_kb": size_kb,
                   "feature_count": len(col_idx), "eta_ms": eta, "p": args.p,
                   "train_mae_ms": ev_tr["mae"], "cal_mae_ms": ev_ca["mae"], "test_mae_ms": ev_te["mae"],
                   "test_er": ev_te["er"], "test_coverage": ev_te["coverage"], **timing}
            rows.append(row)
            calibration[route][f"{preset}_{variant}"] = {"eta_ms": eta, "n_cal": len(y_ca)}
            LOG.info("%s %s_%s: n_train=%d n_cal=%d n_test=%d features=%d", route, preset, variant,
                     len(y_tr), len(y_ca), len(y_te), len(col_idx))
            print(f"[{i}/{len(configs)}] {route} {preset}_{variant}: train={train_time:.1f}s "
                  f"eta={eta:.3f}ms ER_test={ev_te['er']:.4f} MAE_test={ev_te['mae']:.2f}ms "
                  f"size={size_kb:.0f}KB", flush=True)
            check_row(row, args.p)
    except KeyboardInterrupt:
        interrupted = True
        LOG.warning("interrupted: writing results for the %d finished configuration(s)", len(rows))

    if rows:
        check_ui_effect(rows)
        write_outputs(output_dir, args, rows, calibration, split_info, metadata, mi_max)
        print_summary(rows, args.p)
        print(f"\noutput: {output_dir}")
    return 130 if interrupted else 0


def build_parser() -> argparse.ArgumentParser:
    """CLI definition."""
    ap = argparse.ArgumentParser(
        description="Train and calibrate GBM quantile latency predictors (M3) from collect_latency_data.py artifacts.")
    ap.add_argument("--artifacts-dir", default="M3/artifacts_blocked", help="directory with the .npz files and collect_metadata.json")
    ap.add_argument("--output-dir", default="M3/models/blocked", help="where models and reports are written")
    ap.add_argument("--presets", nargs="+", choices=sorted(PRESETS), default=list(PRESETS), help="GBM presets to train")
    ap.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS), help="feature variants to train")
    ap.add_argument("--p", type=float, default=0.95, help="target quantile level (default 0.95)")
    ap.add_argument("--seed", type=int, default=42)
    return ap


def main(argv: list | None = None) -> int:
    """Entry point; returns the process exit code (0 ok, 2 input error, 130 interrupted)."""
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="[%(levelname)s] %(message)s")
    args = build_parser().parse_args(argv)
    args.presets = list(dict.fromkeys(args.presets))
    args.variants = list(dict.fromkeys(args.variants))
    try:
        return run(args)
    except DataError as exc:
        LOG.error("%s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
