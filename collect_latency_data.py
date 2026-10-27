#!/usr/bin/env python3
"""collect_latency_data.py - build M3 latency-predictor datasets from E2 JSONL logs (PRO-IDS).

Reads the E2 experiment files in ``--results-dir``:

    e2-edge-lan_*.jsonl        route=edge   LAN clean
    e2-cloud-lan_*.jsonl       route=cloud  LAN clean
    e2-alt-lan_*.jsonl         mixed routes LAN clean
    e2-cloud-wan-mild_*.jsonl  route=cloud  netem 30 ms
    e2-cloud-wan-bad_*.jsonl   route=cloud  netem 100 ms + 1 % loss

keeps only valid samples, selects the 20 most informative of the 77 flow features per
route by mutual information (TRAIN split only, so cal/test never influence selection),
builds the feature matrices

    edge  X = [s_t (10 fields, no n_c), top-20 flow features, b_i]  -> 31 columns
    cloud X = [s_t (11 fields, with n_c), top-20 flow features, b_i] -> 32 columns

with target y = latency_dispatch_ms (raw milliseconds, never scaled), and writes six
float32 ``.npz`` files plus ``collect_metadata.json`` to ``--output-dir``.

Splits (70/15/15 per run, seeded; see --edge-split / --cloud-split)
    random   (default) every run is split at random by sample, run by run, so each split
             contains every network condition (LAN, WAN-mild, WAN-bad). Needed by tree
             models (GBM), which cannot predict outside the target range seen in training.
             Caveat: neighbouring requests share queue state, so test error can be
             optimistic and says nothing about unseen network conditions.
    blocked  every run is cut into contiguous time blocks in file order: first 70 % train,
             next 15 % cal, last 15 % test. Same range coverage as "random", but the
             splits are separated in time (paper 3.7.4: split by session/time rather than
             by consecutive request). Recommended for the paper's main numbers.
    by_run   (cloud only) train = e2-cloud-lan + cloud rows of e2-alt-lan, cal =
             e2-cloud-wan-mild, test = e2-cloud-wan-bad. Splits do NOT overlap in y, so a GBM
             will fail on cal/test: use it only to measure sensitivity to unseen conditions.
    Edge rows come from only two runs (e2-edge-lan, edge rows of e2-alt-lan), so by_run is
    not available for edge.

Extra behaviour beyond the minimal filter rules (all counted in the metadata ``skips``):
rows with an unparsable line, a route that is not edge/cloud, or a route that contradicts
the file they come from (e.g. an edge row in a cloud-only file) are dropped, as are rows
where one of the 10 mandatory s_t fields is missing or non-finite. For the cloud model a
missing or non-finite ``n_c`` is imputed as 0.0 (it is not used by the edge model).

Example:
    python3 collect_latency_data.py --results-dir results --output-dir M3/artifacts \\
        --p 0.95 --seed 42
"""

import argparse
import json
import logging
import math
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.feature_selection import mutual_info_regression

try:  # optional: progress bars only when stderr is a terminal
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

LOG = logging.getLogger("collect_latency_data")

FEATURE_NAMES = [
    "Protocol", "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets",
    "Fwd Packet Length Max", "Fwd Packet Length Min", "Fwd Packet Length Mean",
    "Fwd Packet Length Std", "Bwd Packet Length Max", "Bwd Packet Length Min",
    "Bwd Packet Length Mean", "Bwd Packet Length Std", "Flow Bytes/s",
    "Flow Packets/s", "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max",
    "Flow IAT Min", "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std",
    "Fwd IAT Max", "Fwd IAT Min", "Bwd IAT Total", "Bwd IAT Mean",
    "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min", "Fwd PSH Flags",
    "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags", "Fwd Header Length",
    "Bwd Header Length", "Fwd Packets/s", "Bwd Packets/s", "Min Packet Length",
    "Max Packet Length", "Packet Length Mean", "Packet Length Std",
    "Packet Length Variance", "FIN Flag Count", "SYN Flag Count",
    "RST Flag Count", "PSH Flag Count", "ACK Flag Count", "URG Flag Count",
    "CWE Flag Count", "ECE Flag Count", "Down/Up Ratio", "Average Packet Size",
    "Avg Fwd Segment Size", "Avg Bwd Segment Size", "Fwd Avg Bytes/Bulk",
    "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate", "Bwd Avg Bytes/Bulk",
    "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate", "Subflow Fwd Packets",
    "Subflow Fwd Bytes", "Subflow Bwd Packets", "Subflow Bwd Bytes",
    "Init_Win_bytes_forward", "Init_Win_bytes_backward", "act_data_pkt_fwd",
    "min_seg_size_forward", "Active Mean", "Active Std", "Active Max",
    "Active Min", "Idle Mean", "Idle Std", "Idle Max", "Idle Min",
]
assert len(FEATURE_NAMES) == 77, "FEATURE_NAMES must list exactly 77 features"
N_FEATURES = len(FEATURE_NAMES)

# Column order of the cloud s_t vector (n_c before h_c_ms); the edge vector drops n_c.
S_T_FIELDS_CLOUD = ["q_e", "q_c", "u_e", "u_c", "lambda_hat", "mu_e_hat", "mu_c_hat",
                    "rtt_ms", "tau_hat_bps", "n_c", "h_c_ms"]
S_T_FIELDS_EDGE = [f for f in S_T_FIELDS_CLOUD if f != "n_c"]
N_C_COL = S_T_FIELDS_CLOUD.index("n_c")
EDGE_ST_COLS = [S_T_FIELDS_CLOUD.index(f) for f in S_T_FIELDS_EDGE]

TARGET = "latency_dispatch_ms"
TOP_K = 20
B_I_FORMULA = "len(features) * 8 + 100"
MIN_SPLIT_SAMPLES = 1000
# The five reasons named in the spec come first; the rest are extra safeguards.
SKIP_REASONS = ("status_not_ok", "missing_features", "missing_s_t", "missing_target",
                "non_finite_features", "s_t_field_invalid", "invalid_json",
                "invalid_route", "route_mismatch")


class DataError(Exception):
    """Unrecoverable input problem (reported without a traceback, exit code 2)."""


@dataclass(frozen=True)
class RunSpec:
    """One E2 run: file pattern and the route its rows must have (None = mixed)."""
    key: str
    pattern: str
    expected_route: str | None


RUNS = (
    RunSpec("edge_lan", "e2-edge-lan_*.jsonl", "edge"),
    RunSpec("cloud_lan", "e2-cloud-lan_*.jsonl", "cloud"),
    RunSpec("alt_lan", "e2-alt-lan_*.jsonl", None),
    RunSpec("cloud_wan_mild", "e2-cloud-wan-mild_*.jsonl", "cloud"),
    RunSpec("cloud_wan_bad", "e2-cloud-wan-bad_*.jsonl", "cloud"),
)


# --------------------------------------------------------------------------- #
# Data containers
# --------------------------------------------------------------------------- #
@dataclass
class RouteData:
    """Validated samples of one route: raw features, s_t (cloud column order), target."""
    feats: np.ndarray  # (N, 77) float64
    st: np.ndarray     # (N, 11) float64; n_c is NaN when missing (only used internally)
    y: np.ndarray      # (N,) float64, milliseconds

    @classmethod
    def empty(cls) -> "RouteData":
        return cls(np.empty((0, N_FEATURES)), np.empty((0, len(S_T_FIELDS_CLOUD))), np.empty(0))

    @classmethod
    def from_lists(cls, feats: list, st: list, y: list) -> "RouteData":
        if not y:
            return cls.empty()
        return cls(np.asarray(feats, dtype=np.float64), np.asarray(st, dtype=np.float64),
                   np.asarray(y, dtype=np.float64))

    @classmethod
    def concat(cls, parts: list) -> "RouteData":
        parts = [p for p in parts if p.n > 0]
        if not parts:
            return cls.empty()
        return cls(np.concatenate([p.feats for p in parts]),
                   np.concatenate([p.st for p in parts]),
                   np.concatenate([p.y for p in parts]))

    @property
    def n(self) -> int:
        return int(self.y.shape[0])

    def take(self, idx: np.ndarray) -> "RouteData":
        return RouteData(self.feats[idx], self.st[idx], self.y[idx])


@dataclass
class FileStats:
    """Line counters for one JSONL file."""
    lines_read: int = 0
    kept: int = 0
    skipped: Counter = field(default_factory=Counter)


# --------------------------------------------------------------------------- #
# Loading and filtering
# --------------------------------------------------------------------------- #
def _finite_number(v: object) -> bool:
    """True for a real finite int/float (bool excluded)."""
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def validate_record(rec: object, expected_route: str | None):
    """Apply the filtering rules to one parsed line.

    Returns a skip-reason string, or ``(route, features[77], s_t[11], y)`` for a valid
    sample. Checks run in a fixed order and a line is counted under its FIRST failure.
    """
    if not isinstance(rec, dict):
        return "invalid_json"
    if rec.get("status") != "ok":
        return "status_not_ok"
    y = rec.get(TARGET)
    if not _finite_number(y):
        return "missing_target"
    feats = rec.get("features")
    if not isinstance(feats, list) or len(feats) != N_FEATURES:
        return "missing_features"
    try:
        f = np.asarray(feats, dtype=np.float64)
    except (TypeError, ValueError):
        return "non_finite_features"
    if not np.isfinite(f).all():
        return "non_finite_features"
    snap = rec.get("s_t_snapshot")
    if not isinstance(snap, dict):
        return "missing_s_t"
    st = []
    for name in S_T_FIELDS_CLOUD:
        v = snap.get(name)
        if name == "n_c":  # optional: edge never uses it, cloud imputes 0.0
            st.append(float(v) if _finite_number(v) else float("nan"))
        elif _finite_number(v):
            st.append(float(v))
        else:
            return "s_t_field_invalid"
    route = rec.get("route")
    if route not in ("edge", "cloud"):
        return "invalid_route"
    if expected_route is not None and route != expected_route:
        return "route_mismatch"
    return route, f, st, float(y)


def _iter_lines(fh, desc: str):
    """Wrap a file in a tqdm bar only when tqdm exists and stderr is a terminal."""
    if tqdm is not None and sys.stderr.isatty():
        return tqdm(fh, desc=desc, unit="line", leave=False)
    return fh


def load_file(path: Path, expected_route: str | None) -> tuple[dict, FileStats]:
    """Parse one JSONL file; returns ({'edge': RouteData, 'cloud': RouteData}, stats)."""
    stats = FileStats()
    acc = {"edge": ([], [], []), "cloud": ([], [], [])}
    # errors="replace": a bad byte only ruins its own line (counted as invalid_json).
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in _iter_lines(fh, path.name):
            line = line.strip()
            if not line:
                continue
            stats.lines_read += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                stats.skipped["invalid_json"] += 1
                rec = None
            if rec is not None:
                res = validate_record(rec, expected_route)
                if isinstance(res, str):
                    stats.skipped[res] += 1
                else:
                    route, f, st, y = res
                    acc[route][0].append(f)
                    acc[route][1].append(st)
                    acc[route][2].append(y)
                    stats.kept += 1
            if stats.lines_read % 1000 == 0:
                LOG.info("%s: %d lines read, %d kept", path.name, stats.lines_read, stats.kept)
    return {r: RouteData.from_lists(*lists) for r, lists in acc.items()}, stats


def load_all(results_dir: Path) -> tuple[dict, dict, dict]:
    """Load every run that exists. Returns (data[run_key][route], file_stats, matched_files)."""
    data: dict = {}
    stats: dict = {}
    matched: dict = {}
    for run in RUNS:
        files = sorted(results_dir.glob(run.pattern))
        matched[run.pattern] = [p.name for p in files]
        if not files:
            LOG.warning("no file matches %s in %s; the split that needs it will be skipped",
                        run.pattern, results_dir)
            continue
        if len(files) > 1:
            LOG.warning("%d files match %s; using all of them in name order: %s",
                        len(files), run.pattern, ", ".join(p.name for p in files))
        merged = {"edge": [], "cloud": []}
        run_stats = FileStats()
        for path in files:
            LOG.info("loading %s", path)
            d, s = load_file(path, run.expected_route)
            for route in merged:
                merged[route].append(d[route])
            run_stats.lines_read += s.lines_read
            run_stats.kept += s.kept
            run_stats.skipped.update(s.skipped)
        data[run.key] = {r: RouteData.concat(parts) for r, parts in merged.items()}
        stats[run.key] = run_stats
        LOG.info("%s: %d lines, %d kept, skipped=%s", run.key, run_stats.lines_read,
                 run_stats.kept, dict(run_stats.skipped) or "none")
    return data, stats, matched


# --------------------------------------------------------------------------- #
# Splitting, feature selection, matrices
# --------------------------------------------------------------------------- #
SPLIT_FRACTIONS = (70, 15)  # train %, cal %; test gets the remainder
EDGE_SPLIT_MODES = ("random", "blocked")
CLOUD_SPLIT_MODES = ("random", "blocked", "by_run")


def split_run(data: RouteData, mode: str, seed: int) -> tuple[RouteData, RouteData, RouteData]:
    """Split ONE run 70/15/15 (integer arithmetic: 15000 -> 10500/2250/2250).

    random: seeded permutation. blocked: contiguous blocks in file (= time) order.
    """
    n = data.n
    order = np.random.RandomState(seed).permutation(n) if mode == "random" else np.arange(n)
    n_train = (n * SPLIT_FRACTIONS[0]) // 100
    n_cal = (n * SPLIT_FRACTIONS[1]) // 100
    return (data.take(order[:n_train]), data.take(order[n_train:n_train + n_cal]),
            data.take(order[n_train + n_cal:]))


def split_pool(runs: dict, mode: str, seed: int) -> tuple[dict, dict]:
    """Split every run separately and concatenate (stratified by run).

    Returns ({'train','cal','test'} -> RouteData, {run_key: {'train','cal','test'} counts}).
    In "random" mode the concatenated splits are shuffled so rows are not grouped by run.
    """
    parts = {"train": [], "cal": [], "test": []}
    counts = {}
    for key, d in runs.items():
        if d.n == 0:
            continue
        for name, piece in zip(("train", "cal", "test"), split_run(d, mode, seed)):
            parts[name].append(piece)
        counts[key] = {name: parts[name][-1].n for name in parts}
    out = {}
    for name, pieces in parts.items():
        merged = RouteData.concat(pieces)
        if mode == "random" and merged.n:
            merged = merged.take(np.random.RandomState(seed + 1).permutation(merged.n))
        out[name] = merged
    return out, counts


def build_splits(data: dict, edge_mode: str, cloud_mode: str, seed: int) -> tuple[dict, dict]:
    """Return (splits[route][name] -> RouteData or None, split counts per run).

    None only occurs for cloud cal/test in "by_run" mode when the source run is missing.
    """
    def part(run_key: str, route: str) -> RouteData:
        return data.get(run_key, {}).get(route, RouteData.empty())

    edge_runs = {"edge_lan": part("edge_lan", "edge"), "alt_lan": part("alt_lan", "edge")}
    if all(d.n == 0 for d in edge_runs.values()):
        raise DataError("no valid edge-route samples: need e2-edge-lan_*.jsonl and/or "
                        "e2-alt-lan_*.jsonl in the results directory")
    edge, edge_counts = split_pool(edge_runs, edge_mode, seed)

    if cloud_mode == "by_run":
        train = RouteData.concat([part("cloud_lan", "cloud"), part("alt_lan", "cloud")])
        cal, test = part("cloud_wan_mild", "cloud"), part("cloud_wan_bad", "cloud")
        if train.n == 0:
            raise DataError("by_run cloud split needs TRAIN samples from e2-cloud-lan_*.jsonl "
                            "and/or e2-alt-lan_*.jsonl (cloud rows)")
        cloud = {"train": train, "cal": cal if cal.n else None, "test": test if test.n else None}
        cloud_counts = {"by_run": {"train": train.n, "cal": cal.n, "test": test.n}}
    else:
        cloud_runs = {k: part(k, "cloud") for k in ("cloud_lan", "alt_lan", "cloud_wan_mild",
                                                    "cloud_wan_bad")}
        if all(d.n == 0 for d in cloud_runs.values()):
            raise DataError("no valid cloud-route samples: need at least one of e2-cloud-lan, "
                            "e2-cloud-wan-mild, e2-cloud-wan-bad, e2-alt-lan in the results directory")
        cloud, cloud_counts = split_pool(cloud_runs, cloud_mode, seed)
    for name in ("cal", "test"):
        if cloud[name] is None:
            LOG.warning("cloud %s split is empty (source run missing or fully filtered); "
                        "latency_cloud_%s.npz will NOT be written", name, name)
    return {"edge": edge, "cloud": cloud}, {"edge": edge_counts, "cloud": cloud_counts}


def select_top_features(train: RouteData, seed: int, k: int = TOP_K) -> tuple[np.ndarray, np.ndarray]:
    """Mutual-information ranking of the 77 raw features on the TRAIN split only.

    Returns (top-k indices in descending MI, MI scores of all 77 features). Ties are
    broken by feature index (stable sort) so the result is deterministic.
    """
    if train.n < 10:
        raise DataError(f"only {train.n} train samples: too few for mutual information")
    mi = mutual_info_regression(train.feats, train.y, random_state=seed)
    order = np.argsort(-mi, kind="stable")
    return order[:k], mi


def x_columns(route: str, top_idx: np.ndarray) -> list:
    """Column names of X for a route, in order."""
    st = S_T_FIELDS_EDGE if route == "edge" else S_T_FIELDS_CLOUD
    return list(st) + [FEATURE_NAMES[i] for i in top_idx] + ["b_i"]


def build_matrix(data: RouteData, route: str, top_idx: np.ndarray) -> np.ndarray:
    """X = [s_t, top-20 flow features, b_i] as float32; cloud n_c NaN -> 0.0."""
    if route == "edge":
        st = data.st[:, EDGE_ST_COLS]
    else:
        st = data.st.copy()
        st[np.isnan(st[:, N_C_COL]), N_C_COL] = 0.0
    b_i = np.full((data.n, 1), data.feats.shape[1] * 8 + 100, dtype=np.float64)
    x = np.hstack([st, data.feats[:, top_idx], b_i]).astype(np.float32)
    if not np.isfinite(x).all():  # e.g. a value that overflows float32
        raise DataError(f"{route}: non-finite values in the float32 feature matrix")
    return x


def y_stats(y: np.ndarray, p: float) -> dict:
    """Summary statistics of a target vector (milliseconds)."""
    return {"N": int(y.size), "y_mean": float(y.mean()), "y_p50": float(np.percentile(y, 50)),
            "y_p95": float(np.percentile(y, 95)), "y_max": float(y.max()),
            "y_quantile_at_p": float(np.quantile(y, p))}


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run_label(run: RunSpec, subset: str | None = None) -> str:
    return run.pattern + (f" ({subset} subset)" if subset else "")


def sources_for(data: dict, edge_mode: str, cloud_mode: str) -> dict:
    """Which runs actually fed each split (only runs that contributed rows are listed)."""
    by_key = {r.key: r for r in RUNS}

    def has(run_key: str, route: str) -> bool:
        return data.get(run_key, {}).get(route, RouteData.empty()).n > 0

    def labels(items: list) -> list:
        return [run_label(by_key[k], sub) for k, route, sub in items if has(k, route)]

    edge_runs = labels([("edge_lan", "edge", None), ("alt_lan", "edge", "edge")])
    cloud_lan = labels([("cloud_lan", "cloud", None), ("alt_lan", "cloud", "cloud")])
    cloud_wan = labels([("cloud_wan_mild", "cloud", None), ("cloud_wan_bad", "cloud", None)])
    desc = {"random": "random 15% of each run", "blocked": "contiguous time block of each run"}
    src = {
        "edge_train_runs": edge_runs,
        "edge_cal_runs": [f"{desc[edge_mode]} (edge rows)"],
        "edge_test_runs": [f"{desc[edge_mode]} (edge rows)"],
    }
    if cloud_mode == "by_run":
        src.update(cloud_train_runs=cloud_lan,
                   cloud_cal_runs=[by_key["cloud_wan_mild"].pattern] if has("cloud_wan_mild", "cloud") else [],
                   cloud_test_runs=[by_key["cloud_wan_bad"].pattern] if has("cloud_wan_bad", "cloud") else [])
    else:
        src.update(cloud_pool_runs=cloud_lan + cloud_wan,
                   cloud_train_runs=["70% of each cloud run"],
                   cloud_cal_runs=[f"{desc[cloud_mode]} (cloud rows)"],
                   cloud_test_runs=[f"{desc[cloud_mode]} (cloud rows)"])
    return src


SPLIT_LABELS = {"random": "stratified_random_70_15_15", "blocked": "blocked_by_run_70_15_15",
                "by_run": "by_run"}
SPLIT_NOTES = {
    "random": ("Per-run random 70/15/15 split by sample (stratified by run), so every split spans "
               "all network conditions; tree models cannot extrapolate beyond the training target "
               "range. Neighbouring requests share queue state, so test error can be optimistic and "
               "does not measure robustness to unseen network conditions."),
    "blocked": ("Per-run contiguous time blocks in file order (first 70% train, next 15% cal, last "
                "15% test). Every split spans all network conditions and the splits are separated in "
                "time (paper 3.7.4)."),
    "by_run": ("Train = LAN runs, cal = WAN-mild, test = WAN-bad. Target ranges barely overlap, so "
               "tree models cannot extrapolate: expect large errors on cal/test."),
}


def run(results_dir: Path, output_dir: Path, p: float, seed: int,
        edge_split: str = "random", cloud_split: str = "random") -> int:
    """Full pipeline. Raises DataError on unrecoverable input problems."""
    if not results_dir.is_dir():
        raise DataError(f"results directory not found: {results_dir}")
    data, file_stats, matched = load_all(results_dir)
    splits, split_counts = build_splits(data, edge_split, cloud_split, seed)  # raises if no train data
    LOG.info("split modes: edge=%s cloud=%s; per-run counts: %s", edge_split, cloud_split, split_counts)

    total_read = sum(s.lines_read for s in file_stats.values())
    total_kept = sum(s.kept for s in file_stats.values())
    reasons = Counter()
    for s in file_stats.values():
        reasons.update(s.skipped)
    LOG.info("lines read=%d kept=%d skipped=%d (%s)", total_read, total_kept,
             total_read - total_kept, dict(reasons) or "none")

    output_dir.mkdir(parents=True, exist_ok=True)
    top_idx, mi_all, columns, stats = {}, {}, {}, {"edge": {}, "cloud": {}}
    summary_lines = []
    for route in ("edge", "cloud"):
        train = splits[route]["train"]
        LOG.info("%s: mutual information on %d train samples x %d features ...",
                 route, train.n, N_FEATURES)
        top_idx[route], mi_all[route] = select_top_features(train, seed)
        columns[route] = x_columns(route, top_idx[route])
        for name in ("train", "cal", "test"):
            d = splits[route][name]
            if d is None:
                stats[route][name] = None
                continue
            if d.n < MIN_SPLIT_SAMPLES:
                LOG.warning("%s %s split has only %d samples (< %d)", route, name, d.n,
                            MIN_SPLIT_SAMPLES)
            x = build_matrix(d, route, top_idx[route])
            y32 = d.y.astype(np.float32)
            if not np.isfinite(y32).all():
                raise DataError(f"{route} {name}: non-finite target after float32 cast")
            path = output_dir / f"latency_{route}_{name}.npz"
            np.savez(path, X=x, y=y32)  # uncompressed on purpose: small and loads faster
            stats[route][name] = y_stats(d.y, p)
            LOG.info("wrote %s X=%s y=%s", path, x.shape, y32.shape)
            s = stats[route][name]
            summary_lines.append(
                f"  {route:<5} {name:<5} X={str(x.shape):<14} y=({d.n},)  y mean={s['y_mean']:.2f} "
                f"p50={s['y_p50']:.2f} p95={s['y_p95']:.2f} max={s['y_max']:.2f} ms")

    skipped_by_reason = {r: int(reasons.get(r, 0)) for r in SKIP_REASONS}
    metadata = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "configuration": {
            "p": p,
            "target": TARGET,
            "split_strategy": f"edge={edge_split}, cloud={cloud_split}",
            "edge_split": SPLIT_LABELS[edge_split],
            "edge_split_note": SPLIT_NOTES[edge_split],
            "cloud_split": SPLIT_LABELS[cloud_split],
            "cloud_split_note": SPLIT_NOTES[cloud_split],
            "seed": seed,
            "b_i_formula": B_I_FORMULA,
            "b_i_value": float(N_FEATURES * 8 + 100),
            "b_i_note": "constant ~716 in current testbed",
            "s_t_fields_edge": S_T_FIELDS_EDGE,
            "s_t_fields_cloud": S_T_FIELDS_CLOUD,
            "cloud_n_c_imputation": "missing or non-finite n_c -> 0.0",
            "edge_feature_count": len(columns["edge"]),
            "cloud_feature_count": len(columns["cloud"]),
            "x_columns_edge": columns["edge"],
            "x_columns_cloud": columns["cloud"],
            "mutual_info": {"estimator": "sklearn.feature_selection.mutual_info_regression",
                            "fit_on": "train split only", "random_state": seed,
                            "selection": f"top-{TOP_K}, descending MI, ties by feature index"},
            "top20_edge_idx": [int(i) for i in top_idx["edge"]],
            "top20_cloud_idx": [int(i) for i in top_idx["cloud"]],
            "top20_edge_mi": [float(mi_all["edge"][i]) for i in top_idx["edge"]],
            "top20_cloud_mi": [float(mi_all["cloud"][i]) for i in top_idx["cloud"]],
            "top20_edge_features": [FEATURE_NAMES[i] for i in top_idx["edge"]],
            "top20_cloud_features": [FEATURE_NAMES[i] for i in top_idx["cloud"]],
            "mutual_info_scores_edge": {n: float(v) for n, v in zip(FEATURE_NAMES, mi_all["edge"])},
            "mutual_info_scores_cloud": {n: float(v) for n, v in zip(FEATURE_NAMES, mi_all["cloud"])},
        },
        "data_stats": stats,
        "sources": {**sources_for(data, edge_split, cloud_split), "matched_files": matched},
        "split_counts_by_run": split_counts,
        "skips": {
            "total_lines_read": total_read,
            "kept": total_kept,
            "skipped_by_reason": skipped_by_reason,
            "per_file": {k: {"lines_read": s.lines_read, "kept": s.kept,
                             "skipped_by_reason": {r: int(s.skipped.get(r, 0)) for r in SKIP_REASONS}}
                         for k, s in file_stats.items()},
        },
    }
    meta_path = output_dir / "collect_metadata.json"
    with meta_path.open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, allow_nan=False)
    LOG.info("wrote %s", meta_path)

    print("\n=== collect_latency_data summary ===")
    print(f"lines read={total_read} kept={total_kept} skipped={total_read - total_kept}")
    print("splits:")
    print("\n".join(summary_lines))
    for route in ("edge", "cloud"):
        top5 = ", ".join(f"{FEATURE_NAMES[i]} ({mi_all[route][i]:.4f})" for i in top_idx[route][:5])
        print(f"top-5 MI features ({route}): {top5}")
    print(f"output: {output_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """CLI definition."""
    ap = argparse.ArgumentParser(
        description="Build M3 latency-predictor datasets (.npz + metadata) from E2 JSONL logs.")
    ap.add_argument("--results-dir", default="results", help="directory with the E2 *.jsonl files")
    ap.add_argument("--output-dir", default="M3/artifacts", help="where .npz files and metadata go")
    ap.add_argument("--p", type=float, default=0.95,
                    help="latency quantile level recorded in the metadata (default 0.95)")
    ap.add_argument("--seed", type=int, default=42, help="seed for MI and the random splits")
    ap.add_argument("--edge-split", choices=EDGE_SPLIT_MODES, default="random",
                    help="edge split: random (per run, by sample) or blocked (per-run time blocks)")
    ap.add_argument("--cloud-split", choices=CLOUD_SPLIT_MODES, default="random",
                    help="cloud split: random / blocked (per run, over all network conditions) "
                         "or by_run (LAN train, WAN-mild cal, WAN-bad test)")
    return ap


def main(argv: list | None = None) -> int:
    """Entry point; returns the process exit code (0 ok, 2 input/data error)."""
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="[%(levelname)s] %(message)s")
    args = build_parser().parse_args(argv)
    if not (0.0 < args.p < 1.0):
        build_parser().error("--p must be in (0, 1)")
    try:
        return run(Path(args.results_dir), Path(args.output_dir), args.p, args.seed,
                   args.edge_split, args.cloud_split)
    except DataError as exc:
        LOG.error("%s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
