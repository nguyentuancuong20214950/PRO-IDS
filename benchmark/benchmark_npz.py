#!/usr/bin/env python3
"""benchmark_npz.py - latency / resource benchmark for quantized NpzMLP detectors.

PRO-IDS (Predictive Risk-Aware Offloading for Edge-Cloud IDS).

CAVEAT: Results are preliminary, single-VM, 1 physical core, not representative of production edge/cloud.

What is measured
----------------
NpzMLP stores INT5/INT6 weights. On a CPU without integer-native matmul, the
weights must be dequantized to float before the forward pass, so quantization
may not be faster than FP32 on this hardware. To show WHERE the time goes, the
cost is split into separately timed modes:

  dequant_only          NpzMLP.dequantize() on a fresh instance (the instance
                        is built OUTSIDE the timer; only dequantize() is timed).
  forward_pass          NpzMLP.forward(X) on an instance dequantized once
                        before timing (pure matmul/activation cost).
  predict_proba         NpzMLP.predict_proba(X) on a pre-dequantized instance
                        (forward + whatever per-call overhead the class adds).
  dequant_plus_forward  dequantize() + forward(X) in ONE timed block, with the
                        raw quantized weights already in memory (instance built
                        outside the timer). Cost of re-dequantizing per request.
  full                  NpzMLP(path) [reads the .npz from disk] + dequantize()
                        + forward(X) in one timed block: worst-case cold start.

Interpretation note: "dequant_plus_forward" and "full" would be identical if
both included the constructor, so the file read/parse is what separates them.
"full" can only be truly disk-cold if the OS page cache is empty; check
resources.disk_read_bytes (0 means served from cache).

Threads: OMP/MKL/OPENBLAS/NUMEXPR thread env vars are exported from
--num-threads BEFORE numpy is imported (BLAS reads them once at load time).
"""

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import threading
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _early_num_threads(argv):
    """Scan argv for --num-threads without importing numpy (default 1)."""
    value = None
    for i, tok in enumerate(argv):
        if tok == "--num-threads" and i + 1 < len(argv):
            value = argv[i + 1]
        elif tok.startswith("--num-threads="):
            value = tok.split("=", 1)[1]
    try:
        n = int(value) if value is not None else 1
    except ValueError:
        return 1
    return n if n >= 1 else 1


# Thread pinning: must happen before numpy is imported, otherwise the BLAS
# library has already read its thread count and later changes are ignored.
_NUMPY_PRELOADED = "numpy" in sys.modules
for _name in THREAD_ENV_VARS:
    os.environ[_name] = str(_early_num_threads(sys.argv[1:]))

import numpy as np  # noqa: E402
import psutil  # noqa: E402

CAVEAT = ("Results are preliminary, single-VM, 1 physical core, "
          "not representative of production edge/cloud.")
MODES = ("dequant_only", "forward_pass", "predict_proba",
         "dequant_plus_forward", "full")
MODE_DEFINITIONS = {
    "dequant_only": "NpzMLP.dequantize() on a fresh instance (instance built outside the timer; wall_time_s therefore includes one untimed NpzMLP(path) construction per iteration)",
    "forward_pass": "NpzMLP.forward(X) on an instance dequantized once before timing",
    "predict_proba": "NpzMLP.predict_proba(X) on a pre-dequantized instance",
    "dequant_plus_forward": "dequantize() + forward(X) in one timed block; quantized weights already in memory (untimed NpzMLP(path) per iteration is included in wall_time_s)",
    "full": "NpzMLP(path) (disk read) + dequantize() + forward(X) in one timed block; cold only if the OS page cache is empty",
}
METRIC_NOTES = {
    "throughput_per_s": "timed iterations / wall-clock seconds of the whole loop; includes harness overhead and any untimed per-iteration setup",
    "throughput_busy_per_s": "iterations / seconds spent inside timed blocks only (= 1/mean latency); excludes harness overhead, so it is an upper bound on achievable rate",
    "samples_per_s, samples_busy_per_s": "the two throughput definitions above multiplied by batch_size",
    "disk_read_bytes": "0 means the page cache served the read; a real cold start requires dropping caches (root)",
    "rss": "rss_before_mb is taken after the input pool is built, so the pool (config.pool_bytes) is part of the baseline; growth during the run is rss_peak_mb - rss_before_mb",
}
ALLOWED_BATCH_SIZES = (1, 8, 32, 128, 512)
MAX_POOL_ROWS = 100_000
SAMPLE_INTERVAL_S = 0.1
STAT_KEYS = ("mean", "median", "p50", "p90", "p95", "p99", "min", "max", "std")


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def log(msg):
    """Write a diagnostic line to stderr (stdout is reserved for the summary)."""
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()


def to_jsonable(obj):
    """Make obj JSON-safe: numpy -> python, non-finite float -> None."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, np.generic):
        return to_jsonable(obj.item())
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    if isinstance(obj, Path):
        return str(obj)
    return obj


def mb(n_bytes):
    """Bytes -> MiB."""
    return float(n_bytes) / (1024.0 * 1024.0)


def fmt(v, digits=6):
    """Format a possibly-None number for the stdout summary."""
    return "n/a" if v is None else f"{v:.{digits}f}"


def get_cpu_model():
    """CPU model string from /proc/cpuinfo, falling back to platform."""
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as fh:
            for line in fh:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def get_git_commit():
    """Current git commit or None (best effort)."""
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"],
                             cwd=Path(__file__).resolve().parent,
                             capture_output=True, text=True, timeout=5, check=True)
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001 - optional metadata
        return None


def numpy_blas_info():
    """BLAS build info from np.show_config(mode='dicts'); never raises."""
    try:
        cfg = np.show_config(mode="dicts")
        deps = cfg.get("Build Dependencies", {})
        return {"blas": deps.get("blas"), "lapack": deps.get("lapack")}
    except Exception as exc:  # noqa: BLE001 - older numpy has no mode="dicts"
        return {"error": f"{type(exc).__name__}: {exc}"}


def format_warnings(caught):
    """De-duplicated 'Category: message' strings from recorded warnings."""
    seen, out = set(), []
    for w in caught:
        text = f"{w.category.__name__}: {w.message}"
        if text not in seen:
            seen.add(text)
            out.append(text)
    return out


def load_npz_class():
    """Import NpzMLP from the project-root npz_model.py."""
    from npz_model import NpzMLP  # noqa: WPS433 - local project module
    return NpzMLP


# --------------------------------------------------------------------------- #
# Input data
# --------------------------------------------------------------------------- #
def _cell_is_numeric(cell):
    """True for numeric (or empty) CSV cells; used to detect a header row."""
    cell = cell.strip()
    if cell == "":
        return True
    try:
        float(cell)
        return True
    except ValueError:
        return False


def read_csv_features(path, input_dim, max_rows):
    """Load feature rows from a CSV as float32, dropping a Label column.

    Reads at most max_rows data rows. Non-finite values become 0.0 via
    nan_to_num. Returns (array[float32], info dict, list of warning strings).
    """
    notes = []
    rows = []
    # latin-1 maps every byte to a character, so decoding can never fail or
    # silently substitute (CICIDS2017 labels contain a non-UTF-8 0x96 byte).
    # A UTF-8 BOM, if present, shows up as 3 latin-1 chars and is stripped below.
    with Path(path).open("r", newline="", encoding="latin-1") as fh:
        reader = csv.reader(fh)
        first = next(reader, None)
        if first is None:
            raise ValueError(f"{path} is empty")
        if first:
            first[0] = first[0].replace("\u00ef\u00bb\u00bf", "")
        header_detected = not all(_cell_is_numeric(c) for c in first)
        names = [c.strip() for c in first] if header_detected else None
        keep = list(range(len(first)))
        label_dropped = False
        if names is not None:
            drop = {i for i, n in enumerate(names) if n.lower() in ("label", "labels")}
            label_dropped = bool(drop)
            keep = [i for i in keep if i not in drop]
        if len(keep) < input_dim:
            raise ValueError(f"{path}: {len(keep)} feature columns < --input-dim {input_dim}")
        if len(keep) > input_dim:
            notes.append(f"CSV has {len(keep)} feature columns; using the first {input_dim}")
            keep = keep[:input_dim]

        bad = [0]  # unparsable / missing cells, counted instead of hidden

        def parse(row):
            out = []
            for i in keep:
                if i < len(row):
                    try:
                        out.append(float(row[i]))
                        continue
                    except ValueError:
                        pass
                bad[0] += 1
                out.append(float("nan"))
            return out

        if not header_detected:
            rows.append(parse(first))
        for row in reader:
            if len(rows) >= max_rows:
                break
            if row:
                rows.append(parse(row))
    if not rows:
        raise ValueError(f"{path}: no data rows")
    arr = np.asarray(rows, dtype=np.float64).astype(np.float32)  # cast first, as specified
    n_bad = int((~np.isfinite(arr)).sum())
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if n_bad:
        notes.append(f"{n_bad} non-finite/unparsable feature values replaced by 0.0 "
                     f"({bad[0]} were unparsable cells)")
    info = {"csv_path": str(path), "encoding": "latin-1", "unparsable_cells": bad[0], "rows_loaded": int(arr.shape[0]),
            "feature_columns_used": int(arr.shape[1]), "header_detected": header_detected,
            "label_column_dropped": label_dropped, "nonfinite_values_replaced": n_bad}
    return arr, info, notes


def build_batches(batch_size, n_calls, input_dim, seed, csv_rows):
    """Pre-build input batches OUTSIDE the timed region (no allocation inside).

    Returns a list of C-contiguous float32 views of shape (batch_size, input_dim);
    the timed loop cycles through them.
    """
    n_batches = max(1, min(1000, n_calls, MAX_POOL_ROWS // batch_size))
    total = n_batches * batch_size
    if csv_rows is None:
        pool = np.random.RandomState(seed).randn(total, input_dim).astype(np.float32)
    else:
        pool = csv_rows[np.arange(total) % len(csv_rows)]
    pool = np.ascontiguousarray(pool, dtype=np.float32)
    return [pool[i * batch_size:(i + 1) * batch_size] for i in range(n_batches)]


# --------------------------------------------------------------------------- #
# Resource sampling
# --------------------------------------------------------------------------- #
class ResourceSampler:
    """Background 100 ms sampler for process RSS, process CPU % and system CPU %.

    Runs only around the timed loop. Caveats: on a 1-vCPU VM the sampler thread
    competes with the benchmark (negligible at 10 Hz). If the timed window is
    shorter than one interval there are no ticks, and CPU figures fall back to a
    single average over the whole window (reported in cpu_source). That fallback
    is derived from cpu_times() (~10 ms resolution), so for windows of a few ms it
    can read 0% or >100%: treat it as unreliable and prefer longer runs.
    """

    def __init__(self, interval=SAMPLE_INTERVAL_S):
        self.interval = interval
        self._proc = psutil.Process(os.getpid())
        self._stop = threading.Event()
        self._thread = None
        self._rss, self._pcpu, self._scpu = [], [], []

    @property
    def started(self):
        """True once start() has been called."""
        return self._thread is not None

    def start(self):
        """Prime psutil's CPU counters, record the 'before' values, start sampling."""
        self._proc.cpu_percent(interval=None)
        psutil.cpu_percent(interval=None)
        t = self._proc.cpu_times()
        self._cpu0 = t.user + t.system
        self._wall0 = time.perf_counter()
        self._rss_before = self._proc.memory_info().rss
        self._avail_before = psutil.virtual_memory().available
        try:
            self._io0 = self._proc.io_counters().read_bytes
        except (AttributeError, psutil.Error):
            self._io0 = None
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while not self._stop.wait(self.interval):
            self._rss.append(self._proc.memory_info().rss)
            self._pcpu.append(self._proc.cpu_percent(interval=None))
            self._scpu.append(psutil.cpu_percent(interval=None))

    def stop(self):
        """Stop sampling and return the aggregated resource dict."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        wall = time.perf_counter() - self._wall0
        t = self._proc.cpu_times()
        rss_after = self._proc.memory_info().rss
        avail_after = psutil.virtual_memory().available
        try:
            disk = self._proc.io_counters().read_bytes - self._io0
        except (AttributeError, psutil.Error, TypeError):
            disk = None
        n = len(self._pcpu)
        if n:
            pm, px = float(np.mean(self._pcpu)), float(np.max(self._pcpu))
            sm, sx = float(np.mean(self._scpu)), float(np.max(self._scpu))
            source = "100ms_samples"
        else:
            used = (t.user + t.system) - self._cpu0
            pm = px = 100.0 * used / wall if wall > 0 else None
            sm = sx = float(psutil.cpu_percent(interval=None))
            source = "window_average"
        rss_all = [self._rss_before, rss_after] + self._rss
        return {
            "rss_before_mb": mb(self._rss_before),
            "rss_after_mb": mb(rss_after),
            "rss_peak_mb": mb(max(rss_all)),
            "rss_mean_mb": mb(float(np.mean(self._rss if self._rss else rss_all))),
            "proc_cpu_pct_mean": pm, "proc_cpu_pct_max": px,
            "sys_cpu_pct_mean": sm, "sys_cpu_pct_max": sx,
            "cpu_source": source, "n_samples": n,
            "ram_available_before_mb": mb(self._avail_before),
            "ram_available_after_mb": mb(avail_after),
            # read_bytes counts real storage reads: 0 means the page cache served it.
            "disk_read_bytes": disk,
        }


# --------------------------------------------------------------------------- #
# Measurement
# --------------------------------------------------------------------------- #
def make_step(mode, npz_cls, path, batches):
    """Build step(i) -> (elapsed_ns, output) for one inference mode.

    Only the code between the two perf_counter_ns() calls is timed. Anything
    that must NOT be paid by the mode (e.g. building a fresh instance for
    dequant_only) happens before t0. This is what separates the dequantization
    cost from the matmul cost.
    """
    n = len(batches)
    pc = time.perf_counter_ns
    path = str(path)

    if mode in ("forward_pass", "predict_proba"):
        model = npz_cls(path)
        model.dequantize()  # once, before timing: weights are float from here on
        call = model.forward if mode == "forward_pass" else model.predict_proba

        def step(i):
            x = batches[i % n]
            t0 = pc()
            out = call(x)
            t1 = pc()
            return t1 - t0, out
    elif mode == "dequant_only":
        def step(i):
            m = npz_cls(path)  # fresh instance, untimed (includes the file read)
            t0 = pc()
            m.dequantize()
            t1 = pc()
            return t1 - t0, None
    elif mode == "dequant_plus_forward":
        def step(i):
            m = npz_cls(path)  # quantized weights loaded into memory, untimed
            x = batches[i % n]
            t0 = pc()
            m.dequantize()
            out = m.forward(x)
            t1 = pc()
            return t1 - t0, out
    elif mode == "full":
        def step(i):
            x = batches[i % n]
            t0 = pc()
            m = npz_cls(path)  # disk read + parse INSIDE the timer
            m.dequantize()
            out = m.forward(x)
            t1 = pc()
            return t1 - t0, out
    else:
        raise ValueError(f"unknown mode {mode}")
    return step


def latency_stats(lat_ns):
    """Latency statistics in ms plus throughput (timed calls per second)."""
    keys = STAT_KEYS + ("throughput_per_s", "samples_per_s",
                        "throughput_busy_per_s", "samples_busy_per_s")
    if lat_ns.size == 0:
        return {k: None for k in keys}
    ms = lat_ns.astype(np.float64) / 1e6
    p50, p90, p95, p99 = (float(v) for v in np.percentile(ms, [50, 90, 95, 99]))
    busy_s = float(ms.sum()) / 1e3  # time actually inside the timed blocks
    return {
        "mean": float(ms.mean()), "median": p50, "p50": p50, "p90": p90,
        "p95": p95, "p99": p99, "min": float(ms.min()), "max": float(ms.max()),
        "std": float(ms.std()),  # population std (ddof=0)
        # Busy-time rate (= 1/mean latency): excludes loop/harness overhead.
        "throughput_busy_per_s": (lat_ns.size / busy_s) if busy_s > 0 else None,
        # Wall-clock rates need the loop's wall time and batch size: set by run_mode().
        "throughput_per_s": None,
        "samples_per_s": None,
        "samples_busy_per_s": None,
    }


def run_mode(mode, npz_cls, path, batches, batch_size, warmup, iterations, input_source):
    """Warm up, then time `iterations` calls of `mode`; returns a run dict."""
    step = make_step(mode, npz_cls, path, batches)
    lat = np.empty(iterations, dtype=np.int64)
    done, interrupted, out = 0, False, None
    sampler = ResourceSampler()
    resources, wall = {}, 0.0
    process_threads = None  # set after warmup; stays None if interrupted during warmup
    try:
        for i in range(warmup):  # unmeasured; also builds caches / spins up BLAS
            step(i)
        # OS threads in this process after warmup: shows whether BLAS actually
        # spawned workers (1 thread requested -> expect just the main thread).
        process_threads = psutil.Process().num_threads()
        sampler.start()
        w0 = time.perf_counter()
        try:
            for i in range(iterations):
                dt, out = step(i)
                lat[i] = dt
                done = i + 1
        finally:
            wall = time.perf_counter() - w0  # also valid for a partial run
    except KeyboardInterrupt:
        interrupted = True
        log("Interrupted: keeping partial results.")
    if sampler.started:
        resources = sampler.stop()
    if out is not None and not np.all(np.isfinite(np.asarray(out, dtype=np.float64))):
        log(f"WARNING: non-finite values in {mode} output")
    stats = latency_stats(lat[:done])
    if stats["throughput_busy_per_s"] is not None:
        stats["samples_busy_per_s"] = stats["throughput_busy_per_s"] * batch_size
    if done and wall > 0:
        stats["throughput_per_s"] = done / wall
        stats["samples_per_s"] = done * batch_size / wall
    stats["wall_time_s"] = wall
    return {
        "config": {"inference_method": mode, "batch_size": batch_size,
                   "iterations": iterations, "iterations_completed": done,
                   "warmup": warmup, "input_source": input_source,
                   "input_shape": [batch_size, int(batches[0].shape[1])],
                   "pool_batches": len(batches),
                   "pool_bytes": int(sum(b.nbytes for b in batches))},
        "latency_ms": stats,
        "resources": resources,
        "process_threads_after_warmup": process_threads,
        "status": "interrupted" if interrupted else "complete",
    }


def print_run(run):
    """Human-readable block: one line per statistic."""
    cfg, lat, res = run["config"], run["latency_ms"], run["resources"]
    print(f"=== {cfg['inference_method']} | batch={cfg['batch_size']} | "
          f"iterations={cfg['iterations_completed']}/{cfg['iterations']} | "
          f"input={cfg['input_source']} {tuple(cfg['input_shape'])} | {run['status']} ===")
    for k in STAT_KEYS:
        print(f"  latency_{k + '_ms':<10}: {fmt(lat[k])}")
    print(f"  throughput_per_s   : {fmt(lat['throughput_per_s'], 1)}  (wall-clock)")
    print(f"  throughput_busy/s  : {fmt(lat['throughput_busy_per_s'], 1)}  (timed blocks only)")
    print(f"  samples_per_s      : {fmt(lat['samples_per_s'], 1)}  (wall-clock)")
    print(f"  samples_busy/s     : {fmt(lat['samples_busy_per_s'], 1)}  (timed blocks only)")
    if res:
        print(f"  rss_peak_mb        : {fmt(res['rss_peak_mb'], 2)}")
        print(f"  rss_mean_mb        : {fmt(res['rss_mean_mb'], 2)}")
        print(f"  proc_cpu_pct       : mean {fmt(res['proc_cpu_pct_mean'], 1)} / "
              f"max {fmt(res['proc_cpu_pct_max'], 1)} ({res['cpu_source']})")
        print(f"  sys_cpu_pct        : mean {fmt(res['sys_cpu_pct_mean'], 1)} / "
              f"max {fmt(res['sys_cpu_pct_max'], 1)}")
        print(f"  ram_available_mb   : {fmt(res['ram_available_before_mb'], 1)} -> "
              f"{fmt(res['ram_available_after_mb'], 1)}")
        print(f"  disk_read_bytes    : {res['disk_read_bytes']}")
    print()


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #
def self_test(npz_cls, path, input_dim):
    """Smoke test: build, dequantize, forward + predict_proba on synthetic input."""
    ok = True
    for dtype in (np.float64, np.float32):
        try:
            x = np.random.RandomState(0).randn(4, input_dim).astype(dtype)
            model = npz_cls(str(path))
            model.dequantize()
            fwd = np.asarray(model.forward(x))
            proba = np.asarray(model.predict_proba(x))
            checks = [
                (f"forward shape {fwd.shape} == (4, 1)", fwd.shape == (4, 1)),
                (f"forward finite", bool(np.all(np.isfinite(fwd)))),
                (f"predict_proba shape {proba.shape} == (4, 2)", proba.shape == (4, 2)),
                (f"predict_proba finite", bool(np.all(np.isfinite(proba)))),
            ]
        except Exception as exc:  # noqa: BLE001 - report any failure as FAIL
            checks = [(f"exception {type(exc).__name__}: {exc}", False)]
        for label, passed in checks:
            print(f"  [{np.dtype(dtype).name}] {'ok  ' if passed else 'FAIL'} {label}")
            ok = ok and passed
    print("PASS" if ok else "FAIL")
    return ok


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def build_parser():
    """CLI (abbreviations off so the early --num-threads scan matches argparse)."""
    p = argparse.ArgumentParser(
        description="Benchmark quantized NpzMLP detectors (dequant vs forward cost).",
        allow_abbrev=False)
    p.add_argument("--model", required=True, help="Path to the .npz model.")
    p.add_argument("--num-threads", type=int, default=1,
                   help="Sets OMP/MKL/OPENBLAS/NUMEXPR threads (process start only).")
    p.add_argument("--iterations", type=int, default=1000)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--batch-size", type=int, nargs="+", default=[1],
                   help=f"One or more of {list(ALLOWED_BATCH_SIZES)}.")
    p.add_argument("--inference-method", nargs="+", default=["forward_pass"],
                   choices=list(MODES) + ["all"])
    p.add_argument("--input-dim", type=int, default=77)
    p.add_argument("--label", default="")
    p.add_argument("--input-csv", default=None,
                   help="CSV of features (Label column dropped). Omit for synthetic input.")
    p.add_argument("--output", default=None, help="Output JSON path.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--trained-sklearn-version", default="1.9.1",
                   help="sklearn version the reference models were trained with.")
    p.add_argument("--self-test", action="store_true",
                   help="Synthetic smoke test of the model, then exit.")
    return p


def main(argv=None):
    """Entry point; returns the process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    for b in args.batch_size:
        if b not in ALLOWED_BATCH_SIZES:
            parser.error(f"--batch-size values must be in {list(ALLOWED_BATCH_SIZES)}")
    if args.iterations < 1 or args.warmup < 0 or args.num_threads < 1 or args.input_dim < 1:
        parser.error("--iterations/--num-threads/--input-dim must be >= 1 and --warmup >= 0")
    model_path = Path(args.model)
    if not model_path.is_file():
        log(f"ERROR: model file not found: {model_path}")
        return 2
    modes = list(MODES) if "all" in args.inference_method else list(dict.fromkeys(args.inference_method))
    batch_sizes = list(dict.fromkeys(args.batch_size))
    if args.iterations < 10:
        log("WARNING: --iterations < 10: percentiles will be very noisy.")
    np.random.seed(args.seed)

    notes = []
    runs = []
    status_interrupted = False
    # Record (never swallow) warnings such as sklearn's InconsistentVersionWarning.
    # "default" = once per location, so a repeating warning cannot flood the list
    # or add recording cost inside the timed loops.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("default")
        try:
            npz_cls = load_npz_class()
        except ImportError as exc:
            log(f"ERROR: cannot import NpzMLP from npz_model.py: {exc}")
            return 2

        if args.self_test:
            print(f"benchmark_npz.py self-test | model={model_path}")
            ok = self_test(npz_cls, model_path, args.input_dim)
            for text in format_warnings(caught):
                log(f"WARNING: {text}")
            return 0 if ok else 1

        # ---- model probe (load time, disk I/O, metadata) ------------------ #
        proc = psutil.Process(os.getpid())
        rss_before_load = proc.memory_info().rss  # cold: before model and before input data
        try:
            io0 = proc.io_counters().read_bytes
        except (AttributeError, psutil.Error):
            io0 = None
        t0 = time.perf_counter()
        probe = npz_cls(str(model_path))
        load_time_ms = (time.perf_counter() - t0) * 1e3
        rss_after_load = proc.memory_info().rss
        try:
            load_disk = proc.io_counters().read_bytes - io0
        except (AttributeError, psutil.Error, TypeError):
            load_disk = None
        t0 = time.perf_counter()
        probe.dequantize()
        first_dequant_ms = (time.perf_counter() - t0) * 1e3
        rss_after_dequant = proc.memory_info().rss
        try:
            model_info = probe.info()
        except Exception as exc:  # noqa: BLE001 - metadata must not abort the run
            model_info = {"error": f"info() failed: {type(exc).__name__}: {exc}"}
        del probe

        # ---- input data -------------------------------------------------- #
        csv_rows, input_info = None, {"source": "synthetic", "seed": args.seed}
        if args.input_csv:
            csv_path = Path(args.input_csv)
            if not csv_path.is_file():
                log(f"ERROR: --input-csv not found: {csv_path}")
                return 2
            need = max(max(1, min(1000, args.warmup + args.iterations, MAX_POOL_ROWS // b)) * b
                       for b in batch_sizes)
            try:
                csv_rows, csv_info, csv_notes = read_csv_features(csv_path, args.input_dim, need)
            except ValueError as exc:
                log(f"ERROR: {exc}")
                return 2
            input_info = {"source": "csv", **csv_info}
            notes += csv_notes
        input_source = input_info["source"]

        # ---- benchmark loops ---------------------------------------------- #
        try:
            for bs in batch_sizes:
                batches = build_batches(bs, args.warmup + args.iterations,
                                        args.input_dim, args.seed, csv_rows)
                for mode in modes:
                    log(f"running {mode} | batch={bs} ...")
                    run = run_mode(mode, npz_cls, model_path, batches, bs,
                                   args.warmup, args.iterations, input_source)
                    runs.append(run)
                    print_run(run)
                    if run["status"] == "interrupted":
                        status_interrupted = True
                        raise KeyboardInterrupt
        except KeyboardInterrupt:
            status_interrupted = True

    # sklearn is imported only now: importing it earlier would add tens of MB to
    # RSS and distort the memory numbers of a model that does not need sklearn.
    sk_mod = sys.modules.get("sklearn")
    if sk_mod is None:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                import sklearn as sk_mod  # noqa: WPS433
        except ImportError:
            sk_mod = None
    sk_version = getattr(sk_mod, "__version__", None)

    warn_list = format_warnings(caught)
    if sk_version and sk_version != args.trained_sklearn_version:
        warn_list.append(f"sklearn runtime {sk_version} != training {args.trained_sklearn_version} (accepted)")
    for text in warn_list:
        log(f"WARNING: {text}")
    if _NUMPY_PRELOADED:
        warn_list.append("numpy was imported before thread env vars were set")

    first_threads = next((r["process_threads_after_warmup"] for r in runs
                          if r["process_threads_after_warmup"] is not None), None)
    ts = datetime.now(timezone.utc).replace(microsecond=0)
    result = {
        "script": "benchmark_npz.py",
        "caveat": CAVEAT,
        "timestamp": ts.isoformat(),
        "git_commit": get_git_commit(),
        "status": "interrupted" if status_interrupted else "complete",
        "model_path": str(model_path),
        "model_label": args.label,
        "model_info": model_info,
        "model_load": {"load_time_ms": load_time_ms, "first_dequantize_ms": first_dequant_ms,
                       "disk_read_bytes": load_disk,
                       "disk_read_bytes_note": METRIC_NOTES["disk_read_bytes"],
                       "rss_before_load_mb": mb(rss_before_load),
                       "rss_after_load_mb": mb(rss_after_load),
                       "rss_after_first_dequantize_mb": mb(rss_after_dequant),
                       "file_size_bytes": os.path.getsize(model_path)},
        "host": {
            "hostname": platform.node(), "cpu_model": get_cpu_model(),
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cores": psutil.cpu_count(logical=True),
            "ram_total_mb": mb(psutil.virtual_memory().total),
            "os": platform.platform(), "python": platform.python_version(),
            "numpy": np.__version__, "sklearn": sk_version,
        },
        "threads": {
            "requested": args.num_threads,
            "env": {k: os.environ.get(k) for k in THREAD_ENV_VARS},
            "numpy_actual": {
                "blas_build": numpy_blas_info(),
                "process_os_threads_after_warmup": first_threads,
                "note": "measured after warmup, before the sampler starts: main thread plus any BLAS worker threads",
            },
        },
        "input": input_info,
        "mode_definitions": {m: MODE_DEFINITIONS[m] for m in modes},
        "metric_notes": METRIC_NOTES,
        "runs": runs,
        "notes": notes,
        "warnings": warn_list,
    }
    if len(runs) == 1:  # flat layout for the single-run case
        result["config"] = runs[0]["config"]
        result["latency_ms"] = runs[0]["latency_ms"]
        result["resources"] = runs[0]["resources"]

    out_path = Path(args.output) if args.output else Path(
        f"results_{model_path.stem}_{ts.strftime('%Y%m%dT%H%M%SZ')}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(to_jsonable(result), fh, indent=2, sort_keys=True, allow_nan=False)
    print(f"CAVEAT: {CAVEAT}")
    print(f"wrote {out_path}")
    return 130 if status_interrupted else 0


if __name__ == "__main__":
    sys.exit(main())
