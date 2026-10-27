#!/usr/bin/env python3
"""Inference latency / resource benchmark for scikit-learn MLPClassifier detectors.

PRO-IDS (edge-cloud adaptive offloading for network intrusion detection), E1
milestone. Measures LOCAL inference only (one forward pass on an already
prepared feature vector): no network, no queueing, no preprocessing.

Inference methods
-----------------
``predict_proba``  Public API. Includes sklearn input validation
                   (``check_array``, dtype / feature-count checks) and the
                   ``[1-p, p]`` stacking. Closest to what production code calls.
``predict``        Same validation overhead, returns class labels.
``forward_pass``   ``model._forward_pass_fast(X, check_input=False)``: skips
                   validation, so it is the closest to a "pure" forward pass.
                   Falls back to an equivalent NumPy forward pass if the private
                   method does not exist in the installed sklearn.

Thread control
--------------
BLAS / OpenMP thread counts are read when NumPy is first imported, so
``--num-threads`` is scanned from ``sys.argv`` and exported to the environment
BEFORE numpy / sklearn are imported. ``threadpoolctl.threadpool_limits`` is
applied again around the benchmark and ``threadpool_info()`` is recorded.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import platform
import random
import re
import socket
import subprocess
import sys
import threading
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

THREAD_ENV_VARS: tuple[str, ...] = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def parse_early_num_threads(argv: Sequence[str]) -> int:
    """Scan ``argv`` for ``--num-threads`` without importing numpy/sklearn.

    Args:
        argv: Command-line tokens (without the program name).

    Returns:
        The requested thread count, or 1 if absent or invalid (argparse will
        report invalid values properly later).
    """
    value: Optional[str] = None
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


def configure_thread_env(num_threads: int) -> dict[str, str]:
    """Export BLAS/OpenMP thread env vars. Must run before importing numpy.

    Args:
        num_threads: Thread count to export.

    Returns:
        Mapping of env var name to the value that is now set.
    """
    for name in THREAD_ENV_VARS:
        os.environ[name] = str(num_threads)
    return {name: os.environ[name] for name in THREAD_ENV_VARS}


_NUMPY_ALREADY_IMPORTED = "numpy" in sys.modules
_ENV_THREADS = configure_thread_env(parse_early_num_threads(sys.argv[1:]))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import psutil  # noqa: E402
import sklearn  # noqa: E402
from sklearn.exceptions import ConvergenceWarning  # noqa: E402
from sklearn.neural_network import MLPClassifier  # noqa: E402
from threadpoolctl import threadpool_info, threadpool_limits  # noqa: E402

LOG = logging.getLogger("benchmark_sklearn")

EXPECTED_PARAM_COUNT: dict[str, int] = {"ad1": 3041, "ad3": 214529}
DEFAULT_MODEL_PATH: dict[str, str] = {
    "ad1": "models/detector/AD1v2_model.joblib",
    "ad3": "models/detector/AD3_model.joblib",
}
MAX_POOL_ROWS = 100_000
STAT_KEYS: tuple[str, ...] = ("mean", "std", "min", "max", "p50", "p90", "p95", "p99")
CSV_FIELDS: tuple[str, ...] = (
    "run_id", "framework", "inference_method", "model", "label", "hostname",
    "timestamp_utc", "batch_size", "iterations", "num_threads", "load_time_sec",
    "latency_mean_ms", "latency_p50_ms", "latency_p95_ms", "latency_p99_ms",
    "throughput_sps", "cpu_percent_overall", "ram_rss_peak_mb", "param_count",
    "size_bytes", "git_commit",
)


# --------------------------------------------------------------------------- #
# Generic helpers
# --------------------------------------------------------------------------- #
def sanitize(obj: Any) -> Any:
    """Convert to JSON-safe types; non-finite floats become ``None``."""
    if isinstance(obj, dict):
        return {str(k): sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return sanitize(obj.tolist())
    if isinstance(obj, np.generic):
        return sanitize(obj.item())
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, Path):
        return str(obj)
    return obj


def mb(n_bytes: float) -> float:
    """Bytes to MiB."""
    return float(n_bytes) / (1024.0 * 1024.0)


def get_git_commit() -> Optional[str]:
    """Return the current git commit hash, or ``None`` on any failure."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True, text=True, timeout=5, check=True,
        )
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001 - best effort by design
        return None


def get_cpu_model() -> str:
    """Best-effort CPU model string."""
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as fh:
            for line in fh:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def collect_metadata(env_threads: dict[str, str]) -> dict[str, Any]:
    """Collect host / library metadata."""
    return {
        "hostname": socket.gethostname(),
        "os": platform.platform(),
        "kernel": platform.release(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
        "joblib": joblib.__version__,
        "psutil": psutil.__version__,
        "cpu_model": get_cpu_model(),
        "cpu_physical_cores": psutil.cpu_count(logical=False),
        "cpu_logical_cores": psutil.cpu_count(logical=True),
        "ram_total_gb": round(psutil.virtual_memory().total / 1024 ** 3, 3),
        "env_threads": env_threads,
    }


def read_cpu_freq_mhz() -> Optional[float]:
    """Current CPU frequency in MHz, or ``None``."""
    try:
        freq = psutil.cpu_freq()
        return float(freq.current) if freq else None
    except Exception:  # noqa: BLE001
        return None


def read_cpu_temp_c() -> Optional[float]:
    """First ``coretemp`` / ``k10temp`` reading in deg C, or ``None``."""
    try:
        sensors = psutil.sensors_temperatures()
    except (AttributeError, OSError):
        return None
    for key in ("coretemp", "k10temp"):
        entries = sensors.get(key)
        if entries:
            return float(entries[0].current)
    return None


def read_rapl() -> Optional[dict[str, tuple[int, int]]]:
    """Read Intel RAPL package energy counters (uJ). ``None`` if unreadable."""
    base = Path("/sys/class/powercap")
    out: dict[str, tuple[int, int]] = {}
    try:
        dirs = [d for d in base.glob("intel-rapl:*") if d.name.count(":") == 1]
    except OSError:
        return None
    for d in dirs:
        try:
            energy = int((d / "energy_uj").read_text())
            max_range = int((d / "max_energy_range_uj").read_text())
        except (OSError, ValueError):
            continue
        out[d.name] = (energy, max_range)
    return out or None


def rapl_power_w(
    before: Optional[dict[str, tuple[int, int]]],
    after: Optional[dict[str, tuple[int, int]]],
    wall_sec: float,
) -> Optional[float]:
    """Average package power over the interval from two RAPL snapshots."""
    if not before or not after or wall_sec <= 0:
        return None
    total_uj = 0
    for name, (e0, max_range) in before.items():
        if name not in after:
            continue
        delta = after[name][0] - e0
        if delta < 0:  # counter wrapped
            delta += max_range
        total_uj += delta
    return total_uj / 1e6 / wall_sec


# --------------------------------------------------------------------------- #
# Resource sampling
# --------------------------------------------------------------------------- #
class ResourceSampler:
    """Background sampler for system CPU % and process RSS.

    Samples every ``interval`` seconds. If the measured window is shorter than
    one interval (common for the tiny AD1 model), the CPU figures fall back to
    a single reading averaged over the whole window.
    """

    def __init__(self, interval: float = 0.1) -> None:
        self.interval = interval
        self._proc = psutil.Process(os.getpid())
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._overall: list[float] = []
        self._percore: list[list[float]] = []
        self._rss_peak = 0
        self._t0 = 0.0
        self._cpu0: Any = None

    @property
    def started(self) -> bool:
        """Whether ``start`` has been called."""
        return self._thread is not None

    def start(self) -> None:
        """Prime psutil counters and start the sampling thread."""
        psutil.cpu_percent(interval=None)
        psutil.cpu_percent(interval=None, percpu=True)
        self._cpu0 = self._proc.cpu_times()
        self._rss_peak = self._proc.memory_info().rss
        self._t0 = time.perf_counter()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            self._overall.append(psutil.cpu_percent(interval=None))
            self._percore.append(psutil.cpu_percent(interval=None, percpu=True))
            self._rss_peak = max(self._rss_peak, self._proc.memory_info().rss)

    def stop(self) -> dict[str, Any]:
        """Stop sampling and return aggregated numbers."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        wall = time.perf_counter() - self._t0
        cpu1 = self._proc.cpu_times()
        rss_now = self._proc.memory_info().rss
        self._rss_peak = max(self._rss_peak, rss_now)
        n = len(self._overall)
        if n:
            overall = float(np.mean(self._overall))
            percore = np.mean(np.asarray(self._percore), axis=0).tolist()
        else:
            overall = psutil.cpu_percent(interval=None)
            percore = psutil.cpu_percent(interval=None, percpu=True)
        proc_cpu = None
        if wall > 0 and self._cpu0 is not None:
            used = (cpu1.user + cpu1.system) - (self._cpu0.user + self._cpu0.system)
            proc_cpu = 100.0 * used / wall
        return {
            "cpu_percent_overall": overall,
            "cpu_percent_per_core": percore,
            "cpu_percent_process": proc_cpu,
            "cpu_samples": n,
            "ram_rss_peak_mb": mb(self._rss_peak),
            "ram_rss_after_mb": mb(rss_now),
        }


# --------------------------------------------------------------------------- #
# Model loading / inference
# --------------------------------------------------------------------------- #
def load_model(path: Path) -> tuple[MLPClassifier, dict[str, Any]]:
    """Load a ``.joblib`` MLPClassifier and time the load.

    Args:
        path: Path to the model file.

    Returns:
        ``(model, load_info)`` where ``load_info`` holds load time, file size,
        disk I/O deltas and any warnings raised while unpickling (e.g. the
        sklearn ``InconsistentVersionWarning``).

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        TypeError: If the object is not an ``MLPClassifier``.
    """
    if not path.is_file():
        raise FileNotFoundError(f"model file not found: {path}")
    proc = psutil.Process(os.getpid())
    try:
        io0 = proc.io_counters()
    except (AttributeError, psutil.Error):
        io0 = None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        t0 = time.perf_counter()
        model = joblib.load(path)
        load_time = time.perf_counter() - t0
    try:
        io1 = proc.io_counters()
    except (AttributeError, psutil.Error):
        io1 = None
    if not isinstance(model, MLPClassifier):
        raise TypeError(f"expected MLPClassifier, got {type(model).__name__}")
    return model, {
        "load_time_sec": load_time,
        "size_bytes": os.path.getsize(path),
        "disk_read_bytes": (io1.read_bytes - io0.read_bytes) if io0 and io1 else None,
        "disk_write_bytes": (io1.write_bytes - io0.write_bytes) if io0 and io1 else None,
        "warnings": [f"{w.category.__name__}: {w.message}" for w in caught],
    }


def build_self_test_model(seed: int, n_features: int = 77) -> MLPClassifier:
    """Fit a tiny MLP on random data so ``coefs_`` exists (no files needed)."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((200, n_features))
    y = (x[:, 0] > 0).astype(int)
    model = MLPClassifier(hidden_layer_sizes=(4,), max_iter=1, random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(x, y)
    return model


def describe_model(model: MLPClassifier) -> dict[str, Any]:
    """Architecture facts recorded in the JSON."""
    param_count = sum(c.size for c in model.coefs_) + sum(b.size for b in model.intercepts_)
    return {
        "n_layers": int(model.n_layers_),
        "hidden_layer_sizes": list(np.atleast_1d(model.hidden_layer_sizes)),
        "activation": model.activation,
        "out_activation": model.out_activation_,
        "coefs_shapes": [list(c.shape) for c in model.coefs_],
        "intercepts_shapes": [list(b.shape) for b in model.intercepts_],
        "n_features_in": int(model.n_features_in_),
        "classes": model.classes_.tolist(),
        "param_count": int(param_count),
    }


def _numpy_forward(model: MLPClassifier, x: np.ndarray) -> np.ndarray:
    """Pure NumPy forward pass (no validation); fallback for ``forward_pass``."""
    acts: dict[str, Callable[[np.ndarray], np.ndarray]] = {
        "relu": lambda a: np.maximum(a, 0.0),
        "tanh": np.tanh,
        "logistic": lambda a: 1.0 / (1.0 + np.exp(-a)),
        "identity": lambda a: a,
    }
    hidden = acts[model.activation]
    out_act = acts.get(model.out_activation_, acts["identity"])
    a = x
    last = len(model.coefs_) - 1
    for i, (w, b) in enumerate(zip(model.coefs_, model.intercepts_)):
        a = a @ w + b
        if i != last:
            a = hidden(a)
    return out_act(a)


def make_inference_fn(
    model: MLPClassifier, method: str
) -> tuple[Callable[[np.ndarray], Any], str]:
    """Return ``(callable, description)`` for the requested inference method."""
    if method == "predict_proba":
        return model.predict_proba, "predict_proba (with sklearn input validation)"
    if method == "predict":
        return model.predict, "predict (with sklearn input validation)"
    fast = getattr(model, "_forward_pass_fast", None)
    if fast is not None:
        import inspect

        if "check_input" in inspect.signature(fast).parameters:
            return (lambda x: fast(x, check_input=False)), \
                "_forward_pass_fast(check_input=False) (no validation)"
        LOG.warning("_forward_pass_fast has no check_input arg; it may validate.")
        return fast, "_forward_pass_fast (validation behaviour unknown)"
    LOG.warning("_forward_pass_fast missing; using manual NumPy forward pass.")
    return (lambda x: _numpy_forward(model, x)), "manual NumPy forward pass (no validation)"


# --------------------------------------------------------------------------- #
# Input generation
# --------------------------------------------------------------------------- #
def load_csv_rows(path: Path, n_features: int, max_rows: int) -> np.ndarray:
    """Read the first ``n_features`` numeric columns of a CSV (header optional)."""
    if not path.is_file():
        raise FileNotFoundError(f"input CSV not found: {path}")
    with path.open("r", newline="", encoding="utf-8") as fh:
        first = fh.readline()
    skip = 0
    try:
        [float(v) for v in first.strip().split(",")[:n_features]]
    except ValueError:
        skip = 1
    try:
        data = np.loadtxt(
            path, delimiter=",", skiprows=skip, usecols=range(n_features),
            dtype=np.float64, max_rows=max_rows, ndmin=2,
        )
    except (IndexError, ValueError) as exc:
        raise ValueError(f"cannot read {n_features} numeric columns from {path}: {exc}")
    if data.size == 0:
        raise ValueError(f"no data rows in {path}")
    if not np.isfinite(data).all():
        raise ValueError(f"{path} contains NaN/inf in the first {n_features} columns")
    return data


def build_batches(
    mode: str, csv_path: Optional[Path], n_features: int, batch_size: int,
    n_calls: int, seed: int, dtype: np.dtype,
) -> list[np.ndarray]:
    """Pre-build input batches OUTSIDE the timed loop.

    The pool holds at most ``min(1000, n_calls)`` batches (and at most
    ``MAX_POOL_ROWS`` rows); the timed loop cycles through it.
    """
    n_batches = max(1, min(1000, n_calls, MAX_POOL_ROWS // batch_size))
    total = n_batches * batch_size
    if mode == "random":
        pool = np.random.default_rng(seed).standard_normal((total, n_features))
    else:
        assert csv_path is not None
        data = load_csv_rows(csv_path, n_features, max_rows=total)
        pool = data[np.arange(total) % len(data)]  # cycle if too few rows
    pool = np.ascontiguousarray(pool.astype(dtype))
    return [pool[i * batch_size:(i + 1) * batch_size] for i in range(n_batches)]


# --------------------------------------------------------------------------- #
# Measurement
# --------------------------------------------------------------------------- #
def latency_stats(arr: np.ndarray) -> dict[str, Optional[float]]:
    """mean/std/min/max/p50/p90/p95/p99 of ``arr`` (None entries if empty)."""
    if arr.size == 0:
        return {k: None for k in STAT_KEYS}
    p50, p90, p95, p99 = np.percentile(arr, [50, 90, 95, 99])
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()) if arr.size > 1 else 0.0,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "p50": float(p50), "p90": float(p90), "p95": float(p95), "p99": float(p99),
    }


def scale(stats: dict[str, Optional[float]], factor: float) -> dict[str, Optional[float]]:
    """Multiply every stat by ``factor`` (None preserved)."""
    return {k: (None if v is None else v * factor) for k, v in stats.items()}


def run_measurement(
    fn: Callable[[np.ndarray], Any], batches: list[np.ndarray],
    warmup: int, iterations: int,
) -> dict[str, Any]:
    """Run warm-up then the timed loop; survives Ctrl-C with partial data."""
    n = len(batches)
    perf = time.perf_counter
    interrupted = False
    warm_done = 0
    done = 0
    lat = np.empty(iterations, dtype=np.float64)
    last_out: Any = None
    sampler = ResourceSampler()
    proc = psutil.Process(os.getpid())
    rss_before = proc.memory_info().rss
    rapl0: Optional[dict[str, tuple[int, int]]] = None
    rapl1: Optional[dict[str, tuple[int, int]]] = None
    wall = 0.0
    wall0: Optional[float] = None
    try:
        for i in range(warmup):
            last_out = fn(batches[i % n])
            warm_done = i + 1
        rapl0 = read_rapl()
        sampler.start()
        wall0 = perf()
        for i in range(iterations):
            x = batches[i % n]
            t0 = perf()
            last_out = fn(x)
            lat[i] = perf() - t0
            done = i + 1
        wall = perf() - wall0
    except KeyboardInterrupt:
        interrupted = True
        LOG.warning("Interrupted; writing partial results.")
        wall = perf() - wall0 if wall0 is not None else 0.0
    rapl1 = read_rapl()
    res = sampler.stop() if sampler.started else {}
    return {
        "latencies": lat[:done].copy(),
        "warmup_completed": warm_done,
        "iterations_completed": done,
        "wall_time_sec": wall,
        "interrupted": interrupted,
        "last_output": last_out,
        "resources": res,
        "ram_rss_before_mb": mb(rss_before),
        "cpu_power_w": rapl_power_w(rapl0, rapl1, wall),
    }


def to_markdown(r: dict[str, Any]) -> str:
    """Human-readable report."""
    lm, ls = r["latency_ms"], r["latency_per_sample_ms"]
    f = lambda v: "n/a" if v is None else f"{v:.4f}"  # noqa: E731
    cfg, mi, res, md = r["config"], r["model_info"], r["resources"], r["metadata"]
    lines = [
        f"# Benchmark: {r['run_id']}",
        "",
        f"- Framework: sklearn {md['sklearn']} | Inference method: `{r['inference_method']}` "
        f"({r['inference_impl']})",
        f"- Threads: {cfg['num_threads']} | Batch size: {cfg['batch_size']} | "
        f"Iterations: {cfg['iterations_completed']}/{cfg['iterations']} "
        f"(warm-up {cfg['warmup_completed']}/{cfg['warmup']}) | dtype: {cfg['dtype']}",
        f"- Status: {r['status']} | Host: {md['hostname']} | CPU: {md['cpu_model']} "
        f"({md['cpu_physical_cores']}c/{md['cpu_logical_cores']}t) | RAM: {md['ram_total_gb']} GB",
        f"- Model: {mi['path']} | params {mi['param_count']} | hidden {mi['hidden_layer_sizes']} "
        f"| load {f(mi['load_time_sec'])} s",
        "",
        "## Latency (ms)",
        "",
        "| stat | per batch | per sample |",
        "|---|---|---|",
    ]
    lines += [f"| {k} | {f(lm[k])} | {f(ls[k])} |" for k in STAT_KEYS]
    lines += [
        "",
        f"Throughput: {f(r['throughput_samples_per_sec'])} samples/s "
        f"(wall {f(r['measured_wall_time_sec'])} s)",
        "",
        "## Resources",
        "",
        f"- CPU % (system, mean): {f(res.get('cpu_percent_overall'))} "
        f"| CPU % (this process): {f(res.get('cpu_percent_process'))}",
        f"- RSS MB before/peak/after: {f(res.get('ram_rss_before_mb'))} / "
        f"{f(res.get('ram_rss_peak_mb'))} / {f(res.get('ram_rss_after_mb'))}",
        f"- Available RAM MB: {f(res.get('ram_available_mb'))} | CPU freq MHz: "
        f"{f(res.get('cpu_freq_mhz'))} | CPU temp C: {f(res.get('cpu_temp_c'))} "
        f"| CPU power W: {f(res.get('cpu_power_w'))}",
    ]
    if r["notes"]:
        lines += ["", "## Notes", "", r["notes"]]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def utc_stamp() -> tuple[str, str]:
    """Return (ISO-8601 timestamp, filename-safe timestamp)."""
    now = datetime.now(timezone.utc).replace(microsecond=0)
    return now.strftime("%Y-%m-%dT%H:%M:%SZ"), now.strftime("%Y-%m-%dT%H-%M-%SZ")


def execute(args: argparse.Namespace) -> dict[str, Any]:
    """Load the model, run the benchmark, and assemble the result dict."""
    notes: list[str] = []
    ts_iso, ts_file = utc_stamp()
    if args.iterations < 10:
        LOG.warning("--iterations < 10: percentiles will be very noisy.")
        notes.append("iterations < 10: percentiles are noisy")
    if _NUMPY_ALREADY_IMPORTED:
        notes.append("numpy was imported before thread env vars were set")
        LOG.warning(notes[-1])
    if os.environ.get("OMP_NUM_THREADS") != str(args.num_threads):
        LOG.warning("OMP_NUM_THREADS (%s) != --num-threads (%d)",
                    os.environ.get("OMP_NUM_THREADS"), args.num_threads)

    rss0 = psutil.Process(os.getpid()).memory_info().rss
    if args.self_test:
        model = build_self_test_model(args.seed)
        load_info: dict[str, Any] = {"load_time_sec": None, "size_bytes": None,
                                     "disk_read_bytes": None, "disk_write_bytes": None,
                                     "warnings": []}
        model_path = "<self-test>"
    else:
        model_path = args.model_path or DEFAULT_MODEL_PATH[args.model]
        model, load_info = load_model(Path(model_path))
        for w in load_info["warnings"]:
            LOG.warning("while loading model: %s", w)
        if load_info["warnings"]:
            notes.append("load warnings: " + " | ".join(load_info["warnings"]))
    info = describe_model(model)
    expected = EXPECTED_PARAM_COUNT.get(args.model)
    if expected is not None and info["param_count"] != expected:
        msg = f"param_count {info['param_count']} != expected {expected} for {args.model}"
        LOG.warning(msg)
        notes.append(msg)

    fn, impl = make_inference_fn(model, args.inference_method)
    if args.batch_size > 1:
        notes.append("batch_size > 1: both per-batch and per-sample latency reported")
    dtype = np.dtype(args.dtype)
    batches = build_batches(
        args.input_mode, Path(args.input_csv) if args.input_csv else None,
        info["n_features_in"], args.batch_size, args.warmup + args.iterations,
        args.seed, dtype,
    )
    avail_mb = mb(psutil.virtual_memory().available)

    with threadpool_limits(limits=args.num_threads):
        tp_info = threadpool_info()
        for lib in tp_info:
            if lib.get("num_threads") != args.num_threads:
                LOG.warning("threadpool %s reports %s threads (requested %d)",
                            lib.get("internal_api"), lib.get("num_threads"), args.num_threads)
        m = run_measurement(fn, batches, args.warmup, args.iterations)

    out = m["last_output"]
    if out is not None and not np.all(np.isfinite(np.asarray(out, dtype=np.float64))):
        notes.append("model output contained non-finite values")
        LOG.warning(notes[-1])
    done = m["iterations_completed"]
    lat = m["latencies"]
    per_batch = latency_stats(lat)
    per_sample = scale(per_batch, 1.0 / args.batch_size)
    wall = m["wall_time_sec"]
    throughput = (args.batch_size * done / wall) if wall > 0 and done else None
    if m["resources"].get("cpu_samples") == 0:
        notes.append("measured window < 100 ms: CPU % is one reading averaged over the window")

    res = dict(m["resources"])
    res.update({
        "ram_rss_before_mb": mb(rss0),
        "ram_rss_before_loop_mb": m["ram_rss_before_mb"],
        "ram_available_mb": avail_mb,
        "cpu_freq_mhz": read_cpu_freq_mhz(),
        "disk_read_bytes": load_info["disk_read_bytes"],
        "disk_write_bytes": load_info["disk_write_bytes"],
        "cpu_temp_c": read_cpu_temp_c(),
        "cpu_power_w": m["cpu_power_w"],
    })
    metadata = collect_metadata(_ENV_THREADS)
    metadata["threadpool_info"] = tp_info

    label = re.sub(r"[^A-Za-z0-9._-]+", "-", args.label).strip("-")
    model_tag = "selftest" if args.self_test else args.model
    run_id = "_".join(p for p in (model_tag, label, ts_file, "sklearn") if p)
    mi = {"path": model_path, "size_bytes": load_info["size_bytes"],
          "load_time_sec": load_info["load_time_sec"], **info,
          "expected_param_count": expected, "load_warnings": load_info["warnings"],
          "disk_read_bytes_during_load": load_info["disk_read_bytes"],
          "disk_write_bytes_during_load": load_info["disk_write_bytes"]}
    return {
        "run_id": run_id,
        "model": model_tag,
        "label": args.label,
        "timestamp_utc": ts_iso,
        "framework": "sklearn",
        "inference_method": args.inference_method,
        "inference_impl": impl,
        "git_commit": get_git_commit(),
        "cli_args": vars(args),
        "metadata": metadata,
        "model_info": mi,
        "config": {
            "batch_size": args.batch_size, "warmup": args.warmup,
            "warmup_completed": m["warmup_completed"], "iterations": args.iterations,
            "iterations_completed": done, "num_threads": args.num_threads,
            "input_mode": args.input_mode, "dtype": args.dtype, "seed": args.seed,
            "inference_method": args.inference_method, "pool_batches": len(batches),
        },
        "latency_sec": per_batch,
        "latency_ms": scale(per_batch, 1e3),
        "latency_per_sample_sec": per_sample,
        "latency_per_sample_ms": scale(per_sample, 1e3),
        "throughput_samples_per_sec": throughput,
        "measured_wall_time_sec": wall,
        "resources": res,
        "status": "interrupted" if m["interrupted"] else "complete",
        "notes": "; ".join(notes),
    }


def write_outputs(result: dict[str, Any], out_dir: Path) -> tuple[Path, Path, Optional[Path]]:
    """Write JSON, Markdown, and append to ``summary_sklearn.csv``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    clean = sanitize(result)
    json_path = out_dir / f"{result['run_id']}.json"
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(clean, fh, indent=2, allow_nan=False)
    md_path = out_dir / f"{result['run_id']}.md"
    md_path.write_text(to_markdown(clean), encoding="utf-8")

    csv_path: Optional[Path] = None
    if result["config"]["iterations_completed"] > 0:
        csv_path = out_dir / "summary_sklearn.csv"
        row = {
            "run_id": clean["run_id"], "framework": "sklearn",
            "inference_method": clean["inference_method"], "model": clean["model"],
            "label": clean["label"], "hostname": clean["metadata"]["hostname"],
            "timestamp_utc": clean["timestamp_utc"],
            "batch_size": clean["config"]["batch_size"],
            "iterations": clean["config"]["iterations_completed"],
            "num_threads": clean["config"]["num_threads"],
            "load_time_sec": clean["model_info"]["load_time_sec"],
            "latency_mean_ms": clean["latency_ms"]["mean"],
            "latency_p50_ms": clean["latency_ms"]["p50"],
            "latency_p95_ms": clean["latency_ms"]["p95"],
            "latency_p99_ms": clean["latency_ms"]["p99"],
            "throughput_sps": clean["throughput_samples_per_sec"],
            "cpu_percent_overall": clean["resources"].get("cpu_percent_overall"),
            "ram_rss_peak_mb": clean["resources"].get("ram_rss_peak_mb"),
            "param_count": clean["model_info"]["param_count"],
            "size_bytes": clean["model_info"]["size_bytes"],
            "git_commit": clean["git_commit"],
        }
        write_header = not csv_path.exists() or csv_path.stat().st_size == 0
        if not write_header:
            with csv_path.open("r", newline="", encoding="utf-8") as fh:
                existing = next(csv.reader(fh), [])
            if existing != list(CSV_FIELDS):
                raise ValueError(f"{csv_path} has a different header; move or delete it")
        with csv_path.open("a", newline="", encoding="utf-8") as fh:
            try:
                import fcntl
                fcntl.flock(fh, fcntl.LOCK_EX)
            except (ImportError, OSError):
                pass
            writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS, lineterminator="\n")
            if write_header:
                writer.writeheader()
            writer.writerow({k: ("" if v is None else v) for k, v in row.items()})
    return json_path, md_path, csv_path


def self_test_ok(result: dict[str, Any]) -> bool:
    """True iff key latency / throughput metrics are finite and positive."""
    vals = [result["throughput_samples_per_sec"], *(result["latency_sec"][k] for k in
            ("mean", "min", "max", "p50", "p90", "p95", "p99"))]
    return (result["status"] == "complete"
            and all(v is not None and math.isfinite(v) and v > 0 for v in vals))


def build_parser() -> argparse.ArgumentParser:
    """CLI definition (abbreviations disabled so the early thread scan is exact)."""
    p = argparse.ArgumentParser(
        description="Benchmark sklearn MLPClassifier inference (PRO-IDS AD1/AD3).",
        allow_abbrev=False)
    p.add_argument("--model", choices=["ad1", "ad3"], help="Which model (required unless --self-test).")
    p.add_argument("--model-path", default=None, help="Path to .joblib (default detector model path for the selected model).")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--iterations", type=int, default=1000)
    p.add_argument("--num-threads", type=int, default=1, help="BLAS/OMP threads (set at process start).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default="results")
    p.add_argument("--input-mode", choices=["random", "csv"], default="random")
    p.add_argument("--input-csv", default=None)
    p.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    p.add_argument("--label", default="")
    p.add_argument("--inference-method", choices=["predict_proba", "predict", "forward_pass"],
                   default="predict_proba")
    p.add_argument("--self-test", action="store_true", help="Quick test with an inline dummy MLP.")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point. Returns a process exit code."""
    logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                        format="%(asctime)s %(levelname)s %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.self_test and not args.model:
        parser.error("--model is required unless --self-test is given")
    if args.input_mode == "csv" and not args.input_csv:
        parser.error("--input-csv is required with --input-mode csv")
    for name in ("batch_size", "iterations", "num_threads"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be >= 1")
    if args.warmup < 0:
        parser.error("--warmup must be >= 0")
    if args.self_test:
        args.warmup, args.iterations, args.inference_method = 20, 100, "predict_proba"
        args.batch_size, args.input_mode = 1, "random"
    random.seed(args.seed)
    np.random.seed(args.seed)

    try:
        result = execute(args)
    except (FileNotFoundError, TypeError, ValueError) as exc:
        LOG.error("%s", exc)
        return 2

    if args.self_test:
        print(json.dumps(sanitize(result), indent=2, allow_nan=False))
        return 0 if self_test_ok(result) else 1

    json_path, md_path, csv_path = write_outputs(result, Path(args.output_dir))
    lm = result["latency_ms"]
    print(f"[{result['status']}] {result['run_id']}")
    if lm["mean"] is not None:
        print(f"  {result['inference_method']} | threads={args.num_threads} | batch={args.batch_size}"
              f" | mean {lm['mean']:.4f} ms | p50 {lm['p50']:.4f} | p95 {lm['p95']:.4f}"
              f" | p99 {lm['p99']:.4f} ms | {result['throughput_samples_per_sec']:.0f} samples/s")
    print(f"  wrote {json_path}\n  wrote {md_path}\n  summary: {csv_path}")
    return 130 if result["status"] == "interrupted" else 0


if __name__ == "__main__":
    sys.exit(main())
