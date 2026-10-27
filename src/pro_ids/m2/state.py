"""m2_state.py - shared building blocks for the PRO-IDS M2 (system monitoring) module.

Pure Python, standard library only (no numpy, no psutil, no yaml). Contains:

* ``EMA``              exponential moving average, paper Eq. (15).
* ``StateVector``      the state s_t of paper Eq. (12) plus diagnostics.
* small helpers        ISO timestamps, config key validation, JSONL append and a
                       rate-limited warning logger used by both M2 services.

Field-name mapping vs. the paper (Eq. 12):
    lambda-hat -> lambda_hat      mu_e-hat -> mu_e_hat      mu_c-hat -> mu_c_hat
    tau-hat    -> tau_hat_bps     RTT      -> rtt_ms        h_c      -> h_c_ms
    q_e, q_c, u_e, u_c, n_c keep their names.

Units: ``tau_hat_bps`` is BITS per second (application-level bytes notified by
the orchestrator, multiplied by 8). ``rtt_ms`` / ``h_c_ms`` are milliseconds.
"""

import json
import logging
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


def iso_now() -> str:
    """Current UTC time, ISO 8601 with millisecond precision and a 'Z' suffix."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def utc_stamp() -> str:
    """Compact UTC stamp (YYYYMMDDTHHMMSSZ) used in JSONL file names."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def require_keys(cfg: dict, dotted_keys: list) -> None:
    """Raise KeyError listing every missing ``a.b.c`` path in ``cfg``."""
    missing = []
    for dotted in dotted_keys:
        node = cfg
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node or node[part] is None:
                missing.append(dotted)
                break
            node = node[part]
    if missing:
        raise KeyError("missing required config keys: " + ", ".join(missing))


def append_jsonl(path: Path, record: dict) -> None:
    """Append one JSON object as a single line (open/write/flush/close each call).

    Opening per sample (2 Hz) costs nothing and keeps the file valid even if the
    process is killed between samples.
    """
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        fh.flush()


class RateLimitedWarner:
    """Log a repeating failure once, then every ``every``-th consecutive time.

    Used from a single sampler thread only (no locking). ``recovered(key)``
    resets the counter and logs one INFO line if the key had been failing.
    """

    def __init__(self, logger: logging.Logger, every: int = 20):
        self._log = logger
        self._every = every
        self._counts: dict = {}

    def failure(self, key: str, message: str) -> None:
        n = self._counts.get(key, 0) + 1
        self._counts[key] = n
        if n == 1 or n % self._every == 0:
            suffix = "" if n == 1 else f" (consecutive failure #{n})"
            self._log.warning("%s%s", message, suffix)

    def recovered(self, key: str) -> None:
        n = self._counts.pop(key, 0)
        if n:
            self._log.info("%s recovered after %d consecutive failure(s)", key, n)


class EMA:
    """Exponential moving average with warm-up (paper Eq. 15).

    ``value`` is None until the first valid sample. None / NaN / +-inf inputs
    are ignored and the previous value is returned unchanged.
    """

    def __init__(self, beta: float = 0.3):
        if not (0 < beta <= 1):
            raise ValueError(f"ema beta must be in (0, 1], got {beta}")
        self.beta = beta
        self.value: float | None = None
        self._n = 0

    def update(self, x: float | None) -> float | None:
        """Return the updated EMA, or the previous value if x is None/NaN/inf."""
        if x is None or x != x:  # NaN check without numpy
            return self.value
        if x == float("inf") or x == float("-inf"):
            return self.value
        self._n += 1
        if self.value is None:
            self.value = float(x)
        else:
            self.value = self.beta * float(x) + (1 - self.beta) * self.value
        return self.value

    @property
    def warmed_up(self) -> bool:
        """True after at least 3 valid samples."""
        return self._n >= 3


@dataclass
class StateVector:
    """System state s_t (paper Eq. 12) plus diagnostics. None = not observable."""

    timestamp_utc: str
    timestamp_mono: float  # time.perf_counter() at sample time (edge monotonic clock)
    # --- 11 fields from paper Eq. (12), in order ---
    q_e: int | None = None
    q_c: int | None = None
    u_e: float | None = None
    u_c: float | None = None
    lambda_hat: float | None = None
    mu_e_hat: float | None = None
    mu_c_hat: float | None = None
    rtt_ms: float | None = None
    tau_hat_bps: float | None = None
    n_c: int | None = None
    h_c_ms: float | None = None
    # --- diagnostics ---
    q_c_is_proxy: bool = False
    cloud_poll_ok: bool = False
    cloud_poll_error: str | None = None
    m5_edge_ok: bool = False
    m5_edge_error: str | None = None
    samples_total: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def monotonic() -> float:
    """Single place for the monotonic clock used by all M2 timing."""
    return time.perf_counter()
