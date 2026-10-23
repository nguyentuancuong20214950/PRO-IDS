#!/usr/bin/env python3
"""m2_cloud_telemetry.py - PRO-IDS M2, cloud side.

Runs on the CLOUD VM. A background thread samples cloud-side state every
``m2.sample_interval_ms`` and a FastAPI app exposes the newest sample:

    GET /telemetry   latest snapshot (503 until the first sample exists)
    GET /health      liveness + sample counters

Each snapshot holds q_c (real ``requests_in_flight`` of M5-cloud), u_c (system CPU %),
mu_c_hat (EMA-smoothed service rate, requests/s) and diagnostics. Every snapshot is
also appended to ``<output_dir>/m2_cloud_<UTC stamp>.jsonl``.

M5-cloud reports its own in-flight count. This is a direct active-request count,
but only a proxy for waiting-queue occupancy because the service does not expose
the number of requests waiting to begin inference. ``is_proxy`` is therefore true.

Degradation: if M5-cloud is unreachable, q_c and mu_c_hat are null (mu_c_hat keeps the
last EMA value), ``m5_cloud_ok`` is false and the service keeps running and answering.

Service rate: mu_obs = delta(requests_completed) / delta(t) between two SUCCESSFUL
reads of M5-cloud (the baseline only advances on success, so an outage yields the
average rate over the gap, not a spike). A counter that goes backwards (M5-cloud
restart) re-baselines instead of producing a negative rate.

Smoke test 1 (cloud VM):
    cd testbed
    python3 m2_cloud_telemetry.py --config config.yaml
    curl -s http://127.0.0.1:8003/health    | python3 -m json.tool
    curl -s http://127.0.0.1:8003/telemetry | python3 -m json.tool
  Expect q_c / u_c / mu_c_hat / timestamp_utc. With M5-cloud stopped: q_c = null,
  mu_c_hat is null before its first observation or retains its last EMA afterward;
  the endpoint continues returning HTTP 200 (503 only before the first sample).
"""

import argparse
import logging
import sys
import threading
import time
from pathlib import Path

import psutil
import requests
import uvicorn
import yaml
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from m2_state import EMA, RateLimitedWarner, append_jsonl, iso_now, require_keys, utc_stamp

SERVICE = "m2_cloud"
REQUIRED_KEYS = [
    "cloud.port", "cloud.advertised_host",
    "m2.sample_interval_ms", "m2.ema_beta", "m2.cloud_telemetry_port",
    "m2.output_dir", "m2.log_level", "logging.level",
]
M5_TIMEOUT_SEC = 0.5


def load_config(path: str) -> tuple[dict, Path]:
    """Load and validate the YAML config. Returns (cfg, directory containing it)."""
    cfg_path = Path(path).resolve()
    with open(cfg_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    require_keys(cfg, REQUIRED_KEYS)
    if cfg["m2"]["sample_interval_ms"] <= 0:
        raise ValueError("m2.sample_interval_ms must be > 0")
    return cfg, cfg_path.parent


class CloudTelemetry:
    """Owns the shared state and the sampler thread of the M2 cloud service."""

    def __init__(self, cfg: dict, base_dir: Path, log: logging.Logger):
        self.log = log
        self.cloud_port = int(cfg["cloud"]["port"])
        self.interval_s = cfg["m2"]["sample_interval_ms"] / 1000.0
        out_dir = base_dir / cfg["m2"]["output_dir"]
        out_dir.mkdir(parents=True, exist_ok=True)
        self.out_path = out_dir / f"m2_cloud_{utc_stamp()}.jsonl"

        # Guards latest_snapshot and samples_total (shared with HTTP handlers).
        # Never held while doing HTTP or file I/O.
        self.state_lock = threading.Lock()
        self.latest_snapshot: dict | None = None
        self.samples_total = 0

        # Sampler-thread-only state (no lock needed).
        self.ema_mu_c = EMA(beta=cfg["m2"]["ema_beta"])
        self.prev_completed: int | None = None
        self.prev_sample_mono: float | None = None
        self.warner = RateLimitedWarner(log)

        self.start_mono = time.perf_counter()
        self.stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        psutil.cpu_percent(interval=None)  # prime: the first call always returns 0.0

    # ------------------------------------------------------------------ sampling
    def sample_once(self) -> dict:
        """Take one sample, publish it, append it to the JSONL log."""
        q_c = completed = None
        m5_ok, m5_error = False, None
        try:
            r = requests.get(f"http://127.0.0.1:{self.cloud_port}/metrics",
                             timeout=M5_TIMEOUT_SEC)
            r.raise_for_status()
            metrics = r.json()
            q_c = int(metrics["requests_in_flight"])
            completed = int(metrics["requests_completed"])
            m5_ok = True
        except Exception as exc:  # noqa: BLE001 - degrade, never crash the sampler
            m5_error = f"{type(exc).__name__}: {exc}"
        # Timestamp taken right after the read so (counter, time) pair up tightly.
        t_now = time.perf_counter()

        if m5_ok:
            self.warner.recovered("m5_cloud")
        else:
            self.warner.failure("m5_cloud", f"M5-cloud /metrics unreachable: {m5_error}")

        u_c = psutil.cpu_percent(interval=None)  # system-wide CPU % since last call

        # Observed service rate. Baseline (count, time) advances ONLY on success.
        mu_obs = None
        if m5_ok:
            if self.prev_completed is not None and self.prev_sample_mono is not None:
                dt = t_now - self.prev_sample_mono
                delta = completed - self.prev_completed
                if delta < 0:
                    self.log.warning("requests_completed went backwards (%d -> %d): "
                                     "M5-cloud restarted? re-baselining", self.prev_completed, completed)
                elif dt > 0:
                    mu_obs = delta / dt
            self.prev_completed, self.prev_sample_mono = completed, t_now
        mu_c_hat = self.ema_mu_c.update(mu_obs)  # None obs -> last EMA value is kept

        with self.state_lock:
            self.samples_total += 1
            n = self.samples_total
            snapshot = {
                "timestamp_utc": iso_now(),
                "t_cloud_mono": t_now,
                "q_c": q_c,
                "u_c": u_c,
                "mu_c_hat": mu_c_hat,
                "m5_cloud_ok": m5_ok,
                "m5_cloud_error": m5_error,
                "is_proxy": True,  # active requests proxy waiting-queue occupancy
                "samples_total": n,
            }
            self.latest_snapshot = snapshot

        try:
            append_jsonl(self.out_path, snapshot)
            self.warner.recovered("jsonl")
        except OSError as exc:
            self.warner.failure("jsonl", f"cannot write {self.out_path}: {exc}")

        self.log.debug("sample %d: q_c=%s u_c=%.1f mu_c_hat=%s", n, q_c, u_c, mu_c_hat)
        if n % 10 == 0:
            self.log.info("sample %d: q_c=%s u_c=%.1f%% mu_c_hat=%s m5_ok=%s",
                          n, q_c, u_c, None if mu_c_hat is None else round(mu_c_hat, 3), m5_ok)
        return snapshot

    def _sample_loop(self) -> None:
        # Fixed-rate schedule: no cumulative drift; if a sample overruns we skip
        # ahead instead of firing a burst of catch-up samples.
        next_t = time.perf_counter()
        while not self.stop_event.is_set():
            try:
                self.sample_once()
            except Exception:  # noqa: BLE001 - the sampler thread must never die
                self.log.exception("unexpected error in sample loop")
            next_t += self.interval_s
            now = time.perf_counter()
            if next_t < now:
                next_t = now
            if self.stop_event.wait(next_t - now):
                break

    def start(self) -> None:
        self._thread = threading.Thread(target=self._sample_loop, name="m2-cloud-sampler",
                                        daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    # ------------------------------------------------------------------ read side
    def get_snapshot(self) -> dict | None:
        with self.state_lock:
            return dict(self.latest_snapshot) if self.latest_snapshot else None

    def health(self) -> dict:
        snap = self.get_snapshot()
        with self.state_lock:
            n = self.samples_total
        return {
            "status": "ok",
            "service": SERVICE,
            "uptime_sec": round(time.perf_counter() - self.start_mono, 3),
            "samples_total": n,
            "last_sample_utc": snap["timestamp_utc"] if snap else None,
        }


def build_app(svc: CloudTelemetry) -> FastAPI:
    """FastAPI app: thin HTTP layer over CloudTelemetry (no auth, no rate limiting)."""
    app = FastAPI(title="PRO-IDS M2 cloud telemetry")

    @app.get("/telemetry")
    def telemetry():
        snap = svc.get_snapshot()
        if snap is None:
            return JSONResponse(status_code=503, content={"error": "no samples yet"})
        return snap

    @app.get("/health")
    def health():
        return svc.health()

    return app


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="PRO-IDS M2 cloud telemetry service")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args(argv)

    try:
        cfg, base_dir = load_config(args.config)
    except (OSError, KeyError, ValueError, yaml.YAMLError) as exc:
        sys.stderr.write(f"{SERVICE}: config error: {exc}\n")
        return 2

    logging.basicConfig(level=getattr(logging, str(cfg["logging"]["level"]).upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    log = logging.getLogger(SERVICE)

    try:
        svc = CloudTelemetry(cfg, base_dir, log)
    except (OSError, ValueError) as exc:
        log.error("cannot initialise: %s", exc)
        return 2
    port = int(cfg["m2"]["cloud_telemetry_port"])
    log.info("starting: telemetry port=%d (0.0.0.0), M5-cloud port=%d, advertised_host=%s, "
             "sample_interval_ms=%s, ema_beta=%s, output=%s",
             port, svc.cloud_port, cfg["cloud"]["advertised_host"],
             cfg["m2"]["sample_interval_ms"], cfg["m2"]["ema_beta"], svc.out_path)

    svc.start()
    try:
        # 0.0.0.0 so the edge VM can reach /telemetry.
        uvicorn.run(build_app(svc), host="0.0.0.0", port=port,
                    log_level=str(cfg["m2"]["log_level"]).lower(), workers=1)
    finally:
        svc.stop()
        log.info("%s shutting down", SERVICE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
