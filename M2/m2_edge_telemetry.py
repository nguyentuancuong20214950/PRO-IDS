#!/usr/bin/env python3
"""m2_edge_telemetry.py - PRO-IDS M2, edge side.

Runs on the EDGE VM. Every ``m2.sample_interval_ms`` a sampler thread builds the full
state vector s_t (paper Eq. 12, see ``m2_state.StateVector``) from
  * M5-edge  GET /metrics            -> q_e
  * psutil                           -> u_e (system-wide CPU %)
  * orchestrator notifications       -> lambda_hat, tau_hat_bps, n_c
  * M5-edge GET /metrics             -> mu_e_hat from requests_completed deltas
  * M2-cloud GET /telemetry          -> q_c, u_c, mu_c_hat, rtt_ms, h_c_ms
and appends it to ``<output_dir>/m2_edge_<UTC stamp>.jsonl``. The state is only
OBSERVED here; decisions are M4's job.

HTTP API (bound to 127.0.0.1; the orchestrator runs on the edge):
    GET  /state            latest StateVector (503 until the first sample)
    GET  /health           liveness, last cloud poll status and age
    POST /notify_arrival   {}                                  arrivals_total += 1
    POST /notify_completion {}                                 completions_total += 1
    POST /notify_bytes     {"bytes_sent": 320, "bytes_received": 48}
    POST /notify_n_c       {"n_c": 3}

Design notes (read before changing anything)
--------------------------------------------
* Double EMA on mu_c_hat. M2-cloud already EMA-smooths its service rate. The edge
  applies a second EMA to the value it polls. This is intentional: it damps jitter
  of the polling channel (the same cloud snapshot can be read twice, or a poll can
  be delayed). Consequence: mu_c_hat lags the true rate more than a single EMA.
* h_c_ms never resets on failure. h_c(t) = t - t_last (Eq. 13), t_last being the
  edge's monotonic time of the last SUCCESSFUL cloud poll. If M2-cloud dies, h_c_ms
  keeps growing so M4 can see how stale the cloud view is. It is null only if no
  poll has ever succeeded. In /state it is recomputed at READ time (M4 reads at
  decision time); the JSONL line stores the value at sample time. It measures time
  since the edge RECEIVED telemetry, not the age of the cloud's own measurement
  (which can be up to one cloud sample interval older).
* Cloud outage keeps last-known values. When a poll fails, q_c, u_c, mu_c_hat and
  rtt_ms keep their last known values and cloud_poll_ok=false / h_c_ms flag the
  staleness; M4 should judge freshness with h_c_ms, not with q_c == null.
* rtt_ms is the application-level time of one HTTP GET /telemetry on a fresh TCP
  connection (handshake + request/response + server work), EMA-smoothed. It is an
  upper bound on pure network RTT.
* tau_hat_bps is BITS per second: 8 * (bytes_sent + bytes_received) notified by the
  orchestrator per second, EMA-smoothed. It is offered traffic, not link capacity.
* mu_e_hat is derived only from M5-edge ``requests_completed`` counter deltas.
  /notify_completion remains accepted for compatibility, but does not feed this
  edge service-rate estimate.
* q_e and q_c use M5 in-flight counts. These count active requests and are proxies
  for waiting-queue occupancy; the services do not expose waiting queue lengths.
* Timing: samples are scheduled at a fixed rate, but one slow cloud poll (timeout
  up to ``cloud_telemetry_timeout_sec``) stretches that cycle.
* Locking: ``state_lock`` guards counters, n_c, latest_state and the last-poll time
  shared with HTTP handlers. It is never held during HTTP or file I/O.

Smoke test 2 - edge alone (M2-cloud not running):
    python3 m2_edge_telemetry.py --config config.yaml
    curl -s http://127.0.0.1:8004/health | python3 -m json.tool
    curl -s http://127.0.0.1:8004/state  | python3 -m json.tool
  Expect q_e from M5-edge, u_e, cloud_poll_ok=false, h_c_ms=null (stays null).

Smoke test 3 - full stack (M5-cloud, M5-edge, M2-cloud, M2-edge running):
    curl -X POST http://127.0.0.1:8004/notify_arrival      # twice
    curl -X POST http://127.0.0.1:8004/notify_completion
    curl -X POST http://127.0.0.1:8004/notify_bytes -H "Content-Type: application/json" \\
         -d '{"bytes_sent": 320, "bytes_received": 48}'
    curl -X POST http://127.0.0.1:8004/notify_n_c -H "Content-Type: application/json" \\
         -d '{"n_c": 2}'
  After 1-2 samples /state shows q_e, q_c, cloud_poll_ok=true, small h_c_ms, rtt_ms,
  n_c=2; lambda_hat / mu_e_hat / tau_hat_bps are non-null from the 2nd sample on
  (EMA.warmed_up after 3).

Smoke test 4 - M2-cloud goes down: kill it while polling /state every 500 ms.
  cloud_poll_ok flips to false, h_c_ms keeps growing, q_c / u_c keep last values.
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
from pydantic import BaseModel, Field

from m2_state import (EMA, RateLimitedWarner, StateVector, append_jsonl, iso_now,
                      require_keys, utc_stamp)

SERVICE = "m2_edge"
REQUIRED_KEYS = [
    "edge.port", "cloud.advertised_host", "cloud.port",
    "m2.sample_interval_ms", "m2.ema_beta", "m2.cloud_telemetry_port", "m2.edge_state_port",
    "m2.cloud_telemetry_timeout_sec", "m2.output_dir", "m2.log_level", "logging.level",
]
M5_TIMEOUT_SEC = 0.5


class BytesNotice(BaseModel):
    bytes_sent: int = Field(ge=0)
    bytes_received: int = Field(ge=0)


class NCNotice(BaseModel):
    n_c: int = Field(ge=0)


def load_config(path: str) -> tuple[dict, Path]:
    """Load and validate the YAML config. Returns (cfg, directory containing it)."""
    cfg_path = Path(path).resolve()
    with open(cfg_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    require_keys(cfg, REQUIRED_KEYS)
    if cfg["m2"]["sample_interval_ms"] <= 0:
        raise ValueError("m2.sample_interval_ms must be > 0")
    return cfg, cfg_path.parent


class EdgeTelemetry:
    """Owns the shared state, the notification counters and the sampler thread."""

    def __init__(self, cfg: dict, base_dir: Path, log: logging.Logger):
        self.log = log
        beta = cfg["m2"]["ema_beta"]
        self.edge_port = int(cfg["edge"]["port"])
        self.cloud_url = (f"http://{cfg['cloud']['advertised_host']}:"
                          f"{int(cfg['m2']['cloud_telemetry_port'])}/telemetry")
        self.cloud_timeout = float(cfg["m2"]["cloud_telemetry_timeout_sec"])
        self.interval_s = cfg["m2"]["sample_interval_ms"] / 1000.0
        out_dir = base_dir / cfg["m2"]["output_dir"]
        out_dir.mkdir(parents=True, exist_ok=True)
        self.out_path = out_dir / f"m2_edge_{utc_stamp()}.jsonl"

        # Guards everything the HTTP handlers also touch (see module docstring).
        self.state_lock = threading.Lock()
        self.arrivals_total = 0
        self.completions_total = 0
        self.bytes_total = 0
        self.last_n_c: int | None = None
        self.latest_state: StateVector | None = None
        self.samples_total = 0
        self.last_successful_cloud_poll_mono: float | None = None

        # Sampler-thread-only state (no lock needed).
        self.ema_lambda = EMA(beta)
        self.ema_mu_e = EMA(beta)
        self.ema_mu_c = EMA(beta)  # second smoothing stage on top of the cloud's own EMA
        self.ema_rtt = EMA(beta)
        self.ema_tau = EMA(beta)
        self.prev_sample_mono: float | None = None
        self.prev_arrivals = 0
        self.prev_bytes = 0
        self.prev_m5_edge_completed: int | None = None
        self.prev_m5_edge_sample_mono: float | None = None
        self.last_cloud_snapshot: dict | None = None  # last SUCCESSFUL cloud poll
        self.last_m5_e_metrics: dict | None = None
        self.warner = RateLimitedWarner(log)

        self.start_mono = time.perf_counter()
        self.stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        psutil.cpu_percent(interval=None)  # prime: the first call always returns 0.0

    # ------------------------------------------------------- orchestrator notifications
    def notify_arrival(self) -> None:
        with self.state_lock:
            self.arrivals_total += 1

    def notify_completion(self) -> None:
        with self.state_lock:
            self.completions_total += 1

    def notify_bytes(self, sent: int, received: int) -> None:
        with self.state_lock:
            self.bytes_total += sent + received

    def notify_n_c(self, n_c: int) -> None:
        with self.state_lock:
            self.last_n_c = n_c

    # ------------------------------------------------------------------ sampling
    def sample_once(self) -> StateVector:
        """Build, publish and log one state vector."""
        # Timestamp and counter snapshot are taken together, then the lock is released
        # BEFORE any HTTP call.
        t_now = time.perf_counter()
        with self.state_lock:
            cur_arrivals = self.arrivals_total
            cur_bytes = self.bytes_total
            n_c = self.last_n_c

        # --- arrival and byte rates from orchestrator notifications ---
        lambda_obs = tau_obs = None
        if self.prev_sample_mono is not None:
            dt = t_now - self.prev_sample_mono
            if dt > 0:
                lambda_obs = (cur_arrivals - self.prev_arrivals) / dt
                tau_obs = (cur_bytes - self.prev_bytes) * 8.0 / dt  # bytes -> bits per second
        self.prev_sample_mono = t_now
        self.prev_arrivals, self.prev_bytes = cur_arrivals, cur_bytes
        lambda_hat = self.ema_lambda.update(lambda_obs)
        tau_hat = self.ema_tau.update(tau_obs)

        # --- local edge observations ---
        q_e, m5_e_ok, m5_e_error = None, False, None
        completed = None
        m5_e_sample_mono = None
        try:
            r = requests.get(f"http://127.0.0.1:{self.edge_port}/metrics", timeout=M5_TIMEOUT_SEC)
            r.raise_for_status()
            metrics = r.json()
            q_e = int(metrics["requests_in_flight"])
            completed = int(metrics["requests_completed"])
            m5_e_sample_mono = time.perf_counter()
            self.last_m5_e_metrics = metrics
            m5_e_ok = True
        except Exception as exc:  # noqa: BLE001 - degrade, never crash the sampler
            m5_e_error = f"{type(exc).__name__}: {exc}"
        if m5_e_ok:
            self.warner.recovered("m5_edge")
        else:
            self.warner.failure("m5_edge", f"M5-edge /metrics unreachable: {m5_e_error}")

        # Use successful inference counts from M5-edge, not orchestrator outcome
        # notifications, to estimate the edge service rate. Advance the baseline
        # only after a successful read so a metrics outage does not create a spike.
        mu_e_obs = None
        if m5_e_ok:
            if (self.prev_m5_edge_completed is not None
                    and self.prev_m5_edge_sample_mono is not None):
                dt = m5_e_sample_mono - self.prev_m5_edge_sample_mono
                delta = completed - self.prev_m5_edge_completed
                if delta < 0:
                    self.log.warning(
                        "M5-edge requests_completed went backwards (%d -> %d); re-baselining",
                        self.prev_m5_edge_completed, completed,
                    )
                elif dt > 0:
                    mu_e_obs = delta / dt
            self.prev_m5_edge_completed = completed
            self.prev_m5_edge_sample_mono = m5_e_sample_mono
        mu_e_hat = self.ema_mu_e.update(mu_e_obs)

        u_e = psutil.cpu_percent(interval=None)

        # --- cloud telemetry poll (also yields the RTT observation) ---
        cloud_ok, cloud_error, cloud_snap, rtt_obs_ms = False, None, None, None
        t_poll_start = time.perf_counter()
        try:
            r = requests.get(self.cloud_url, timeout=self.cloud_timeout)
            r.raise_for_status()
            cloud_snap = r.json()
            t_poll_end = time.perf_counter()
            rtt_obs_ms = (t_poll_end - t_poll_start) * 1000.0
            cloud_ok = True
        except Exception as exc:  # noqa: BLE001
            t_poll_end = time.perf_counter()
            cloud_error = f"{type(exc).__name__}: {exc}"
        if cloud_ok:
            with self.state_lock:
                self.last_successful_cloud_poll_mono = t_poll_end
                last_ok = t_poll_end
            self.last_cloud_snapshot = cloud_snap
            self.warner.recovered("cloud_poll")
        else:
            with self.state_lock:
                last_ok = self.last_successful_cloud_poll_mono
            self.warner.failure("cloud_poll", f"cloud telemetry poll failed: {cloud_error}")
        rtt_hat = self.ema_rtt.update(rtt_obs_ms)  # None on failure -> last value kept

        # h_c is measured against "now" (after the poll), so it is never negative.
        h_c_ms = None if last_ok is None else (time.perf_counter() - last_ok) * 1000.0

        # Cloud fields: fresh when the poll worked, otherwise LAST KNOWN values.
        src = cloud_snap if cloud_ok else self.last_cloud_snapshot
        q_c = src.get("q_c") if src else None
        u_c = src.get("u_c") if src else None
        mu_c_obs = cloud_snap.get("mu_c_hat") if cloud_ok else None
        mu_c_hat = self.ema_mu_c.update(mu_c_obs)

        with self.state_lock:
            self.samples_total += 1
            n = self.samples_total
        state = StateVector(
            timestamp_utc=iso_now(), timestamp_mono=t_now,
            q_e=q_e, q_c=q_c, u_e=u_e, u_c=u_c,
            lambda_hat=lambda_hat, mu_e_hat=mu_e_hat, mu_c_hat=mu_c_hat,
            rtt_ms=rtt_hat, tau_hat_bps=tau_hat, n_c=n_c, h_c_ms=h_c_ms,
            q_c_is_proxy=True,  # in-flight count proxies waiting-queue occupancy
            cloud_poll_ok=cloud_ok, cloud_poll_error=cloud_error,
            m5_edge_ok=m5_e_ok, m5_edge_error=m5_e_error,
            samples_total=n,
        )
        with self.state_lock:
            self.latest_state = state

        try:
            append_jsonl(self.out_path, state.to_dict())
            self.warner.recovered("jsonl")
        except OSError as exc:
            self.warner.failure("jsonl", f"cannot write {self.out_path}: {exc}")

        self.log.debug("sample %d: %s", n, state.to_dict())
        if n % 10 == 0:
            r1 = lambda v: None if v is None else round(v, 2)  # noqa: E731
            self.log.info("sample %d: q_e=%s q_c=%s lambda=%s mu_e=%s mu_c=%s rtt_ms=%s h_c_ms=%s "
                          "cloud_ok=%s m5_edge_ok=%s", n, q_e, q_c, r1(lambda_hat), r1(mu_e_hat),
                          r1(mu_c_hat), r1(rtt_hat), r1(h_c_ms), cloud_ok, m5_e_ok)
        return state

    def _sample_loop(self) -> None:
        # Fixed-rate schedule without drift; an overrun skips ahead (no catch-up burst).
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
        self._thread = threading.Thread(target=self._sample_loop, name="m2-edge-sampler",
                                        daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    # ------------------------------------------------------------------ read side
    def get_state_dict(self) -> dict | None:
        """Latest state with h_c_ms recomputed at read time (see module docstring)."""
        with self.state_lock:
            st = self.latest_state
            last_ok = self.last_successful_cloud_poll_mono
        if st is None:
            return None
        d = st.to_dict()
        d["h_c_ms"] = None if last_ok is None else (time.perf_counter() - last_ok) * 1000.0
        return d

    def health(self) -> dict:
        with self.state_lock:
            st = self.latest_state
            last_ok = self.last_successful_cloud_poll_mono
            n = self.samples_total
        return {
            "status": "ok",
            "service": SERVICE,
            "uptime_sec": round(time.perf_counter() - self.start_mono, 3),
            "samples_total": n,
            "last_sample_utc": st.timestamp_utc if st else None,
            "last_cloud_poll_ok": st.cloud_poll_ok if st else None,
            "last_cloud_poll_age_ms": None if last_ok is None
            else (time.perf_counter() - last_ok) * 1000.0,
        }


def build_app(svc: EdgeTelemetry) -> FastAPI:
    """FastAPI app: thin HTTP layer over EdgeTelemetry."""
    app = FastAPI(title="PRO-IDS M2 edge state")

    @app.get("/state")
    def state():
        d = svc.get_state_dict()
        if d is None:
            return JSONResponse(status_code=503, content={"error": "no samples yet"})
        return d

    @app.get("/health")
    def health():
        return svc.health()

    @app.post("/notify_arrival")
    def notify_arrival():
        svc.notify_arrival()
        return {"status": "ok"}

    @app.post("/notify_completion")
    def notify_completion():
        svc.notify_completion()
        return {"status": "ok"}

    @app.post("/notify_bytes")
    def notify_bytes(body: BytesNotice):
        svc.notify_bytes(body.bytes_sent, body.bytes_received)
        return {"status": "ok"}

    @app.post("/notify_n_c")
    def notify_n_c(body: NCNotice):
        svc.notify_n_c(body.n_c)
        return {"status": "ok"}

    return app


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="PRO-IDS M2 edge state service")
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
        svc = EdgeTelemetry(cfg, base_dir, log)
    except (OSError, ValueError) as exc:
        log.error("cannot initialise: %s", exc)
        return 2
    port = int(cfg["m2"]["edge_state_port"])
    log.info("starting: state port=%d (127.0.0.1), M5-edge port=%d, cloud telemetry=%s "
             "(timeout %.2fs), sample_interval_ms=%s, ema_beta=%s, output=%s",
             port, svc.edge_port, svc.cloud_url, svc.cloud_timeout,
             cfg["m2"]["sample_interval_ms"], cfg["m2"]["ema_beta"], svc.out_path)

    svc.start()
    try:
        # Edge-side only: the orchestrator runs on the same VM.
        uvicorn.run(build_app(svc), host="127.0.0.1", port=port,
                    log_level=str(cfg["m2"]["log_level"]).lower(), workers=1)
    finally:
        svc.stop()
        log.info("%s shutting down", SERVICE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
