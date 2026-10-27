"""Run the PRO-IDS Phase B testbed: source requests, sample M2, and dispatch to M5.

Routing is deliberately hardcoded to edge-only, cloud-only, or alternating.
This orchestrator does not implement M3 predictors, adaptive M4 decisions,
deadline enforcement, traffic shaping, or load generation.
"""
from __future__ import annotations

import argparse
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from uuid import uuid4

import requests
import yaml

from logging_utils import (
    SUMMARY_FIELDS,
    JsonlWriter,
    SummaryCsvWriter,
    iso_filename_token,
    iso_now,
)
from request_source import RequestSource

LOG = logging.getLogger("pro_ids.orchestrator")


def _percentile(values: list[float], percentile: float) -> float | None:
    """Return the nearest-rank-like percentile used by the project prompt."""
    if not values:
        return None
    ordered = sorted(values)
    index = max(
        0,
        min(len(ordered) - 1, int(round((percentile / 100.0) * (len(ordered) - 1)))),
    )
    return ordered[index]


def _mean(values: list[float]) -> float | None:
    """Return the arithmetic mean or None for an empty list."""
    return sum(values) / len(values) if values else None


def _round(value: float | None, digits: int = 6) -> float | None:
    """Round a value while preserving None for unavailable statistics."""
    return None if value is None else round(value, digits)


def _git_commit() -> str | None:
    """Return the current Git commit hash when available."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=2,
            check=True,
        )
        return result.stdout.strip() or None
    except Exception:
        return None


def _notify(endpoint: str, body: dict | None, timeout: float, base_url: str) -> None:
    """Fire-and-forget a best-effort notification to the co-located M2 service."""
    try:
        requests.post(
            f"{base_url}/{endpoint}",
            json=body or {},
            timeout=timeout,
        )
    except Exception:
        pass


def _health_check(url: str, name: str, *, required: bool, dry_run: bool) -> bool:
    """Check an HTTP health endpoint, exiting on required failure outside dry-run."""
    try:
        response = requests.get(url, timeout=3.0)
        response.raise_for_status()
        try:
            details = response.json()
        except ValueError:
            details = response.text
        LOG.info("%s healthy: %s", name, details)
        return True
    except Exception as exc:
        if required and not dry_run:
            LOG.error("%s unreachable: %s. Exiting.", name, exc)
            return False
        LOG.warning("%s unreachable%s: %s", name, " but --dry-run, continuing" if dry_run else "", exc)
        return False


def _load_config(path: Path) -> dict:
    """Load YAML configuration and validate the orchestrator settings."""
    try:
        with path.open("r", encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
    except OSError as exc:
        raise SystemExit(f"cannot read config {path}: {exc}") from exc
    if not isinstance(config, dict):
        raise SystemExit("config must contain a YAML mapping")
    settings = config.get("orchestrator")
    if not isinstance(settings, dict):
        raise SystemExit("missing config section: orchestrator")
    required = (
        "feature_dim", "policy", "cloud_timeout_sec",
        "m2_state_read_timeout_sec", "notify_timeout_sec",
        "progress_log_every_n", "output_dir",
    )
    for key in required:
        if key not in settings:
            raise SystemExit(f"missing config key: orchestrator.{key}")
    return config


def _summary_line(name: str, values: list[float], p50: bool = False) -> str:
    """Format a mean and optional percentile pair for the stderr summary."""
    mean = _mean(values)
    if p50:
        return f"  {name}: mean={_round(mean)} p50={_round(_percentile(values, 50))} p95={_round(_percentile(values, 95))} p99={_round(_percentile(values, 99))}"
    return f"  {name}: mean={_round(mean)} p95={_round(_percentile(values, 95))}"


def main() -> int:
    """Parse options, run one testbed experiment, and write its logs and summary."""
    parser = argparse.ArgumentParser(description="PRO-IDS Phase B request orchestrator")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--input-csv")
    parser.add_argument("--num-requests", type=int, default=1000)
    parser.add_argument("--policy")
    parser.add_argument("--output-dir")
    parser.add_argument("--label", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cloud-timeout", type=float)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    config = _load_config(config_path)
    settings = config["orchestrator"]
    edge_base = f"http://{config['edge']['host']}:{config['edge']['port']}"
    cloud_base = (
        f"http://{config['cloud']['advertised_host']}:{config['cloud']['port']}"
    )
    m2_edge_base = f"http://127.0.0.1:{config['m2']['edge_state_port']}"
    policy = args.policy or settings["policy"]
    if policy not in {"edge-only", "cloud-only", "alternate"}:
        parser.error("--policy must be edge-only, cloud-only, or alternate")
    if args.num_requests < 0:
        parser.error("--num-requests cannot be negative")
    if args.cloud_timeout is not None and args.cloud_timeout <= 0:
        parser.error("--cloud-timeout must be positive")

    if args.input_csv and not Path(args.input_csv).expanduser().is_file():
        print(f"input CSV does not exist: {args.input_csv}", file=sys.stderr)
        return 2

    output_dir = Path(
        args.output_dir or settings["output_dir"]
    ).expanduser()
    if not output_dir.is_absolute():
        output_dir = (config_path.parent / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cloud_timeout_value = (
        args.cloud_timeout
        if args.cloud_timeout is not None
        else float(settings["cloud_timeout_sec"])
    )
    run_id = (
        f"{args.label or 'unlabeled'}_{policy}_{iso_filename_token()}_{uuid4().hex}"
    )
    jsonl_path = output_dir / f"{run_id}.jsonl"
    summary_path = output_dir / "summary_runs.csv"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    LOG.info("orchestrator starting")
    LOG.info("  config: %s", config_path)
    LOG.info("  policy: %s", policy)
    LOG.info("  num_requests: %d", args.num_requests)
    LOG.info("  input: %s", args.input_csv or "synthetic")
    LOG.info("  output_dir: %s", output_dir)
    LOG.info("  run_id: %s", run_id)
    LOG.info("  cloud_timeout: %ss", cloud_timeout_value)
    LOG.info("  m2_state_read_timeout: %ss", settings["m2_state_read_timeout_sec"])
    LOG.info("  dry_run: %s", args.dry_run)

    if not _health_check(
        f"{edge_base}/health", "m5_edge",
        required=True, dry_run=args.dry_run,
    ) and not args.dry_run:
        return 2

    if not _health_check(
        f"{cloud_base}/health", "m5_cloud",
        required=True, dry_run=args.dry_run,
    ) and not args.dry_run:
        return 2

    _health_check(
        f"{m2_edge_base}/health", "m2_edge",
        required=False, dry_run=args.dry_run,
    )

    try:
        if args.input_csv:
            source = RequestSource.from_csv(
                Path(args.input_csv),
                feature_dim=int(settings["feature_dim"]),
                num_requests=args.num_requests,
            )
        else:
            source = RequestSource.synthetic(
                n=args.num_requests,
                feature_dim=int(settings["feature_dim"]),
                seed=args.seed,
            )
    except (OSError, ValueError) as exc:
        LOG.error("could not initialize request source: %s", exc)
        return 2

    edge_ok = edge_timeout = edge_error = 0
    cloud_ok = cloud_timeout = cloud_error = 0
    latency_total_ok: list[float] = []
    latency_dispatch_ok: list[float] = []
    latency_m2_read_ok: list[float] = []
    edge_latency: list[float] = []
    cloud_latency: list[float] = []
    confusion = {"tn": 0, "fp": 0, "fn": 0, "tp": 0}
    m2_state_ok_count = m2_state_null_count = 0
    m2_h_c_values: list[float] = []
    m2_rtt_values: list[float] = []
    m2_q_e_values: list[float] = []
    m2_q_c_values: list[float] = []
    n_c = 0
    started_utc = iso_now()
    t_run_start = time.perf_counter()
    notes = ""
    idx = -1

    def handle_signal(signum, frame):
        """Convert termination signals into the normal partial-summary path."""
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    with JsonlWriter(jsonl_path) as jsonl:
        try:
            for idx, (request_id, features, true_label) in enumerate(source):
                if idx >= args.num_requests:
                    break

                _notify(
                    "notify_arrival", None,
                    float(settings["notify_timeout_sec"]),
                    m2_edge_base,
                )

                t0_m2 = time.perf_counter()
                state = None
                try:
                    response = requests.get(
                        f"{m2_edge_base}/state",
                        timeout=float(settings["m2_state_read_timeout_sec"]),
                    )
                    response.raise_for_status()
                    state = response.json()
                    m2_state_ok_count += 1
                    for key, target in (
                        ("h_c_ms", m2_h_c_values),
                        ("rtt_ms", m2_rtt_values),
                        ("q_e", m2_q_e_values),
                        ("q_c", m2_q_c_values),
                    ):
                        value = state.get(key)
                        if isinstance(value, (int, float)) and not isinstance(value, bool):
                            target.append(float(value))
                except Exception:
                    m2_state_null_count += 1
                    state = None
                latency_m2_read_ms = (time.perf_counter() - t0_m2) * 1000.0
                features_list = features.tolist()

                if policy == "edge-only":
                    route = "edge"
                elif policy == "cloud-only":
                    route = "cloud"
                else:
                    route = "edge" if idx % 2 == 0 else "cloud"

                if args.dry_run:
                    jsonl.write({
                        "request_id": request_id,
                        "index": idx,
                        "timestamp_utc": iso_now(),
                        "route": route,
                        "status": "dry_run",
                        "latency_total_ms": None,
                        "latency_dispatch_ms": None,
                        "latency_m2_read_ms": round(latency_m2_read_ms, 6),
                        "prob_attack": None,
                        "predicted_label": None,
                        "true_label": true_label,
                        "service_latency_ms": None,
                        "error_msg": None,
			"features": features_list,
                        "s_t_snapshot": state,
                    })
                else:
                    if route == "cloud":
                        n_c += 1
                        _notify(
                            "notify_n_c", {"n_c": n_c},
                            float(settings["notify_timeout_sec"]),
                            m2_edge_base,
                        )

                    dispatch_start = time.perf_counter()
                    status = "ok"
                    error_msg = None
                    probability = None
                    predicted_label = None
                    service_latency_ms = None
                    bytes_sent = len(features) * 4
                    bytes_received = 0

                    try:
                        if route == "edge":
                            url = f"{edge_base}/predict"
                            timeout = 5.0
                        else:
                            url = f"{cloud_base}/predict"
                            timeout = cloud_timeout_value
                        response = requests.post(
                            url,
                            json={"request_id": request_id, "x": features_list},
                            timeout=timeout,
                        )
                        response.raise_for_status()
                        body = response.json()
                        probability = body.get("prob_attack", body.get("probability"))
                        predicted_label = body.get(
                            "predicted_label", body.get("prediction")
                        )
                        service_latency_ms = body.get(
                            "service_latency_ms", body.get("latency_ms")
                        )
                        if probability is None or predicted_label is None:
                            raise ValueError("M5 response missing probability or prediction")
                        probability = float(probability)
                        predicted_label = int(predicted_label)
                        if service_latency_ms is not None:
                            service_latency_ms = float(service_latency_ms)
                        bytes_received = len(response.content)
                    except requests.Timeout:
                        status = "timeout"
                        error_msg = "request timed out"
                    except requests.RequestException as exc:
                        status = "error"
                        error_msg = str(exc)
                    except (ValueError, TypeError, KeyError) as exc:
                        status = "error"
                        error_msg = f"invalid M5 response: {exc}"
                    finally:
                        latency_dispatch_ms = (
                            time.perf_counter() - dispatch_start
                        ) * 1000.0
                        if route == "cloud":
                            n_c -= 1
                            _notify(
                                "notify_n_c", {"n_c": n_c},
                                float(settings["notify_timeout_sec"]),
                                m2_edge_base,
                            )

                    _notify(
                        "notify_completion", None,
                        float(settings["notify_timeout_sec"]), m2_edge_base,
                    )
                    _notify(
                        "notify_bytes",
                        {
                            "bytes_sent": bytes_sent,
                            "bytes_received": bytes_received,
                        },
                        float(settings["notify_timeout_sec"]),
                        m2_edge_base,
                    )

                    latency_total_ms = latency_m2_read_ms + latency_dispatch_ms
                    if status == "ok":
                        if route == "edge":
                            edge_ok += 1
                            edge_latency.append(latency_dispatch_ms)
                        else:
                            cloud_ok += 1
                            cloud_latency.append(latency_dispatch_ms)
                        latency_total_ok.append(latency_total_ms)
                        latency_dispatch_ok.append(latency_dispatch_ms)
                        latency_m2_read_ok.append(latency_m2_read_ms)
                        if true_label is not None:
                            if predicted_label == 1 and true_label == 1:
                                confusion["tp"] += 1
                            elif predicted_label == 1 and true_label == 0:
                                confusion["fp"] += 1
                            elif predicted_label == 0 and true_label == 1:
                                confusion["fn"] += 1
                            else:
                                confusion["tn"] += 1
                    elif status == "timeout":
                        if route == "edge":
                            edge_timeout += 1
                        else:
                            cloud_timeout += 1
                    elif route == "edge":
                        edge_error += 1
                    else:
                        cloud_error += 1

                    jsonl.write({
                        "request_id": request_id,
                        "index": idx,
                        "timestamp_utc": iso_now(),
                        "route": route,
                        "status": status,
                        "latency_total_ms": round(latency_total_ms, 6),
                        "latency_dispatch_ms": round(latency_dispatch_ms, 6),
                        "latency_m2_read_ms": round(latency_m2_read_ms, 6),
                        "prob_attack": probability,
                        "predicted_label": predicted_label,
                        "true_label": true_label,
                        "service_latency_ms": service_latency_ms,
                        "error_msg": error_msg,
                        "features": features_list,
                        "s_t_snapshot": state,
                    })

                every_n = int(settings["progress_log_every_n"])
                if every_n > 0 and (idx + 1) % every_n == 0:
                    LOG.info(
                        "processed %d/%d | edge ok=%d timeout=%d error=%d | "
                        "cloud ok=%d timeout=%d error=%d | m2_state_ok=%d",
                        idx + 1, args.num_requests,
                        edge_ok, edge_timeout, edge_error,
                        cloud_ok, cloud_timeout, cloud_error,
                        m2_state_ok_count,
                    )
        except KeyboardInterrupt:
            notes = f"interrupted at request {max(idx, 0)}"
            LOG.warning(notes)

    t_run_end = time.perf_counter()
    tn, fp, fn, tp = (
        confusion["tn"], confusion["fp"], confusion["fn"], confusion["tp"]
    )
    labeled = tn + fp + fn + tp
    if labeled:
        accuracy = (tn + tp) / labeled
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )
        fpr = fp / (fp + tn) if fp + tn else None
        fnr = fn / (fn + tp) if fn + tp else None
    else:
        accuracy = f1 = fpr = fnr = None

    num_completed = edge_ok + cloud_ok
    drop_rate = (
        (args.num_requests - num_completed) / args.num_requests
        if args.num_requests else None
    )
    summary_row = {
        "run_id": run_id,
        "label": args.label,
        "policy": policy,
        "num_requests": args.num_requests,
        "num_completed": num_completed,
        "edge_ok": edge_ok,
        "edge_timeout": edge_timeout,
        "edge_error": edge_error,
        "cloud_ok": cloud_ok,
        "cloud_timeout": cloud_timeout,
        "cloud_error": cloud_error,
        "drop_rate": _round(drop_rate),
        "total_duration_sec": _round(t_run_end - t_run_start, 3),
        "latency_total_mean_ms": _round(_mean(latency_total_ok)),
        "latency_total_p50_ms": _round(_percentile(latency_total_ok, 50)),
        "latency_total_p95_ms": _round(_percentile(latency_total_ok, 95)),
        "latency_total_p99_ms": _round(_percentile(latency_total_ok, 99)),
        "latency_dispatch_mean_ms": _round(_mean(latency_dispatch_ok)),
        "latency_dispatch_p95_ms": _round(_percentile(latency_dispatch_ok, 95)),
        "latency_m2_read_mean_ms": _round(_mean(latency_m2_read_ok)),
        "latency_m2_read_p95_ms": _round(_percentile(latency_m2_read_ok, 95)),
        "edge_latency_mean_ms": _round(_mean(edge_latency)),
        "edge_latency_p95_ms": _round(_percentile(edge_latency, 95)),
        "cloud_latency_mean_ms": _round(_mean(cloud_latency)),
        "cloud_latency_p95_ms": _round(_percentile(cloud_latency, 95)),
        "accuracy": _round(accuracy),
        "f1": _round(f1),
        "fpr": _round(fpr),
        "fnr": _round(fnr),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "m2_state_ok_count": m2_state_ok_count,
        "m2_state_null_count": m2_state_null_count,
        "m2_mean_h_c_ms": _round(_mean(m2_h_c_values)),
        "m2_mean_rtt_ms": _round(_mean(m2_rtt_values)),
        "m2_mean_q_e": _round(_mean(m2_q_e_values)),
        "m2_mean_q_c": _round(_mean(m2_q_c_values)),
        "started_utc": started_utc,
        "finished_utc": iso_now(),
        "git_commit": _git_commit(),
        "notes": notes,
    }
    with SummaryCsvWriter(summary_path, list(SUMMARY_FIELDS)) as writer:
        writer.append(summary_row)

    print("orchestrator finished", file=sys.stderr)
    print(f"  run_id: {run_id}", file=sys.stderr)
    print(f"  duration: {t_run_end - t_run_start:.3f}s", file=sys.stderr)
    print(f"  requests: {args.num_requests}", file=sys.stderr)
    print(
        f"  edge: ok={edge_ok} timeout={edge_timeout} error={edge_error}",
        file=sys.stderr,
    )
    print(
        f"  cloud: ok={cloud_ok} timeout={cloud_timeout} error={cloud_error}",
        file=sys.stderr,
    )
    print(f"  drop_rate: {_round(drop_rate)}", file=sys.stderr)
    print(_summary_line("latency_total", latency_total_ok, p50=True), file=sys.stderr)
    print(_summary_line("latency_dispatch", latency_dispatch_ok), file=sys.stderr)
    print(
        f"  m2_state_ok: {m2_state_ok_count}, m2_state_null: {m2_state_null_count}",
        file=sys.stderr,
    )
    print(f"  jsonl: {jsonl_path}", file=sys.stderr)
    print(f"  summary: {summary_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
