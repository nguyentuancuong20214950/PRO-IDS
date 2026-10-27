"""PRO-IDS M5 edge inference service using the quantized NPZ model."""


import argparse
import logging
import math
import os
import sys
import threading
import time
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def read_config(filename):
    path = Path(filename).expanduser().resolve()
    try:
        with path.open(encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
    except OSError as exc:
        raise SystemExit(f"cannot read config {path}: {exc}") from exc

    if not isinstance(config, dict):
        raise SystemExit("config must be a YAML mapping")

    service = config.get("edge")
    orchestrator = config.get("orchestrator")
    logging_config = config.get("logging", {})
    if not isinstance(service, dict) or not isinstance(orchestrator, dict):
        raise SystemExit("config must contain edge and orchestrator sections")
    required = ("host", "port", "model_path", "num_threads")
    for key in required:
        if key not in service:
            raise SystemExit(f"missing config key: edge.{key}")

    config = dict(service)
    config["input_dim"] = orchestrator["feature_dim"]
    config["threshold"] = 0.5
    config["log_level"] = str(logging_config.get("level", "INFO")).upper()
    if config["log_level"] not in {"DEBUG", "INFO", "WARNING", "ERROR"}:
        raise SystemExit("log_level must be DEBUG, INFO, WARNING, or ERROR")
    if not isinstance(config["port"], int) or not 1 <= config["port"] <= 65535:
        raise SystemExit("port must be an integer from 1 to 65535")
    if not isinstance(config["input_dim"], int) or config["input_dim"] < 1:
        raise SystemExit("input_dim must be a positive integer")
    if not isinstance(config["num_threads"], int) or config["num_threads"] < 1:
        raise SystemExit("num_threads must be a positive integer")
    if not isinstance(config["threshold"], (int, float)) or not 0 <= config["threshold"] <= 1:
        raise SystemExit("threshold must be between 0 and 1")

    model_path = Path(str(config["model_path"])).expanduser()
    if not model_path.is_absolute():
        model_path = (path.parent / model_path).resolve()
    config["model_path"] = str(model_path)

    return config


def main():
    parser = argparse.ArgumentParser(description="PRO-IDS quantized edge service")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "config.yaml"),
    )
    args = parser.parse_args()
    config = read_config(args.config)

    # Set BLAS/OpenMP limits before importing NumPy or the model module.
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = str(config["num_threads"])

    import numpy as np
    import uvicorn
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, ConfigDict

    from .npz_model import NpzMLP

    logging.basicConfig(
        level=getattr(logging, config["log_level"]),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    logger = logging.getLogger("pro_ids.m5.edge")

    model = NpzMLP(config["model_path"])
    model.dequantize()
    model_info = model.info()

    class PredictRequest(BaseModel):
        model_config = ConfigDict(extra="forbid")
        request_id: str | None = None
        x: list[float]

    app = FastAPI(title="PRO-IDS M5 Edge", version="1.0.0")
    counts = {
        "requests_total": 0,
        "requests_in_flight": 0,
        "requests_completed": 0,
        "requests_failed": 0,
    }
    lock = threading.Lock()
    start_mono = time.perf_counter()

    @app.on_event("startup")
    def mark_startup():
        nonlocal start_mono
        start_mono = time.perf_counter()

    @app.middleware("http")
    async def track_predict_requests(request: Request, call_next):
        tracked = request.url.path == "/predict" and request.method == "POST"
        if tracked:
            with lock:
                counts["requests_total"] += 1
                counts["requests_in_flight"] += 1
        try:
            response = await call_next(request)
            if tracked:
                with lock:
                    key = "requests_failed" if response.status_code >= 400 else "requests_completed"
                    counts[key] += 1
            return response
        except Exception:
            if tracked:
                with lock:
                    counts["requests_failed"] += 1
            logger.exception("unhandled request error")
            return JSONResponse(
                status_code=500,
                content={"detail": "internal server error"},
            )
        finally:
            if tracked:
                with lock:
                    counts["requests_in_flight"] -= 1

    @app.get("/health")
    def health():
        return {"status": "ok", "service": "edge", "model_info": model_info}

    @app.get("/metrics")
    def metrics():
        with lock:
            snapshot = dict(counts)
        return {
            **snapshot,
            "uptime_sec": time.perf_counter() - start_mono,
        }

    @app.post("/predict")
    def predict(payload: PredictRequest):
        t_received_mono = time.perf_counter()
        if payload.request_id is None or not payload.request_id.strip():
            raise HTTPException(status_code=400, detail="request_id is required")
        if len(payload.x) != config["input_dim"]:
            raise HTTPException(
                status_code=400,
                detail=f"expected {config['input_dim']} features, received {len(payload.x)}",
            )
        if not all(math.isfinite(value) for value in payload.x):
            raise HTTPException(
                status_code=400,
                detail="features must all be finite numbers",
            )

        try:
            row = np.asarray(payload.x, dtype=np.float32).reshape(1, -1)
            output = np.asarray(model.predict_proba(row))
            if output.shape != (1, 2) or not np.isfinite(output).all():
                raise ValueError("model returned invalid output")
            prob_attack = min(1.0, max(0.0, float(output[0, 1])))
        except Exception as exc:
            logger.exception("inference failed for request_id=%s", payload.request_id)
            raise HTTPException(status_code=500, detail="inference failed") from exc

        t_responded_mono = time.perf_counter()
        return {
            "request_id": payload.request_id,
            "prob_attack": prob_attack,
            "predicted_label": int(1 if prob_attack >= 0.5 else 0),
            "service_latency_ms": (t_responded_mono - t_received_mono) * 1000.0,
            "t_received_mono": t_received_mono,
            "t_responded_mono": t_responded_mono,
        }

    uvicorn.run(
        app,
        host=str(config["host"]),
        port=config["port"],
        log_level=config["log_level"].lower(),
    )


if __name__ == "__main__":
    main()
