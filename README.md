# PRO-IDS Edge–Cloud Testbed

PRO-IDS is a research testbed for edge–cloud network intrusion detection. This repository contains M1 request input, M2 telemetry, M5 edge and cloud inference, a fixed-policy orchestrator, and an offline M3 latency-data and model pipeline.

The orchestrator supports `edge-only`, `cloud-only`, and `alternate` routing. M3 latency prediction is not integrated into dispatch, and adaptive M4 logic is not implemented. Risk prediction is a separate workstream maintained by Huan.

## Repository layout

```text
PRO-IDS/
├── README.md
├── config.yaml
├── requirements.txt
├── src/
│   └── pro_ids/
│       ├── orchestrator.py
│       ├── common/logging_utils.py
│       ├── m1/request_source.py
│       ├── m2/
│       │   ├── state.py
│       │   ├── cloud_telemetry.py
│       │   └── edge_telemetry.py
│       ├── m3/
│       │   ├── latency/{collect,train,predictor}.py
│       │   └── risk/train_risk_predictor.py  # Huan's separate workstream
│       └── m5/
│           ├── npz_model.py
│           ├── edge_service.py
│           └── cloud_service.py
├── experiments/run_e2_matrix.sh
├── benchmarks/
│   ├── benchmark_npz.py
│   └── benchmark_sklearn.py
├── models/
│   ├── detector/
│   ├── latency/
│   │   ├── blocked/
│   │   └── random/
│   └── risk/                          # Huan's separate artifacts
├── artifacts/latency/                 # generated collector output
├── data/                              # local datasets; ignored by Git
├── results/                           # generated run logs; ignored by Git
└── scripts/risk/                      # Huan's inspection utilities
```

The collection metadata, training metadata, calibration, and evaluation report for each latency-model cohort are kept beside that cohort's model files. Keep these files together when copying or archiving a model cohort.

## Install and configure

Use Python 3.10 or newer. Install dependencies once from the repository root:

```bash
python3 -m pip install -r requirements.txt
```

Run commands from the repository root. Python modules are loaded from `src` using `PYTHONPATH=src`. Paths in `config.yaml` are resolved relative to the configuration file. The default service ports are:

| Service | Bind address | Port |
|---|---|---:|
| M5 edge | edge VM, localhost | 8001 |
| M5 cloud | cloud VM, all interfaces | 8002 |
| M2 cloud telemetry | cloud VM, all interfaces | 8003 |
| M2 edge state | edge VM, localhost | 8004 |

Set `cloud.advertised_host` to the address the edge VM uses to reach the cloud. The current E2 network uses edge `192.168.9.10` and cloud `192.168.9.11`; update the config and experiment script if your VM network differs.

## Start services and run a baseline

Start the services in this order. Open a terminal for each long-running service.

On the cloud VM, start each service in its own terminal.

```bash
PYTHONPATH=src python3 -m pro_ids.m5.cloud_service --config config.yaml
```

In a second cloud terminal:

```bash
PYTHONPATH=src python3 -m pro_ids.m2.cloud_telemetry --config config.yaml
```

On the edge VM, start each service in its own terminal.

```bash
PYTHONPATH=src python3 -m pro_ids.m5.edge_service --config config.yaml
```

In a second edge terminal:

```bash
PYTHONPATH=src python3 -m pro_ids.m2.edge_telemetry --config config.yaml
```

After all four services are healthy, run a baseline from the edge VM:

```bash
PYTHONPATH=src python3 -m pro_ids.orchestrator \
  --config config.yaml \
  --input-csv data/CICIDS2017v2_test.csv \
  --policy edge-only --num-requests 20 --label e2-edge
```

Use `--policy cloud-only` for the cloud baseline, or `--policy alternate` to alternate routes. Omit `--input-csv` to use deterministic synthetic inputs for a plumbing check. The input CSV should contain 77 numeric features; a `Label` or `Labels` column is used as the ground-truth label if present. Large CSV files are excluded from Git and must be copied into `data/` on each host that needs them.

For a dry run that checks orchestrator startup and logging without issuing detector predictions:

```bash
PYTHONPATH=src python3 -m pro_ids.orchestrator \
  --config config.yaml --num-requests 20 --dry-run --label smoke
```

Stop services in reverse order. Restart M5 and M2 services between independent policy runs when you need fresh counters and telemetry averages; these are reset when their services restart.

## E2 network-condition experiment

`experiments/run_e2_matrix.sh` runs the configured policy matrix and applies network perturbations with Linux traffic-control tools. Run it from the repository root on the edge VM after starting the services:

```bash
bash experiments/run_e2_matrix.sh
```

The script expects the configured CSV, `.venv` environment, `tc`, and network-shaping privileges (typically `sudo`). It writes run logs under `results/`.

## M2 state interpretation

M2 exposes `GET /state` on the edge and `GET /telemetry` on the cloud. It samples resource use, M5 request counters, orchestrator notifications, and cloud telemetry, then writes timestamped JSONL logs under `results/`.

**Queue values are proxies.** `q_e` and `q_c` come from the M5 services' `requests_in_flight` counts. Those counts represent active requests being handled, not requests waiting in an inference queue. The `q_c_is_proxy` and cloud telemetry `is_proxy` fields identify the cloud value as a proxy. Interpret both values as proxies in analysis.

`mu_e_hat` and the cloud-side `mu_c_hat` are estimated from deltas in M5 `requests_completed` over elapsed time. The edge applies a second EMA to the cloud's `mu_c_hat` to smooth polling-channel jitter. When cloud polling fails, the edge retains the last-known values and `h_c_ms` grows from the last successful poll.

## Offline M3 latency-model workflow

Each orchestrator run creates `results/<run_id>.jsonl` and appends a row to `results/summary_runs.csv`. JSONL records include timings, route and outcome, labels and predictions when available, the pre-dispatch `s_t_snapshot`, and the 77 input `features` used by the latency-data collector. Timings use the edge orchestrator's clock. Timed-out and failed requests are logged and included in drop counts; latency summaries cover successful requests.

Collect data and train models from E2 JSONL runs:

```bash
PYTHONPATH=src python3 -m pro_ids.m3.latency.collect \
  --results-dir results \
  --output-dir artifacts/latency/blocked \
  --edge-split blocked --cloud-split blocked

PYTHONPATH=src python3 -m pro_ids.m3.latency.train \
  --artifacts-dir artifacts/latency/blocked \
  --output-dir models/latency/blocked
```

The collector also supports random splits, and a `by_run` split for cloud data. Choose split modes to match the evaluation design. The checked-in `models/latency/blocked` and `models/latency/random` directories contain trained models and their metadata. Run the predictor's self-check with:

```bash
PYTHONPATH=src python3 -m pro_ids.m3.latency.predictor models/latency/blocked
```

The latency predictor currently runs offline and is not called by the orchestrator. Adaptive M3/M4 dispatch, deadline enforcement, and queue limits remain unimplemented. Risk prediction remains Huan's separate workstream.

## Detector models and benchmarks

M5 edge loads `models/detector/AD1v2_int7.npz`; M5 cloud loads `models/detector/AD1v2_model.joblib`. Both services expose `/health`, `/metrics`, and `/predict` with matching response fields.

The `benchmarks/` directory contains local model benchmark programs:

```bash
python3 benchmarks/benchmark_npz.py --help
python3 benchmarks/benchmark_sklearn.py --help
```

Generated benchmark reports should be stored under `results/` or another output directory, not alongside source files.
