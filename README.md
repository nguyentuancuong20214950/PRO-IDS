# PRO-IDS Edge–Cloud Testbed (Phase B)

Phase B provides request sourcing, M5 edge/cloud inference, M2 telemetry, and hardcoded routing for E2 plumbing and baseline measurements. M3 risk/latency predictors and adaptive M4 decisions are not implemented. Routing policies are `edge-only`, `cloud-only`, and `alternate`.

## Architecture

```text
┌────────────────────────────────────┐              ┌────────────────────────────────────┐
│ EDGE VM: 192.168.56.10              │              │ CLOUD VM: 192.168.56.11             │
│                                    │              │                                    │
│ M5-edge service :8001              │              │ M5-cloud service :8002             │
│ M2-edge state  :8004 ──────────────┼─ telemetry ─►│ M2-cloud telemetry :8003           │
│             ▲                      │              │                                    │
│ orchestrator ├──── /predict ────────┼─────────────►│ M5-cloud                           │
│              └──── /predict ──────►│ M5-edge      │                                    │
└────────────────────────────────────┘              └────────────────────────────────────┘
```

M2 obtains M5 metrics and does not consume detector prediction responses. The `q_e` and `q_c` state values are active-request counts from M5's `requests_in_flight` metrics. They are labeled as proxies for waiting-queue occupancy because M5 does not expose a separate waiting queue length.

## Folder layout

```text
.
├── config.yaml
├── requirements.txt
├── request_source.py
├── logging_utils.py
├── orchestrator.py
├── npz_model.py
├── M2/
│   ├── m2_state.py
│   ├── m2_cloud_telemetry.py
│   └── m2_edge_telemetry.py
├── M5/
│   ├── m5_edge_service.py
│   └── m5_cloud_service.py
├── models/
│   ├── AD1_int5.npz
│   └── AD1_model.joblib
├── benchmark/
│   ├── benchmark_npz.py
│   ├── benchmark_sklearn.py
│   └── test_npz_model.py
├── data/
└── results/
```

## Prerequisites and installation

- Two Ubuntu VMs on a host-only network; this guide assumes edge `192.168.56.10` and cloud `192.168.56.11`.
- Python 3.10 or newer on both VMs.
- Copy the testbed files and model assets to both VMs. The edge VM needs `models/AD1_int5.npz` and `npz_model.py`; the cloud VM needs `models/AD1_model.joblib`.
- Allow the edge VM to reach cloud ports 8002 and 8003.
- Put the dataset CSV on the edge VM when using CSV input. Config paths resolve relative to `config.yaml`.

Install dependencies once from the project root on each VM:

```bash
python3 -m pip install -r requirements.txt
```

## Start services

Start each service in a separate terminal, in this order:

1. On the cloud VM, start M5-cloud:

   ```bash
   python3 M5/m5_cloud_service.py --config config.yaml
   ```

2. On the cloud VM, start M2-cloud:

   ```bash
   python3 M2/m2_cloud_telemetry.py --config config.yaml
   ```

3. On the edge VM, start M5-edge:

   ```bash
   python3 M5/m5_edge_service.py --config config.yaml
   ```

4. On the edge VM, start M2-edge:

   ```bash
   python3 M2/m2_edge_telemetry.py --config config.yaml
   ```

5. On the edge VM, run the orchestrator after the four services are healthy:

   ```bash
   python3 orchestrator.py --config config.yaml --input-csv data/CICIDS2017_test.csv \
     --num-requests 1000 --policy edge-only --label run1
   ```

Omit `--input-csv` for deterministic synthetic input, intended for plumbing checks only. The orchestrator warns and continues if M2-edge is unavailable; those runs have null state snapshots and should not be used for state-aware analysis.

Shut down in reverse order: orchestrator, M2-edge, M5-edge, M2-cloud, M5-cloud. Restart the M5 and M2 services between policy runs when independent queue/counter state is required; service counters persist for the lifetime of each service process.

## Routing policies

- `edge-only`: every request goes to M5-edge.
- `cloud-only`: every request goes to M5-cloud.
- `alternate`: even-index requests go to edge; odd-index requests go to cloud.

Routing is hardcoded and does not use M2 state.

## Outputs

- `results/<run_id>.jsonl`: one record per request. Run IDs include a timestamp and random suffix to avoid collisions between rapid runs with the same label and policy.
- `results/summary_runs.csv`: appended run summaries.
- `results/m2_edge_*.jsonl` and `results/m2_cloud_*.jsonl`: telemetry samples.

Per-request records include request identity, route/status, local latency components, predictions and labels, error details, and the pre-dispatch `s_t_snapshot` where M2-edge is available.

## Smoke checks

After starting all four services, check health and the M2 state endpoints from the edge VM:

```bash
curl http://127.0.0.1:8001/health
curl http://192.168.56.11:8002/health
curl http://127.0.0.1:8004/health
curl http://192.168.56.11:8003/telemetry
curl http://127.0.0.1:8004/state
```

Run a dry run, then a short edge-only and cloud-only inference run:

```bash
python3 orchestrator.py --config config.yaml --num-requests 2 --dry-run --label smoke-dry
python3 orchestrator.py --config config.yaml --num-requests 20 --policy edge-only --label smoke-edge
python3 orchestrator.py --config config.yaml --num-requests 20 --policy cloud-only --label smoke-cloud
```

Expected results: health endpoints return `status: ok`; state and telemetry endpoints return JSON snapshots after their first sample; the dry run writes `dry_run` records without calling `/predict`; both inference runs write 20 request records and one summary row with the selected policy.

## Not implemented in Phase B

- M3 risk and latency predictors.
- Adaptive M4 decisions.
- Deadline enforcement and explicit waiting-queue limits.
- `netem`, `tcpreplay`, or load generation.
