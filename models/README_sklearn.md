# benchmark_sklearn.py

Measures **local inference latency and system resource usage** of the two PRO-IDS
scikit-learn `MLPClassifier` detectors (AD1 on the edge VM, AD3 on the cloud VM)
from their `.joblib` files. It fills the `inference_latency` entries of the E1
comparison table.

**Scope:** one forward pass on an already-prepared 77-feature vector. No network,
no queueing, no preprocessing, no feature engineering.

## Install

```bash
python3 -m pip install -r requirements.txt
python3 -c "import sklearn; print(sklearn.__version__)"   # models were trained on 1.9.1
```

If the runtime sklearn version differs from the training version, sklearn emits an
`InconsistentVersionWarning` while loading. The script captures it and stores it in
`model_info.load_warnings` and `notes`. Prefer upgrading to the training version; if
that is impossible, report the mismatch alongside the numbers.

## Thread count

`--num-threads` sets `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`,
`NUMEXPR_NUM_THREADS` and `VECLIB_MAXIMUM_THREADS` **at process start**, before NumPy is
imported. It must be given on the command line; it cannot be changed after NumPy is
loaded (e.g. by importing the script from a notebook that already imported NumPy -
the script warns in that case). `threadpoolctl.threadpool_limits` is applied again
around the benchmark and `threadpool_info()` is stored in the JSON so you can confirm
the effective thread count.

## Usage

```bash
# Self-test (no .joblib needed, prints JSON to stdout, exit 0 if metrics are finite/positive)
python3 benchmark/benchmark_sklearn.py --self-test

# Edge VM
python3 benchmark/benchmark_sklearn.py --model ad1 --model-path models/AD1_model.joblib \
  --num-threads 1 --label edge-vm --inference-method predict_proba

# Cloud VM
python3 benchmark/benchmark_sklearn.py --model ad3 --model-path models/AD3_model.joblib \
  --num-threads 4 --label cloud-vm --inference-method predict_proba

# Same model, validation overhead removed (for the overhead comparison)
python3 benchmark/benchmark_sklearn.py --model ad1 --model-path models/AD1_model.joblib \
  --label edge-vm --inference-method forward_pass
```

Run both `predict_proba` and `forward_pass` for at least one model to quantify sklearn's
validation overhead. Run all arguments with `--help`.

### Inference methods

| Method | What is timed |
|---|---|
| `predict_proba` (default) | Public API: `check_array` / dtype / feature-count validation + forward pass + `[1-p, p]` stacking. Realistic production cost. |
| `predict` | Same validation overhead, returns labels. |
| `forward_pass` | `model._forward_pass_fast(X, check_input=False)`: no validation, closest to a pure forward pass (comparable to the PyTorch numbers). If the private method is missing, an equivalent NumPy forward pass over `coefs_`/`intercepts_` is used. The exact implementation is recorded in `inference_impl`. |

## Outputs (in `--output-dir`, default `results/`)

- `{model}_{label}_{timestamp}_sklearn.json` - full record: CLI args, host/library metadata,
  `threadpool_info`, model architecture, latency stats (s and ms, per batch and per sample),
  throughput, resources, status, notes.
- `{model}_{label}_{timestamp}_sklearn.md` - the same in human-readable form, including
  inference method and thread count.
- `summary_sklearn.csv` - one appended row per completed run (header written once; an
  existing file with a different header is rejected rather than corrupted). In this file
  `latency_*_ms` are **per batch** values; with `--batch-size 1` they equal per-sample values.

Optional metrics (CPU frequency/temperature, RAPL power, disk I/O during load) are `null`
when the VM does not expose them; this is normal on most hypervisors.

## Measurement notes

- Input batches are pre-built before the timed loop (pool of up to 1000 batches, cycled).
  The cycled pool stays cache-warm, so numbers represent steady-state latency.
- Latency is `time.perf_counter()` around each call; every sample is kept.
- CPU % is sampled every 100 ms during the timed loop only (warm-up excluded). For very
  short runs (<100 ms, typical for AD1) there are no ticks and the script uses one reading
  averaged over the whole window; `resources.cpu_samples` tells you which case applies.
  `cpu_percent_process` (process CPU time / wall time) is also reported and is more robust.
- RSS "peak" is the maximum of the 100 ms samples and the readings before/after the loop.
- `Ctrl-C` stops the run and writes partial JSON/MD (`status: "interrupted"`); partial runs
  are not appended to the CSV.
- `--input-mode csv` reads the first `n_features_in_` (77) numeric columns; a header row is
  detected automatically.
