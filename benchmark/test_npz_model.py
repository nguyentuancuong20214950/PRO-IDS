"""
Sanity test for NpzMLP. Verifies:
  1. File loads without error.
  2. Model metadata matches expectations.
  3. Forward pass returns correct shape and range.
  4. Output matches AD1_model.joblib (full precision) within tolerance.

Run:
    python3 benchmark/test_npz_model.py models/AD1_int5.npz models/AD1_model.joblib
    python3 benchmark/test_npz_model.py models/AD1_int6.npz models/AD1_model.joblib
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np

from npz_model import NpzMLP


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: python3 test_npz_model.py <npz_file> <joblib_file>")
        return 1

    npz_path, joblib_path = sys.argv[1], sys.argv[2]

    print("=" * 70)
    print(f"Testing {npz_path} against reference {joblib_path}")
    print("=" * 70)

    # ---- 1. Load -------------------------------------------------------- #
    print("\n[1] Loading npz model...")
    model = NpzMLP(npz_path)
    info = model.info()
    for k, v in info.items():
        print(f"    {k}: {v}")

    print("\n[2] Loading joblib reference...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = joblib.load(joblib_path)
    print(f"    hidden_layer_sizes: {ref.hidden_layer_sizes}")
    print(f"    coefs shapes: {[c.shape for c in ref.coefs_]}")
    print(f"    out_activation_: {ref.out_activation_}")

    # ---- 2. Architecture match check ------------------------------------ #
    ref_shapes = [list(c.shape) for c in ref.coefs_]
    npz_shapes = info["coefs_shapes"]
    if ref_shapes != npz_shapes:
        print(f"\n  ⚠ ARCHITECTURE MISMATCH:")
        print(f"      joblib: {ref_shapes}")
        print(f"      npz:    {npz_shapes}")
        print(f"      → outputs will differ due to architecture, not just quantize")
    else:
        print(f"\n  ✓ Architecture matches: {npz_shapes}")

    # ---- 3. Forward pass sanity ---------------------------------------- #
    print("\n[3] Running forward pass on 1000 random samples...")
    rng = np.random.default_rng(42)
    X = rng.standard_normal((1000, 77)).astype(np.float32)

    out = model.predict_proba(X)
    print(f"    output shape: {out.shape}  (expected (1000, 2))")
    print(f"    column sums all ≈ 1? max deviation: "
          f"{np.abs(out.sum(axis=1) - 1).max():.6e}")
    print(f"    P(class=1) range: [{out[:, 1].min():.4f}, {out[:, 1].max():.4f}]")
    print(f"    P(class=1) mean:  {out[:, 1].mean():.4f}")

    if out.shape != (1000, 2):
        print("    ✗ FAIL: wrong output shape")
        return 1
    if not (0.0 <= out.min() and out.max() <= 1.0):
        print("    ✗ FAIL: output outside [0, 1] — sigmoid wrong?")
        return 1
    if np.abs(out.sum(axis=1) - 1).max() > 1e-5:
        print("    ✗ FAIL: probabilities do not sum to 1")
        return 1
    print("    ✓ shape/range OK")

    # ---- 4. Compare with reference ------------------------------------- #
    print("\n[4] Comparing against joblib reference on same input...")
    ref_out = ref.predict_proba(X)[:, 1]
    npz_out = out[:, 1]
    diff = np.abs(ref_out - npz_out)

    print(f"    max abs diff:  {diff.max():.6f}")
    print(f"    mean abs diff: {diff.mean():.6f}")
    print(f"    p95 abs diff:  {np.percentile(diff, 95):.6f}")
    print(f"    % within 0.01: {(diff < 0.01).mean() * 100:.2f}%")
    print(f"    % within 0.05: {(diff < 0.05).mean() * 100:.2f}%")
    print(f"    class agreement: "
          f"{(np.argmax(out, axis=1) == ref.predict(X)).mean() * 100:.2f}%")

    # Tolerance thresholds — tune after seeing first results
    if diff.max() < 0.05:
        print("\n  ✓ PASS: output matches reference within 0.05")
        return 0
    elif diff.max() < 0.15:
        print("\n  ⚠ WARN: max diff 0.05–0.15 — check quantize scheme")
        return 0
    else:
        print("\n  ✗ FAIL: max diff > 0.15 — forward pass likely wrong")
        return 1


if __name__ == "__main__":
    sys.exit(main())
