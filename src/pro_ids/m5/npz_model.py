"""
NpzMLP: inference for PRO-IDS quantized MLP models stored as .npz.

Supports two file formats:

Format A -- "old" (INT5, INT6):
  - format_version, quantization, activation, classes, input_features,
    hidden_layers, weight_bit_width
  - Per layer L (1-indexed): weight_{L}_int{bits}_values,
    weight_{L}_scale, bias_{L}_fp32
  - Example: weight_1_int5_values, weight_2_int5_values, weight_3_int5_values

Format B -- "new" (INT7):
  - bits, activation, output_activation
  - Per layer L (0-indexed): weight_{L}_int_values,
    weight_{L}_scale, bias_{L}_fp32
  - Example: weight_0_int_values, weight_1_int_values, weight_2_int_values,
             weight_3_int_values
  - Missing metadata (hidden_layers, input_features, classes, quantization,
    format_version) is inferred from weight shapes.

Dequantization is symmetric per-tensor:
    W_float = W_int.astype(float32) * scale

Output activation: Format B stores "logistic"; we normalize it to "sigmoid"
internally so predict_proba() behaves consistently across both formats.
"""

import re
from pathlib import Path
from typing import Any

import numpy as np


def _stable_sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid, avoids overflow for large |x|."""
    x = np.asarray(x, dtype=np.float32)
    out = np.empty_like(x)
    pos = x >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[neg])
    out[neg] = ex / (1.0 + ex)
    return out


class NpzMLP:
    """Minimal forward pass for a quantized MLP stored in an .npz file."""

    def __init__(self, path: str | Path, out_activation: str | None = None) -> None:
        """
        Args:
            path: Path to a .npz model file (INT5, INT6, or INT7).
            out_activation: Override for output activation. If None, use the
                model's stored value (Format B) or default to "logistic"
                (Format A).
        """
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"npz file not found: {self.path}")

        with np.load(self.path, allow_pickle=False) as data:
            files = set(data.files)

            if "bits" in files:
                format_type = "new"
            elif "weight_bit_width" in files:
                format_type = "old"
            else:
                raise ValueError(
                    "Unknown npz format: neither 'bits' nor 'weight_bit_width' "
                    f"found in {self.path}. Keys: {sorted(data.files)}"
                )

            self.format_type = format_type
            if "activation" not in files:
                raise KeyError(
                    f"Missing key 'activation' in {self.path}. "
                    f"Available keys: {sorted(data.files)}"
                )
            self.activation = str(data["activation"][0]).lower().strip()

            if format_type == "old":
                self.format_version = int(data["format_version"][0])
                self.quantization = str(data["quantization"][0])
                self.bits = int(data["weight_bit_width"][0])
                self.input_features = int(data["input_features"][0])
                self.hidden_layers = data["hidden_layers"].tolist()
                self.classes = data["classes"].tolist()
                output_activation_raw = "logistic"
                n_layers = len(self.hidden_layers) + 1
                layer_indices = list(range(1, n_layers + 1))
            else:
                self.bits = int(data["bits"][0])
                self.format_version = 1
                self.quantization = f"per_tensor_symmetric_int{self.bits}"
                if "output_activation" in files:
                    output_activation_raw = str(data["output_activation"][0])
                else:
                    output_activation_raw = "logistic"
                # Discover layer indices from weight_*_int_values keys
                layer_indices = sorted(
                    int(m.group(1))
                    for key in data.files
                    if (m := re.match(r"weight_(\d+)_int_values$", key))
                )
                if not layer_indices:
                    raise ValueError(
                        f"No 'weight_L_int_values' keys found in {self.path}. "
                        f"Keys: {sorted(data.files)}"
                    )
                # Metadata will be inferred from weights after extraction
                self.hidden_layers = []
                self.input_features = 0
                self.classes = [0, 1]

            # Extract per-layer weights, scales, biases
            self.weights_int: list[np.ndarray] = []
            self.scales: list[float] = []
            self.biases: list[np.ndarray] = []

            for L in layer_indices:
                if format_type == "old":
                    w_key = f"weight_{L}_int{self.bits}_values"
                else:
                    w_key = f"weight_{L}_int_values"
                s_key = f"weight_{L}_scale"
                b_key = f"bias_{L}_fp32"

                for k in (w_key, s_key, b_key):
                    if k not in files:
                        raise KeyError(
                            f"Missing key '{k}' in {self.path}. "
                            f"Available keys: {sorted(data.files)}"
                        )

                self.weights_int.append(data[w_key].copy())
                self.scales.append(float(data[s_key][0]))
                self.biases.append(data[b_key].copy())

            # Infer metadata for Format B (new)
            if format_type == "new":
                self.hidden_layers = [
                    int(w.shape[1]) for w in self.weights_int[:-1]
                ]
                self.input_features = int(self.weights_int[0].shape[0])

        # Normalize output activation
        if out_activation is not None:
            self.output_activation = self._normalize_activation(out_activation)
        else:
            self.output_activation = self._normalize_activation(output_activation_raw)

        self.weights_float: list[np.ndarray] | None = None

    @staticmethod
    def _normalize_activation(name: str) -> str:
        """Map activation aliases to a canonical name."""
        name = name.lower().strip()
        if name in ("logistic", "sigmoid"):
            return "sigmoid"
        if name in ("identity", "linear", "none"):
            return "identity"
        return name

    def dequantize(self) -> None:
        """Convert int weights to float32 via per-tensor scale."""
        self.weights_float = [
            w.astype(np.float32) * np.float32(s)
            for w, s in zip(self.weights_int, self.scales)
        ]

    def _apply_hidden_activation(self, a: np.ndarray) -> np.ndarray:
        if self.activation == "relu":
            return np.maximum(a, 0.0)
        if self.activation == "tanh":
            return np.tanh(a)
        if self.activation in ("logistic", "sigmoid"):
            return _stable_sigmoid(a)
        raise ValueError(f"Unknown hidden activation: {self.activation}")

    def _apply_output_activation(self, a: np.ndarray) -> np.ndarray:
        if self.output_activation == "sigmoid":
            return _stable_sigmoid(a)
        if self.output_activation == "identity":
            return a
        raise ValueError(f"Unknown output activation: {self.output_activation}")

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass. Returns P(class=1) of shape (N, 1).

        Args:
            X: Input array of shape (N, 77), dtype float32.

        Returns:
            Array of shape (N, 1) with values in [0, 1] if sigmoid,
            otherwise raw logits.
        """
        if self.weights_float is None:
            self.dequantize()

        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != self.input_features:
            raise ValueError(
                f"expected {self.input_features} features, got {X.shape[1]}"
            )

        a = X
        last = len(self.weights_float) - 1
        for i, (w, b) in enumerate(zip(self.weights_float, self.biases)):
            a = a @ w + b
            if i != last:
                a = self._apply_hidden_activation(a)

        return self._apply_output_activation(a)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities shaped (N, 2), matching sklearn's API.

        Column 0 = P(class=0), column 1 = P(class=1).
        """
        p1 = self.forward(X)
        p1 = p1.reshape(-1, 1) if p1.ndim == 1 else p1
        return np.hstack([1.0 - p1, p1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return class labels (N,) using argmax of predict_proba."""
        return np.argmax(self.predict_proba(X), axis=1)

    def info(self) -> dict[str, Any]:
        """Human-readable summary of the loaded model."""
        return {
            "path": str(self.path),
            "format": "npz_quantized",
            "format_type": self.format_type,
            "format_version": self.format_version,
            "quantization": self.quantization,
            "weight_bit_width": self.bits,
            "hidden_layer_sizes": self.hidden_layers,
            "input_features": self.input_features,
            "classes": self.classes,
            "activation": self.activation,
            "out_activation": self.output_activation,
            "coefs_shapes": [list(w.shape) for w in self.weights_int],
            "scales": self.scales,
            "param_count": int(
                sum(w.size for w in self.weights_int)
                + sum(b.size for b in self.biases)
            ),
            "weight_dtype": str(self.weights_int[0].dtype),
            "bias_dtype": str(self.biases[0].dtype),
        }


def _run_selftest(path: str) -> int:
    """Self-test: load, dequantize, run forward pass. Returns exit code."""
    m = NpzMLP(path)
    info = m.info()
    print(f"File: {path}")
    print(f"  Format type:    {info['format_type']}")
    print(f"  Hidden layers:  {info['hidden_layer_sizes']}")
    print(f"  Bits:           {info['weight_bit_width']}")
    print(f"  Param count:    {info['param_count']}")
    print(f"  Coefs shapes:   {info['coefs_shapes']}")
    print(f"  Activation:     {info['activation']}")
    print(f"  Out activation: {info['out_activation']}")
    print(f"  Input features: {info['input_features']}")

    X = np.random.default_rng(42).standard_normal(
        (3, info["input_features"])
    ).astype(np.float32)
    out = m.predict_proba(X)
    labels = m.predict(X)
    print(f"  predict_proba shape: {out.shape}")
    print(f"  Row sums (should be 1.0): {out.sum(axis=1)}")
    print(f"  Predicted labels: {labels}")

    assert out.shape == (3, 2), f"shape mismatch: {out.shape}"
    assert np.allclose(out.sum(axis=1), 1.0, atol=1e-5), "rows don't sum to 1"
    assert np.all((out >= 0) & (out <= 1)), "probabilities out of [0, 1]"
    print("  PASS")
    return 0


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python3 npz_model.py <path-to-npz>")
        raise SystemExit(1)
    raise SystemExit(_run_selftest(sys.argv[1]))
