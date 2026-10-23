"""
NpzMLP: inference for PRO-IDS quantized MLP models stored as .npz.

File format (from AD1_int5.npz / AD1_int6.npz):
  - weight_L_int{bits}_values : int8 array  (quantized weights, layer L)
  - weight_L_scale            : float32 (1,) (per-tensor scale, layer L)
  - bias_L_fp32               : float32 (dequantized biases, layer L)
  - weight_bit_width          : [5] or [6]
  - activation                : ['relu']      hidden activation
  - hidden_layers             : [16, 8]        hidden layer sizes
  - input_features            : [77]
  - classes                   : [0, 1]
  - quantization              : ['per_tensor_symmetric_int5'] etc.

Dequantization is symmetric per-tensor:
    W_float = W_int.astype(float32) * scale

Output activation is assumed sigmoid (verified via AD1_model.joblib,
which reports out_activation_ = "logistic").
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np


class NpzMLP:
    """Minimal forward pass for a quantized MLP stored in an .npz file."""

    def __init__(self, path: str | Path, out_activation: str = "sigmoid") -> None:
        """
        Args:
            path: Path to a .npz model file.
            out_activation: "sigmoid" (default) or "identity".
        """
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"npz file not found: {self.path}")

        # np.load returns a lazy dict-like object. We must keep it referenced
        # so that the arrays stay memory-mapped; we copy the ones we need.
        with np.load(self.path, allow_pickle=False) as data:
            self.format_version = int(data["format_version"][0])
            self.quantization = str(data["quantization"][0])
            self.activation = str(data["activation"][0])
            self.classes = data["classes"].tolist()
            self.input_features = int(data["input_features"][0])
            self.hidden_layers = data["hidden_layers"].tolist()
            self.weight_bit_width = int(data["weight_bit_width"][0])

            # Number of layers = len(hidden_layers) + 1 output layer
            self.n_layers = len(self.hidden_layers) + 1

            # Extract per-layer weights (int) + scales + biases (float)
            self.weights_int: list[np.ndarray] = []
            self.scales: list[float] = []
            self.biases: list[np.ndarray] = []
            for layer in range(1, self.n_layers + 1):
                w_key = f"weight_{layer}_int{self.weight_bit_width}_values"
                s_key = f"weight_{layer}_scale"
                b_key = f"bias_{layer}_fp32"
                self.weights_int.append(data[w_key].copy())
                self.scales.append(float(data[s_key][0]))
                self.biases.append(data[b_key].copy())

        # Dequantized weights: filled lazily by dequantize()
        self.weights_float: list[np.ndarray] | None = None
        self.out_activation = out_activation

    # ------------------------------------------------------------------ #
    # Core operations
    # ------------------------------------------------------------------ #
    def dequantize(self) -> None:
        """Convert int weights to float32 via per-tensor scale."""
        self.weights_float = [
            w.astype(np.float32) * np.float32(s)
            for w, s in zip(self.weights_int, self.scales)
        ]

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass. Returns P(class=1) of shape (N, 1).

        Must call dequantize() first (or pass through predict_proba which
        does it automatically if needed).

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
        last = self.n_layers - 1
        for layer in range(self.n_layers):
            a = a @ self.weights_float[layer] + self.biases[layer]
            if layer != last:
                # Hidden activation
                if self.activation == "relu":
                    a = np.maximum(a, 0.0)
                elif self.activation == "tanh":
                    a = np.tanh(a)
                elif self.activation == "logistic":
                    a = 1.0 / (1.0 + np.exp(-a))
                else:
                    raise ValueError(f"unknown hidden activation: {self.activation}")

        # Output activation
        if self.out_activation == "sigmoid":
            return 1.0 / (1.0 + np.exp(-a))
        if self.out_activation == "identity":
            return a
        raise ValueError(f"unknown output activation: {self.out_activation}")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities shaped (N, 2), matching sklearn's API.

        Column 0 = P(class=0), column 1 = P(class=1).
        """
        p1 = self.forward(X)
        return np.hstack([1.0 - p1, p1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return class labels (N,) using argmax of predict_proba."""
        return np.argmax(self.predict_proba(X), axis=1)

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #
    def info(self) -> dict[str, Any]:
        """Human-readable summary of the loaded model."""
        return {
            "path": str(self.path),
            "format": "npz_quantized",
            "format_version": self.format_version,
            "quantization": self.quantization,
            "weight_bit_width": self.weight_bit_width,
            "hidden_layer_sizes": self.hidden_layers,
            "input_features": self.input_features,
            "classes": self.classes,
            "activation": self.activation,
            "out_activation": self.out_activation,
            "coefs_shapes": [list(w.shape) for w in self.weights_int],
            "scales": self.scales,
            "param_count": int(
                sum(w.size for w in self.weights_int)
                + sum(b.size for b in self.biases)
            ),
            "weight_dtype": str(self.weights_int[0].dtype),
            "bias_dtype": str(self.biases[0].dtype),
        }