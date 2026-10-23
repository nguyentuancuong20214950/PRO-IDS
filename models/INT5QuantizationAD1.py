
"""
INT5QuantizationAD1.py

Post-training INT5 weight quantization cho AD1.

Original model:
    D:\\PythonProject\\AD1_model.joblib

Test dataset:
    D:\\CICIDS2017\\CICIDS2017_test.csv

Outputs:
    D:\\PythonProject\\AD1_int5.npz
    D:\\PythonProject\\AD1_int5_results.json

Phương pháp:
- Per-tensor symmetric INT5 weight quantization.
- Miền lượng tử hóa: [-15, 15].
- Bias giữ nguyên FP32.
- Mô phỏng inference bằng NumPy sau khi giải lượng tử trọng số về FP32.
- Không chuẩn hóa test data lần thứ hai.

Lưu ý:
INT5 là mô phỏng lượng tử hóa trọng số 5-bit.
Trọng số trong NPZ được lưu bằng int8 container, không phải
được đóng gói vật lý thành đúng 5 bit/trọng số.
"""

import os
import json
import time
import warnings

import joblib
import numpy as np
import pandas as pd

from scipy.special import expit
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

MODEL_PATH = r"D:\PythonProject\AD1_model.joblib"
TEST_PATH = r"D:\CICIDS2017\CICIDS2017_test.csv"

OUTPUT_MODEL_PATH = r"D:\PythonProject\AD1_int5.npz"
OUTPUT_RESULTS_PATH = r"D:\PythonProject\AD1_int5_results.json"

EXPECTED_INPUT_FEATURES = 77
EXPECTED_HIDDEN_LAYERS = (16, 8)

# INT5 symmetric quantization:
# Qmax = 2^(5 - 1) - 1 = 15
INT5_MIN = -15
INT5_MAX = 15

BATCH_SIZE = 8192


# ============================================================
# 2. LOAD MODEL
# ============================================================

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Không tìm thấy model: {MODEL_PATH}"
        )

    model = joblib.load(MODEL_PATH)

    if not hasattr(model, "coefs_"):
        raise TypeError(
            "Model không có coefs_. "
            "Hãy kiểm tra AD1_model.joblib."
        )

    if not hasattr(model, "intercepts_"):
        raise TypeError(
            "Model không có intercepts_."
        )

    if len(model.coefs_) != 3:
        raise ValueError(
            f"Model cần 3 lớp trọng số, "
            f"nhưng nhận được {len(model.coefs_)}."
        )

    input_features = model.coefs_[0].shape[0]

    hidden_layers = tuple(
        weight.shape[1]
        for weight in model.coefs_[:-1]
    )

    if input_features != EXPECTED_INPUT_FEATURES:
        raise ValueError(
            f"Số feature không đúng: {input_features}, "
            f"mong đợi {EXPECTED_INPUT_FEATURES}."
        )

    if hidden_layers != EXPECTED_HIDDEN_LAYERS:
        raise ValueError(
            f"Kiến trúc hidden layer không đúng: {hidden_layers}, "
            f"mong đợi {EXPECTED_HIDDEN_LAYERS}."
        )

    if not hasattr(model, "classes_") or len(model.classes_) != 2:
        raise ValueError(
            "Script này yêu cầu MLPClassifier phân loại nhị phân."
        )

    if not np.array_equal(
        np.asarray(model.classes_),
        np.array([0, 1])
    ):
        raise ValueError(
            f"Thứ tự class phải là [0, 1], "
            f"nhận được {model.classes_}."
        )

    print("=" * 65)
    print("AD1 MODEL")
    print("=" * 65)
    print(f"Original model : {MODEL_PATH}")
    print(f"Input features : {input_features}")
    print(f"Hidden layers  : {hidden_layers}")
    print(f"Classes        : {model.classes_}")
    print(f"Activation     : {model.activation}")

    return model


# ============================================================
# 3. LOAD TEST DATA
# ============================================================

def load_test_data(model):
    if not os.path.exists(TEST_PATH):
        raise FileNotFoundError(
            f"Không tìm thấy test data: {TEST_PATH}"
        )

    df = pd.read_csv(
        TEST_PATH,
        low_memory=False
    )

    # Loại bỏ cột index thừa nếu có.
    unnamed_columns = [
        col for col in df.columns
        if str(col).lower().startswith("unnamed:")
    ]

    if unnamed_columns:
        df = df.drop(
            columns=unnamed_columns
        )

    # Ưu tiên đúng thứ tự feature đã dùng để train model.
    if hasattr(model, "feature_names_in_"):
        feature_columns = list(
            model.feature_names_in_
        )

        missing = [
            col for col in feature_columns
            if col not in df.columns
        ]

        if missing:
            raise ValueError(
                "Test CSV thiếu feature của model: "
                + ", ".join(map(str, missing[:10]))
            )

        label_candidates = [
            col for col in df.columns
            if col not in feature_columns
        ]

        if len(label_candidates) != 1:
            raise ValueError(
                "Không xác định được duy nhất cột label. "
                f"Các cột còn lại: {label_candidates}"
            )

        label_column = label_candidates[0]

    else:
        label_candidates = [
            "Label", "label",
            "Class", "class",
            "target", "Target",
            "attack", "Attack",
        ]

        label_column = next(
            (
                col for col in label_candidates
                if col in df.columns
            ),
            None
        )

        if label_column is None:
            raise ValueError(
                "Không tìm thấy cột label trong test CSV."
            )

        feature_columns = [
            col for col in df.columns
            if col != label_column
        ]

    if len(feature_columns) != EXPECTED_INPUT_FEATURES:
        raise ValueError(
            f"Test data có {len(feature_columns)} feature, "
            f"mong đợi {EXPECTED_INPUT_FEATURES}."
        )

    # Dữ liệu đã tiền xử lý trước đó; không scale lại.
    X_df = df[feature_columns].apply(
        pd.to_numeric,
        errors="coerce"
    )

    if X_df.isna().any().any():
        bad_columns = X_df.columns[
            X_df.isna().any()
        ].tolist()

        raise ValueError(
            "Feature chứa giá trị thiếu hoặc không phải số: "
            f"{bad_columns[:10]}"
        )

    X = X_df.to_numpy(
        dtype=np.float32
    )

    if not np.isfinite(X).all():
        raise ValueError(
            "Test data chứa NaN hoặc Inf."
        )

    y_numeric = pd.to_numeric(
        df[label_column],
        errors="coerce"
    )

    if y_numeric.isna().any():
        raise ValueError(
            f"Cột label '{label_column}' chứa giá trị không hợp lệ."
        )

    y = y_numeric.to_numpy(
        dtype=np.int64
    )

    if not np.isin(y, [0, 1]).all():
        raise ValueError(
            "Label phải là 0 = Benign, 1 = Attack."
        )

    print("\n" + "=" * 65)
    print("CICIDS2017 TEST DATA")
    print("=" * 65)
    print(f"Test file : {TEST_PATH}")
    print(f"Samples   : {len(X):,}")
    print(f"Features  : {X.shape[1]}")
    print(f"Benign    : {np.sum(y == 0):,}")
    print(f"Attack    : {np.sum(y == 1):,}")
    print(f"Label col : {label_column}")

    return X, y


# ============================================================
# 4. METRICS
# ============================================================

def calculate_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["Benign", "Attack"],
        output_dict=True,
        zero_division=0
    )

    return {
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
        "attack_precision": float(
            precision_score(
                y_true,
                y_pred,
                pos_label=1,
                zero_division=0
            )
        ),
        "attack_recall": float(
            recall_score(
                y_true,
                y_pred,
                pos_label=1,
                zero_division=0
            )
        ),
        "attack_f1": float(
            f1_score(
                y_true,
                y_pred,
                pos_label=1,
                zero_division=0
            )
        ),
        "benign_f1": float(
            f1_score(
                y_true,
                y_pred,
                pos_label=0,
                zero_division=0
            )
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0
            )
        ),
        "fnr": float(
            fn / (fn + tp) if (fn + tp) else 0.0
        ),
        "fpr": float(
            fp / (fp + tn) if (fp + tn) else 0.0
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "classification_report": report,
    }


def print_metrics(title, metrics):
    print("\n" + "=" * 65)
    print(title)
    print("=" * 65)

    print(f"Accuracy         : {metrics['accuracy']:.6f}")
    print(f"Attack Precision : {metrics['attack_precision']:.6f}")
    print(f"Attack Recall    : {metrics['attack_recall']:.6f}")
    print(f"Attack F1        : {metrics['attack_f1']:.6f}")
    print(f"Benign F1        : {metrics['benign_f1']:.6f}")
    print(f"Macro-F1         : {metrics['macro_f1']:.6f}")
    print(f"FNR              : {metrics['fnr']:.6f}")
    print(f"FPR              : {metrics['fpr']:.6f}")

    print("\nConfusion Matrix:")
    print(f"TN = {metrics['tn']:,}")
    print(f"FP = {metrics['fp']:,}")
    print(f"FN = {metrics['fn']:,}")
    print(f"TP = {metrics['tp']:,}")

    report = metrics["classification_report"]

    print("\nClassification Report:")
    print(
        f"{'Class':12s}"
        f"{'Precision':>12s}"
        f"{'Recall':>12s}"
        f"{'F1-score':>12s}"
        f"{'Support':>12s}"
    )

    for name in ["Benign", "Attack"]:
        item = report[name]

        print(
            f"{name:12s}"
            f"{item['precision']:12.6f}"
            f"{item['recall']:12.6f}"
            f"{item['f1-score']:12.6f}"
            f"{int(item['support']):12,d}"
        )

    print(f"\nAccuracy: {report['accuracy']:.6f}")

    for name in ["macro avg", "weighted avg"]:
        item = report[name]

        print(
            f"{name:12s}"
            f"{item['precision']:12.6f}"
            f"{item['recall']:12.6f}"
            f"{item['f1-score']:12.6f}"
            f"{int(item['support']):12,d}"
        )


# ============================================================
# 5. INT5 QUANTIZATION
# ============================================================

def quantize_weight_int5(weight):
    """
    Per-tensor symmetric INT5 quantization.

    scale = max(abs(weight)) / 15
    qweight = round(weight / scale)

    qweight nằm trong [-15, 15].
    """

    weight = np.asarray(
        weight,
        dtype=np.float32
    )

    max_abs = float(
        np.max(np.abs(weight))
    )

    if max_abs == 0.0:
        scale = 1.0

        qweight = np.zeros_like(
            weight,
            dtype=np.int8
        )
    else:
        scale = max_abs / INT5_MAX

        qweight = np.clip(
            np.round(weight / scale),
            INT5_MIN,
            INT5_MAX
        ).astype(np.int8)

    reconstructed = (
        qweight.astype(np.float32) * scale
    )

    mse = float(
        np.mean(
            (weight - reconstructed) ** 2
        )
    )

    max_error = float(
        np.max(
            np.abs(weight - reconstructed)
        )
    )

    return (
        qweight,
        np.float32(scale),
        mse,
        max_error
    )


def quantize_model(model):
    print("\n" + "=" * 65)
    print("PER-TENSOR INT5 QUANTIZATION")
    print("=" * 65)

    quantized_weights = []
    scales = []
    biases = []
    layer_info = []

    for index, (weight, bias) in enumerate(
        zip(
            model.coefs_,
            model.intercepts_
        ),
        start=1
    ):
        (
            qweight,
            scale,
            mse,
            max_error
        ) = quantize_weight_int5(weight)

        info = {
            "layer": index,
            "shape": list(weight.shape),
            "scale": float(scale),
            "quantized_min": int(qweight.min()),
            "quantized_max": int(qweight.max()),
            "weight_mse": mse,
            "maximum_weight_error": max_error,
        }

        quantized_weights.append(
            qweight
        )

        scales.append(
            scale
        )

        # Bias giữ nguyên FP32.
        biases.append(
            np.asarray(
                bias,
                dtype=np.float32
            ).copy()
        )

        layer_info.append(
            info
        )

        print(f"\nLayer {index}")
        print(f"  Shape          : {weight.shape}")
        print(f"  Scale          : {float(scale):.10e}")
        print(
            f"  Quantized range: "
            f"[{qweight.min()}, {qweight.max()}]"
        )
        print(f"  Weight MSE     : {mse:.10e}")
        print(f"  Maximum error  : {max_error:.10e}")

    return (
        quantized_weights,
        scales,
        biases,
        layer_info
    )


# ============================================================
# 6. SAVE QUANTIZED MODEL
# ============================================================

def save_quantized_model(
    model,
    quantized_weights,
    scales,
    biases
):
    save_dict = {
        "format_version": np.array(
            [1],
            dtype=np.int32
        ),
        "quantization": np.array(
            ["per_tensor_symmetric_int5"]
        ),
        "activation": np.array(
            [str(model.activation)]
        ),
        "classes": np.array(
            [0, 1],
            dtype=np.int64
        ),
        "input_features": np.array(
            [EXPECTED_INPUT_FEATURES],
            dtype=np.int32
        ),
        "hidden_layers": np.array(
            EXPECTED_HIDDEN_LAYERS,
            dtype=np.int32
        ),
        "weight_bit_width": np.array(
            [5],
            dtype=np.int32
        ),
    }

    for i, (qweight, scale, bias) in enumerate(
        zip(
            quantized_weights,
            scales,
            biases
        ),
        start=1
    ):
        save_dict[f"weight_{i}_int5_values"] = qweight

        save_dict[f"weight_{i}_scale"] = np.array(
            [scale],
            dtype=np.float32
        )

        save_dict[f"bias_{i}_fp32"] = bias

    np.savez_compressed(
        OUTPUT_MODEL_PATH,
        **save_dict
    )

    print(
        f"\nQuantized model saved: {OUTPUT_MODEL_PATH}"
    )


# ============================================================
# 7. SIMULATED INT5 INFERENCE
# ============================================================

def predict_int5_simulated(
    X,
    quantized_weights,
    scales,
    biases
):
    """
    Mô phỏng inference của mô hình đã lượng tử hóa INT5.

    Trọng số được giải lượng tử về FP32 trước khi tính toán.
    Vì vậy thời gian đo không đại diện cho native INT5 inference.
    """

    dequantized_weights = [
        qweight.astype(np.float32) * scale
        for qweight, scale in zip(
            quantized_weights,
            scales
        )
    ]

    predictions = []

    for start in range(
        0,
        len(X),
        BATCH_SIZE
    ):
        end = min(
            start + BATCH_SIZE,
            len(X)
        )

        activations = X[start:end].astype(
            np.float32,
            copy=False
        )

        # Hidden layers: Dense + ReLU.
        for i in range(
            len(dequantized_weights) - 1
        ):
            activations = (
                activations @ dequantized_weights[i]
                + biases[i]
            )

            np.maximum(
                activations,
                0,
                out=activations
            )

        # Output layer.
        logits = (
            activations @ dequantized_weights[-1]
            + biases[-1]
        )

        probabilities = expit(
            logits.reshape(-1)
        )

        batch_predictions = (
            probabilities >= 0.5
        ).astype(np.int64)

        predictions.append(
            batch_predictions
        )

    return np.concatenate(
        predictions
    )


# ============================================================
# 8. MAIN
# ============================================================

def main():
    model = load_model()

    X_test, y_test = load_test_data(
        model
    )

    # --------------------------------------------------------
    # Original FP32 baseline
    # --------------------------------------------------------

    print(
        "\nEvaluating original FP32 model..."
    )

    start_time = time.perf_counter()

    y_pred_fp32 = model.predict(
        X_test
    )

    fp32_time = (
        time.perf_counter() - start_time
    )

    fp32_metrics = calculate_metrics(
        y_test,
        y_pred_fp32
    )

    print_metrics(
        "AD1 FP32 - TEST RESULTS",
        fp32_metrics
    )

    # --------------------------------------------------------
    # INT5 quantization
    # --------------------------------------------------------

    (
        quantized_weights,
        scales,
        biases,
        layer_info
    ) = quantize_model(
        model
    )

    save_quantized_model(
        model,
        quantized_weights,
        scales,
        biases
    )

    # --------------------------------------------------------
    # INT5 simulated inference
    # --------------------------------------------------------

    print(
        "\nEvaluating INT5-quantized model..."
    )

    start_time = time.perf_counter()

    y_pred_int5 = predict_int5_simulated(
        X_test,
        quantized_weights,
        scales,
        biases
    )

    int5_time = (
        time.perf_counter() - start_time
    )

    int5_metrics = calculate_metrics(
        y_test,
        y_pred_int5
    )

    print_metrics(
        "AD1 INT5 - TEST RESULTS",
        int5_metrics
    )

    # --------------------------------------------------------
    # Size and timing
    # --------------------------------------------------------

    original_size = os.path.getsize(
        MODEL_PATH
    )

    quantized_size = os.path.getsize(
        OUTPUT_MODEL_PATH
    )

    original_mb = (
        original_size / (1024 ** 2)
    )

    quantized_mb = (
        quantized_size / (1024 ** 2)
    )

    size_reduction = (
        (1 - quantized_size / original_size) * 100
        if original_size > 0 else 0.0
    )

    fp32_throughput = (
        len(X_test) / fp32_time
        if fp32_time > 0 else 0.0
    )

    int5_throughput = (
        len(X_test) / int5_time
        if int5_time > 0 else 0.0
    )

    total_weight_parameters = sum(
        weight.size
        for weight in model.coefs_
    )

    theoretical_int5_weight_bytes = (
        total_weight_parameters * 5 / 8
    )

    print("\n" + "=" * 65)
    print("AD1 INT5 - SIZE AND TIMING")
    print("=" * 65)

    print(
        f"Original model size      : {original_mb:.6f} MB"
    )

    print(
        f"Quantized NPZ size       : {quantized_mb:.6f} MB"
    )

    print(
        f"Size reduction           : {size_reduction:.2f}%"
    )

    print(
        f"Total weight parameters  : "
        f"{total_weight_parameters:,}"
    )

    print(
        f"Theoretical INT5 weights : "
        f"{theoretical_int5_weight_bytes:.1f} bytes"
    )

    print(
        f"\nFP32 prediction time     : "
        f"{fp32_time:.6f} seconds"
    )

    print(
        f"INT5 simulation time     : "
        f"{int5_time:.6f} seconds"
    )

    print(
        f"FP32 throughput          : "
        f"{fp32_throughput:,.2f} samples/s"
    )

    print(
        f"INT5 simulation throughput: "
        f"{int5_throughput:,.2f} samples/s"
    )

    print(
        "\nNote: INT5 simulation dequantizes weights to FP32. "
        "Timing is not native INT5 performance."
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    results = {
        "model_name": "AD1",
        "quantization": "per_tensor_symmetric_int5",
        "model_path": MODEL_PATH,
        "test_path": TEST_PATH,
        "quantized_model_path": OUTPUT_MODEL_PATH,
        "test_samples": int(len(X_test)),
        "input_features": int(X_test.shape[1]),
        "hidden_layers": list(
            EXPECTED_HIDDEN_LAYERS
        ),
        "classes": [0, 1],
        "quantization_details": {
            "weight_bit_width": 5,
            "weight_dtype_storage": (
                "int8 container, values restricted to [-15, 15]"
            ),
            "weight_range": [
                INT5_MIN,
                INT5_MAX
            ],
            "bias_dtype": "float32",
            "layer_info": layer_info,
        },
        "fp32": {
            **fp32_metrics,
            "prediction_time_seconds": float(
                fp32_time
            ),
            "throughput_samples_per_second": float(
                fp32_throughput
            ),
        },
        "int5_simulated": {
            **int5_metrics,
            "prediction_time_seconds": float(
                int5_time
            ),
            "throughput_samples_per_second": float(
                int5_throughput
            ),
        },
        "model_size": {
            "original_bytes": int(
                original_size
            ),
            "quantized_npz_bytes": int(
                quantized_size
            ),
            "original_mb": float(
                original_mb
            ),
            "quantized_npz_mb": float(
                quantized_mb
            ),
            "size_reduction_percent": float(
                size_reduction
            ),
            "total_weight_parameters": int(
                total_weight_parameters
            ),
            "theoretical_int5_weight_bytes": float(
                theoretical_int5_weight_bytes
            ),
        },
        "notes": [
            "Biases remain FP32.",
            "INT5 values are stored in int8 containers in NPZ.",
            "Weights are dequantized to FP32 for simulated inference.",
            "Timing is not native INT5 hardware inference.",
            "Test data is not normalized again.",
        ],
    }

    with open(
        OUTPUT_RESULTS_PATH,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            results,
            f,
            ensure_ascii=False,
            indent=4
        )

    print(
        f"\nResults saved to: {OUTPUT_RESULTS_PATH}"
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("FP32 VS INT5 SUMMARY")
    print("=" * 65)

    print(
        f"{'Metric':<25}"
        f"{'FP32':>15}"
        f"{'INT5':>15}"
    )

    metrics_to_compare = [
        ("Accuracy", "accuracy"),
        ("Attack Precision", "attack_precision"),
        ("Attack Recall", "attack_recall"),
        ("Attack F1", "attack_f1"),
        ("Benign F1", "benign_f1"),
        ("Macro-F1", "macro_f1"),
        ("FNR", "fnr"),
        ("FPR", "fpr"),
    ]

    for label, key in metrics_to_compare:
        print(
            f"{label:<25}"
            f"{fp32_metrics[key]:>15.6f}"
            f"{int5_metrics[key]:>15.6f}"
        )

    print("\nOutput files:")
    print(
        f"Original model : {MODEL_PATH}"
    )
    print(
        f"INT5 model     : {OUTPUT_MODEL_PATH}"
    )
    print(
        f"Results        : {OUTPUT_RESULTS_PATH}"
    )

    print(
        "\nINT5 quantization completed."
    )


if __name__ == "__main__":
    warnings.filterwarnings(
        "ignore",
        category=UserWarning
    )
    main()
