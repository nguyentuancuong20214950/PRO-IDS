
# -*- coding: utf-8 -*-
"""
M3 - TRAIN RISK PREDICTORS FOR ADAPTIVE OFFLOADING
=================================================

Muc tieu:
    1. Tao nhan rui ro tu hai bo phat hien:
       - Edge: AD1v2 INT7
       - Cloud: AD1v2 FP32
    2. Chi su dung TRAIN de huan luyen risk predictor.
    3. Dung VALIDATION de danh gia trong qua trinh phat trien.
    4. Chi dung TEST de danh gia cuoi cung.
    5. Chi phi bo sot tan cong: C_FN = 10.
    6. Chi phi bao dong gia: C_FP = 1.

Nhan rui ro tren moi mau:
    FN: 10
    FP: 1
    TP/TN: 0

Luu y:
    - Script khong huan luyen lai hai mo hinh AD.
    - INT7 su dung bias_*_fp32 tu file NPZ INT7.
    - Can dam bao ba CSV da duoc tien xu ly giong voi du lieu
      dung khi huan luyen hai mo hinh AD.
"""

import os
import re
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


# ============================================================
# 1. CONFIGURATION
# ============================================================

DATA_DIR = Path(r"D:\CICIDS2017v2")
MODEL_DIR = Path(r"D:\PythonProject\2017v2")
OUTPUT_DIR = MODEL_DIR / "risk_predictor"

TRAIN_PATH = DATA_DIR / "CICIDS2017v2_train.csv"
VALIDATION_PATH = DATA_DIR / "CICIDS2017v2_validation.csv"
TEST_PATH = DATA_DIR / "CICIDS2017v2_test.csv"

CLOUD_MODEL_PATH = MODEL_DIR / "AD1v2_model.joblib"
EDGE_MODEL_PATH = MODEL_DIR / "AD1v2_int7.npz"

LABEL_COLUMN = "Label"

# Chi phi phat hien sai
C_FN = 8.0
C_FP = 1.0
C_MAX = max(C_FN, C_FP)

# Gia tri nay chi gioi han so mau moi tap de tao nhan va huan luyen.
# Dat None neu muon su dung toan bo tap.
MAX_TRAIN_ROWS = 250_000
MAX_VALIDATION_ROWS = 100_000
MAX_TEST_ROWS = 100_000

RANDOM_STATE = 42
BATCH_SIZE = 8192

# So cay cua risk predictor
N_ESTIMATORS = 200
MAX_DEPTH = 20
MIN_SAMPLES_LEAF = 5
N_JOBS = -1

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. GENERAL UTILITIES
# ============================================================

def print_section(title):
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


def check_files():
    paths = {
        "Train CSV": TRAIN_PATH,
        "Validation CSV": VALIDATION_PATH,
        "Test CSV": TEST_PATH,
        "Cloud model": CLOUD_MODEL_PATH,
        "Edge INT7 model": EDGE_MODEL_PATH,
    }

    missing = [f"{name}: {path}" for name, path in paths.items()
               if not path.exists()]

    if missing:
        raise FileNotFoundError(
            "Khong tim thay cac file sau:\n" + "\n".join(missing)
        )


def normalize_label(series):
    """
    Chuyen nhan ve:
        0 = Benign
        1 = Attack

    Chap nhan nhan so 0/1 hoac chuoi Benign/Attack.
    """
    if pd.api.types.is_numeric_dtype(series):
        values = pd.to_numeric(series, errors="coerce")
        unique_values = set(values.dropna().unique())

        if not unique_values.issubset({0, 1, 0.0, 1.0}):
            raise ValueError(
                f"Label dang so khong phai nhi phan 0/1: "
                f"{sorted(unique_values)[:20]}"
            )

        return values.astype("Int64")

    text = series.astype(str).str.strip().str.lower()

    benign_names = {
        "benign", "benigntraffic", "normal", "0", "0.0"
    }

    attack_names = {
        "attack", "malicious", "1", "1.0"
    }

    result = pd.Series(index=series.index, dtype="float64")

    result[text.isin(benign_names)] = 0
    result[text.isin(attack_names)] = 1

    # Ho tro du lieu goc co nhieu ten loai tan cong.
    unknown = result.isna() & series.notna()

    if unknown.any():
        unknown_values = text[unknown].unique()[:20]
        raise ValueError(
            "Phat hien nhan chua duoc anh xa sang Benign/Attack. "
            f"Vi du: {unknown_values.tolist()}. "
            "Neu CSV chua duoc nhi phan hoa, hay sua normalize_label()."
        )

    return result.astype("Int64")


def load_dataset(path, expected_features=None):
    print(f"\nDoc: {path}")

    df = pd.read_csv(path, low_memory=False)

    if LABEL_COLUMN not in df.columns:
        raise ValueError(
            f"Khong tim thay cot nhan '{LABEL_COLUMN}' trong {path}"
        )

    y = normalize_label(df[LABEL_COLUMN])

    # Neu da co danh sach feature tu tap train, su dung dung thu tu do.
    if expected_features is None:
        excluded = {LABEL_COLUMN}
        feature_columns = [
            c for c in df.columns
            if c not in excluded
            and pd.api.types.is_numeric_dtype(df[c])
        ]
    else:
        missing = [c for c in expected_features if c not in df.columns]
        if missing:
            raise ValueError(
                f"{path} thieu feature: {missing[:20]}"
            )
        feature_columns = list(expected_features)

    if not feature_columns:
        raise ValueError(f"Khong tim thay feature so trong {path}")

    # Khong tu dong thay NaN/Inf bang gia tri tuy y:
    # neu du lieu can xu ly, dung mot quy trinh tien xu ly thong nhat.
    X = df[feature_columns].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)

    valid_mask = y.notna() & X.notna().all(axis=1)
    invalid_count = int((~valid_mask).sum())

    X = X.loc[valid_mask].astype(np.float32)
    y = y.loc[valid_mask].astype(np.int8)

    print(f"  Rows goc       : {len(df):,}")
    print(f"  Rows hop le    : {len(X):,}")
    print(f"  Rows bo        : {invalid_count:,}")
    print(f"  So features    : {X.shape[1]}")
    print(f"  Benign         : {(y == 0).sum():,}")
    print(f"  Attack         : {(y == 1).sum():,}")

    return X, y, feature_columns


def deterministic_sample(X, y, max_rows, seed):
    """Lay mau ngau nhien co the tai lap, khong thay doi du lieu goc."""
    if max_rows is None or len(X) <= max_rows:
        return X, y

    rng = np.random.default_rng(seed)
    indices = np.sort(
        rng.choice(len(X), size=max_rows, replace=False)
    )

    return (
        X.iloc[indices].reset_index(drop=True),
        y.iloc[indices].reset_index(drop=True),
    )


# ============================================================
# 3. LOAD CLOUD FP32 MODEL
# ============================================================

def load_cloud_model():
    print_section("LOAD CLOUD FP32 MODEL")

    model = joblib.load(CLOUD_MODEL_PATH)

    if not hasattr(model, "predict"):
        raise TypeError("Cloud model khong co method predict().")

    print(f"Model path: {CLOUD_MODEL_PATH}")
    print(f"Model type: {type(model).__name__}")

    if hasattr(model, "hidden_layer_sizes"):
        print(f"Hidden layers: {model.hidden_layer_sizes}")

    if hasattr(model, "n_features_in_"):
        print(f"Model input features: {model.n_features_in_}")

    return model


def cloud_predict(model, X):
    """
    Goi predict() cua mo hinh Cloud.

    Neu joblib la sklearn Pipeline, cac buoc tien xu ly trong pipeline
    se duoc ap dung tu dong.
    """
    predictions = []

    for start in range(0, len(X), BATCH_SIZE):
        batch = X.iloc[start:start + BATCH_SIZE]

        pred = np.asarray(model.predict(batch)).reshape(-1)

        # Mo hinh AD duoc ky vong la bo phan loai nhi phan.
        # predict() cua sklearn tra ve nhan 0/1.
        pred = (pred >= 0.5).astype(np.int8)

        predictions.append(pred)

    return np.concatenate(predictions)


# ============================================================
# 4. LOAD EDGE INT7 MODEL FROM NPZ
# ============================================================

def scalar_from_npz(data, key, default=None):
    if key not in data:
        return default

    value = np.asarray(data[key]).reshape(-1)

    if len(value) == 0:
        return default

    return value[0].item()


def find_layer_indices(data):
    indices = set()

    for key in data.files:
        match = re.match(r"weight_(\d+)_int_values$", key)
        if match:
            indices.add(int(match.group(1)))

    if not indices:
        # Ho tro ten key cu co hau to khac.
        for key in data.files:
            match = re.match(r"weight_(\d+).*values$", key)
            if match:
                indices.add(int(match.group(1)))

    if not indices:
        raise ValueError(
            "Khong tim thay cac mang weight int trong file INT7 NPZ. "
            f"Cac key hien co: {data.files}"
        )

    return sorted(indices)


def load_edge_model():
    print_section("LOAD EDGE INT7 MODEL")

    data = np.load(EDGE_MODEL_PATH, allow_pickle=False)

    layer_indices = find_layer_indices(data)
    layers = []

    for layer_idx in layer_indices:
        weight_key_candidates = [
            f"weight_{layer_idx}_int_values",
        ]

        weight_key = next(
            (key for key in weight_key_candidates if key in data),
            None,
        )

        if weight_key is None:
            raise KeyError(
                f"Khong tim thay weight int cho layer {layer_idx}"
            )

        scale_key = f"weight_{layer_idx}_scale"
        bias_key = f"bias_{layer_idx}_fp32"

        if scale_key not in data:
            raise KeyError(f"Thieu {scale_key} trong file INT7")

        if bias_key not in data:
            raise KeyError(
                f"Thieu {bias_key}. Khong the xac nhan bias cua INT7; "
                "khong duoc thay bang bias cua Cloud FP32."
            )

        q_weight = np.asarray(data[weight_key], dtype=np.float32)
        scale = np.asarray(data[scale_key], dtype=np.float32).reshape(-1)
        bias = np.asarray(data[bias_key], dtype=np.float32).reshape(-1)

        if len(scale) == 1:
            weight = q_weight * float(scale[0])
        elif q_weight.ndim == 2 and len(scale) == q_weight.shape[1]:
            # Ho tro scale theo cot/output neu file su dung per-channel.
            weight = q_weight * scale.reshape(1, -1)
        else:
            raise ValueError(
                f"Scale khong tuong thich voi weight layer {layer_idx}: "
                f"weight={q_weight.shape}, scale={scale.shape}"
            )

        if weight.shape[1] != len(bias):
            raise ValueError(
                f"Shape khong khop o layer {layer_idx}: "
                f"weight={weight.shape}, bias={bias.shape}"
            )

        layers.append({
            "weight": weight,
            "bias": bias,
        })

        print(
            f"Layer {layer_idx}: "
            f"W={weight.shape}, b={bias.shape}, "
            f"scale={scale.tolist()}"
        )

    activation = str(
        scalar_from_npz(data, "activation", "relu")
    ).lower()

    output_activation = str(
        scalar_from_npz(data, "output_activation", "logistic")
    ).lower()

    print(f"Activation: {activation}")
    print(f"Output activation: {output_activation}")
    print(f"Model path: {EDGE_MODEL_PATH}")

    data.close()

    return {
        "layers": layers,
        "activation": activation,
        "output_activation": output_activation,
    }


def apply_activation(z, activation):
    if activation in {"relu", "rectified_linear"}:
        return np.maximum(z, 0)

    if activation in {"tanh"}:
        return np.tanh(z)

    if activation in {"logistic", "sigmoid"}:
        z = np.clip(z, -80, 80)
        return 1.0 / (1.0 + np.exp(-z))

    if activation in {"identity", "linear", "none"}:
        return z

    raise ValueError(f"Activation chua duoc ho tro: {activation}")


def edge_predict(model, X):
    """
    Forward pass cua MLP INT7 da giai luong tu hoa.

    Luu y: ham nay ap dung truc tiep len X. Neu mo hinh INT7 duoc
    huan luyen tren du lieu da scale, X phai duoc scale bang dung
    scaler cua pipeline huan luyen mo hinh truoc khi goi ham nay.
    """
    predictions = []

    for start in range(0, len(X), BATCH_SIZE):
        batch = X.iloc[start:start + BATCH_SIZE].to_numpy(
            dtype=np.float32,
            copy=True,
        )

        a = batch

        for i, layer in enumerate(model["layers"]):
            z = a @ layer["weight"] + layer["bias"]

            if i < len(model["layers"]) - 1:
                a = apply_activation(z, model["activation"])
            else:
                a = apply_activation(z, model["output_activation"])

        scores = np.asarray(a).reshape(-1)

        # Neu output la probability, nguong phan loai 0.5.
        predictions.append((scores >= 0.5).astype(np.int8))

    return np.concatenate(predictions)


# ============================================================
# 5. CREATE COST/RISK TARGETS
# ============================================================

def calculate_cost(y_true, y_pred):
    """
    FN = 10: bo sot tan cong.
    FP = 1 : bao dong gia.
    TP/TN = 0.

    Tra ve cost CHUA chuan hoa, thuoc {0, 1, 10}.
    """
    y_true = np.asarray(y_true, dtype=np.int8)
    y_pred = np.asarray(y_pred, dtype=np.int8)

    fn = (y_true == 1) & (y_pred == 0)
    fp = (y_true == 0) & (y_pred == 1)

    costs = np.zeros(len(y_true), dtype=np.float32)
    costs[fn] = C_FN
    costs[fp] = C_FP

    return costs


def summarize_detector(name, y_true, y_pred, costs):
    y_true = np.asarray(y_true, dtype=np.int8)
    y_pred = np.asarray(y_pred, dtype=np.int8)

    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())

    total = len(y_true)

    fpr = fp / max(fp + tn, 1)
    fnr = fn / max(fn + tp, 1)
    error_rate = (fp + fn) / max(total, 1)

    return {
        "detector": name,
        "n_samples": total,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "accuracy": (tn + tp) / max(total, 1),
        "FPR": fpr,
        "FNR": fnr,
        "error_rate": error_rate,
        "mean_cost": float(np.mean(costs)),
        "total_cost": float(np.sum(costs)),
    }


def generate_risk_targets(X, y, cloud_model, edge_model, split_name):
    print_section(f"GENERATE RISK TARGETS - {split_name}")

    y_np = y.to_numpy(dtype=np.int8)

    edge_pred = edge_predict(edge_model, X)
    cloud_pred = cloud_predict(cloud_model, X)

    edge_cost = calculate_cost(y_np, edge_pred)
    cloud_cost = calculate_cost(y_np, cloud_pred)

    edge_metrics = summarize_detector(
        "Edge_INT7", y_np, edge_pred, edge_cost
    )
    cloud_metrics = summarize_detector(
        "Cloud_FP32", y_np, cloud_pred, cloud_cost
    )

    print(f"Rows: {len(X):,}")
    print(f"C_FN={C_FN}, C_FP={C_FP}")
    print(
        f"Edge: error={edge_metrics['error_rate']:.6f}, "
        f"FNR={edge_metrics['FNR']:.6f}, "
        f"FPR={edge_metrics['FPR']:.6f}, "
        f"mean cost={edge_metrics['mean_cost']:.6f}"
    )
    print(
        f"Cloud: error={cloud_metrics['error_rate']:.6f}, "
        f"FNR={cloud_metrics['FNR']:.6f}, "
        f"FPR={cloud_metrics['FPR']:.6f}, "
        f"mean cost={cloud_metrics['mean_cost']:.6f}"
    )

    # Risk predictor hoc du doan chi phi ky vong cho tung action.
    # Chuan hoa ve [0,1] de thuan tien ket hop voi cac thanh phan
    # latency/bandwidth trong policy M4.
    y_edge_risk = edge_cost / C_MAX
    y_cloud_risk = cloud_cost / C_MAX

    audit = pd.DataFrame({
        "y_true": y_np,
        "edge_prediction": edge_pred,
        "cloud_prediction": cloud_pred,
        "edge_cost": edge_cost,
        "cloud_cost": cloud_cost,
        "edge_risk_target": y_edge_risk,
        "cloud_risk_target": y_cloud_risk,
    })

    metrics = [edge_metrics, cloud_metrics]

    return y_edge_risk, y_cloud_risk, audit, metrics


# ============================================================
# 6. TRAIN RISK PREDICTORS
# ============================================================

def train_risk_predictor(X_train, y_risk, name):
    print_section(f"TRAIN RISK PREDICTOR - {name}")

    X_fit, X_holdout, y_fit, y_holdout = train_test_split(
        X_train,
        y_risk,
        test_size=0.15,
        random_state=RANDOM_STATE,
    )

    model = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
    )

    model.fit(X_fit, y_fit)

    pred = model.predict(X_holdout)
    pred = np.clip(pred, 0.0, 1.0)

    metrics = {
        "model": name,
        "evaluation": "train_internal_holdout",
        "n_samples": len(X_holdout),
        "MAE": float(mean_absolute_error(y_holdout, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_holdout, pred))),
        "R2": float(r2_score(y_holdout, pred)),
        "true_mean_risk": float(np.mean(y_holdout)),
        "predicted_mean_risk": float(np.mean(pred)),
    }

    print(f"Internal holdout rows: {len(X_holdout):,}")
    print(f"MAE : {metrics['MAE']:.6f}")
    print(f"RMSE: {metrics['RMSE']:.6f}")
    print(f"R2  : {metrics['R2']:.6f}")
    print(f"True mean risk     : {metrics['true_mean_risk']:.6f}")
    print(f"Predicted mean risk: {metrics['predicted_mean_risk']:.6f}")

    return model, metrics


def evaluate_risk_predictor(model, X, y_risk, name, split_name):
    pred = np.clip(model.predict(X), 0.0, 1.0)

    metrics = {
        "model": name,
        "evaluation": split_name,
        "n_samples": len(X),
        "MAE": float(mean_absolute_error(y_risk, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_risk, pred))),
        "R2": float(r2_score(y_risk, pred)),
        "true_mean_risk": float(np.mean(y_risk)),
        "predicted_mean_risk": float(np.mean(pred)),
    }

    print(
        f"{name} / {split_name}: "
        f"MAE={metrics['MAE']:.6f}, "
        f"RMSE={metrics['RMSE']:.6f}, "
        f"R2={metrics['R2']:.6f}"
    )

    return metrics, pred


# ============================================================
# 7. MAIN
# ============================================================

def main():
    print_section("M3 RISK PREDICTOR TRAINING")
    print(f"C_FN: {C_FN}")
    print(f"C_FP: {C_FP}")
    print(f"C_MAX: {C_MAX}")
    print(f"Train CSV     : {TRAIN_PATH}")
    print(f"Validation CSV: {VALIDATION_PATH}")
    print(f"Test CSV      : {TEST_PATH}")
    print(f"Output dir    : {OUTPUT_DIR}")

    check_files()

    # Load the two fixed AD models.
    cloud_model = load_cloud_model()
    edge_model = load_edge_model()

    # Read train first to establish feature order.
    X_train, y_train, feature_columns = load_dataset(TRAIN_PATH)

    X_val, y_val, _ = load_dataset(
        VALIDATION_PATH,
        expected_features=feature_columns,
    )

    X_test, y_test, _ = load_dataset(
        TEST_PATH,
        expected_features=feature_columns,
    )

    # Check compatibility with the Cloud model.
    expected_n_features = getattr(
        cloud_model, "n_features_in_", None
    )

    if expected_n_features is not None:
        if expected_n_features != len(feature_columns):
            raise ValueError(
                f"Cloud model expects {expected_n_features} features, "
                f"but CSV provides {len(feature_columns)}. "
                "Check feature columns and preprocessing."
            )

    # Apply optional deterministic caps to each split independently.
    X_train, y_train = deterministic_sample(
        X_train, y_train, MAX_TRAIN_ROWS, RANDOM_STATE
    )
    X_val, y_val = deterministic_sample(
        X_val, y_val, MAX_VALIDATION_ROWS, RANDOM_STATE + 1
    )
    X_test, y_test = deterministic_sample(
        X_test, y_test, MAX_TEST_ROWS, RANDOM_STATE + 2
    )

    print_section("DATA SPLIT SUMMARY")
    print(f"Train     : {len(X_train):,}")
    print(f"Validation: {len(X_val):,}")
    print(f"Test      : {len(X_test):,}")
    print(f"Features  : {len(feature_columns)}")

    # Generate targets using the fixed detector models.
    train_edge_risk, train_cloud_risk, train_audit, train_ad_metrics = (
        generate_risk_targets(
            X_train, y_train, cloud_model, edge_model, "TRAIN"
        )
    )

    val_edge_risk, val_cloud_risk, val_audit, val_ad_metrics = (
        generate_risk_targets(
            X_val, y_val, cloud_model, edge_model, "VALIDATION"
        )
    )

    test_edge_risk, test_cloud_risk, test_audit, test_ad_metrics = (
        generate_risk_targets(
            X_test, y_test, cloud_model, edge_model, "TEST"
        )
    )

    # Save the generated targets for auditability.
    train_audit.to_csv(
        OUTPUT_DIR / "risk_targets_train.csv", index=False
    )
    val_audit.to_csv(
        OUTPUT_DIR / "risk_targets_validation.csv", index=False
    )
    test_audit.to_csv(
        OUTPUT_DIR / "risk_targets_test.csv", index=False
    )

    # Train only on TRAIN.
    edge_risk_model, edge_train_metrics = train_risk_predictor(
        X_train, train_edge_risk, "Edge_INT7"
    )

    cloud_risk_model, cloud_train_metrics = train_risk_predictor(
        X_train, train_cloud_risk, "Cloud_FP32"
    )

    # Validation and test are not used in model.fit().
    edge_val_metrics, edge_val_pred = evaluate_risk_predictor(
        edge_risk_model, X_val, val_edge_risk,
        "Edge_INT7", "validation"
    )

    cloud_val_metrics, cloud_val_pred = evaluate_risk_predictor(
        cloud_risk_model, X_val, val_cloud_risk,
        "Cloud_FP32", "validation"
    )

    edge_test_metrics, edge_test_pred = evaluate_risk_predictor(
        edge_risk_model, X_test, test_edge_risk,
        "Edge_INT7", "test"
    )

    cloud_test_metrics, cloud_test_pred = evaluate_risk_predictor(
        cloud_risk_model, X_test, test_cloud_risk,
        "Cloud_FP32", "test"
    )

    # Save risk predictors.
    edge_model_out = OUTPUT_DIR / "risk_predictor_edge.joblib"
    cloud_model_out = OUTPUT_DIR / "risk_predictor_cloud.joblib"

    joblib.dump(edge_risk_model, edge_model_out)
    joblib.dump(cloud_risk_model, cloud_model_out)

    # Save prediction audit on validation/test.
    val_pred_audit = val_audit.copy()
    val_pred_audit["edge_risk_predicted"] = edge_val_pred
    val_pred_audit["cloud_risk_predicted"] = cloud_val_pred
    val_pred_audit.to_csv(
        OUTPUT_DIR / "risk_predictor_validation_audit.csv",
        index=False,
    )

    test_pred_audit = test_audit.copy()
    test_pred_audit["edge_risk_predicted"] = edge_test_pred
    test_pred_audit["cloud_risk_predicted"] = cloud_test_pred
    test_pred_audit.to_csv(
        OUTPUT_DIR / "risk_predictor_test_audit.csv",
        index=False,
    )

    # Combine detector metrics and risk-predictor metrics.
    report = {
        "configuration": {
            "C_FN": C_FN,
            "C_FP": C_FP,
            "C_MAX": C_MAX,
            "risk_target_normalization": "cost / C_MAX",
            "train_rows_used": len(X_train),
            "validation_rows_used": len(X_val),
            "test_rows_used": len(X_test),
            "n_features": len(feature_columns),
            "feature_columns": feature_columns,
            "edge_model_path": str(EDGE_MODEL_PATH),
            "cloud_model_path": str(CLOUD_MODEL_PATH),
        },
        "detector_metrics": {
            "train": train_ad_metrics,
            "validation": val_ad_metrics,
            "test": test_ad_metrics,
        },
        "risk_predictor_metrics": {
            "edge_train_internal_holdout": edge_train_metrics,
            "cloud_train_internal_holdout": cloud_train_metrics,
            "edge_validation": edge_val_metrics,
            "cloud_validation": cloud_val_metrics,
            "edge_test": edge_test_metrics,
            "cloud_test": cloud_test_metrics,
        },
        "saved_models": {
            "edge_risk_predictor": str(edge_model_out),
            "cloud_risk_predictor": str(cloud_model_out),
        },
    }

    report_path = OUTPUT_DIR / "risk_predictor_metadata.json"

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Summary CSV for easy comparison.
    summary_rows = [
        *[
            {"group": "detector_train", **m}
            for m in train_ad_metrics
        ],
        *[
            {"group": "detector_validation", **m}
            for m in val_ad_metrics
        ],
        *[
            {"group": "detector_test", **m}
            for m in test_ad_metrics
        ],
        {"group": "risk_predictor", **edge_train_metrics},
        {"group": "risk_predictor", **cloud_train_metrics},
        {"group": "risk_predictor", **edge_val_metrics},
        {"group": "risk_predictor", **cloud_val_metrics},
        {"group": "risk_predictor", **edge_test_metrics},
        {"group": "risk_predictor", **cloud_test_metrics},
    ]

    pd.DataFrame(summary_rows).to_csv(
        OUTPUT_DIR / "risk_predictor_summary.csv",
        index=False,
    )

    print_section("FINISHED")
    print(f"Edge risk predictor : {edge_model_out}")
    print(f"Cloud risk predictor: {cloud_model_out}")
    print(f"Metadata            : {report_path}")
    print(f"Summary CSV         : {OUTPUT_DIR / 'risk_predictor_summary.csv'}")
    print(f"Validation audit    : {OUTPUT_DIR / 'risk_predictor_validation_audit.csv'}")
    print(f"Test audit          : {OUTPUT_DIR / 'risk_predictor_test_audit.csv'}")


if __name__ == "__main__":
    main()
