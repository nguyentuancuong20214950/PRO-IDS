
"""
AD1v2.py
============================================================
AD1v2 - Lightweight MLP Anomaly Detector for Edge

Dataset:
    CICIDS2017v2

Input:
    77 standardized numerical features

Labels:
    0 = Benign
    1 = Attack

Architecture:
    77 -> 16 -> 8 -> 1

Training:
    - Use 100% of the prepared training dataset
    - Adam optimizer
    - ReLU activation
    - Early stopping
    - Evaluate on separate validation and test sets

Important:
    The datasets have already been standardized by Preprocessing2.py.
    Do NOT apply StandardScaler again in this script.

Output directory:
    D:/PythonProject/2017v2/
============================================================
"""

import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.neural_network import MLPClassifier


# ============================================================
# 1. PATH CONFIGURATION
# ============================================================

PROJECT_DIR = Path(r"D:\PythonProject\2017v2")
DATA_DIR = Path(r"D:\CICIDS2017v2")

TRAIN_FILE = DATA_DIR / "CICIDS2017v2_train.csv"
VAL_FILE = DATA_DIR / "CICIDS2017v2_validation.csv"
TEST_FILE = DATA_DIR / "CICIDS2017v2_test.csv"

# The scaler was fitted on the training data in Preprocessing2.py.
SCALER_FILE = DATA_DIR / "CICIDS2017v2_scaler.joblib"

# Use new filenames to avoid overwriting the original AD1.
MODEL_FILE = PROJECT_DIR / "AD1v2_model.joblib"
METADATA_FILE = PROJECT_DIR / "AD1v2_metadata.json"
METRICS_FILE = PROJECT_DIR / "AD1v2_metrics.csv"
REPORT_FILE = PROJECT_DIR / "AD1v2_classification_report.json"


# ============================================================
# 2. MODEL CONFIGURATION
# ============================================================

RANDOM_STATE = 42

# Use all rows in the prepared training dataset.
TRAIN_RATIO = 1.0

# Architecture: 77 -> 16 -> 8 -> 1
HIDDEN_LAYERS = (32, 16, 8)

LEARNING_RATE_INIT = 0.001
BATCH_SIZE = 512
MAX_ITER = 100

EARLY_STOPPING = True
VALIDATION_FRACTION = 0.10
N_ITER_NO_CHANGE = 15


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def get_class_distribution(y):
    """Return sample counts and class proportions."""

    total = len(y)
    benign = int(np.sum(y == 0))
    attack = int(np.sum(y == 1))

    return {
        "total": int(total),
        "benign": benign,
        "attack": attack,
        "benign_ratio": float(benign / total) if total else 0.0,
        "attack_ratio": float(attack / total) if total else 0.0,
    }


def print_class_distribution(name, y):
    """Print Benign/Attack class distribution."""

    dist = get_class_distribution(y)

    print(f"\n{name}")
    print("-" * 60)
    print(f"Total samples : {dist['total']:,}")
    print(
        f"Benign (0)    : {dist['benign']:,} "
        f"({dist['benign_ratio']:.4%})"
    )
    print(
        f"Attack (1)    : {dist['attack']:,} "
        f"({dist['attack_ratio']:.4%})"
    )


def load_dataset(file_path, feature_columns=None):
    """
    Load an already-standardized dataset.

    Do not fit or apply StandardScaler here.
    """

    print(f"Loading: {file_path.name}")

    df = pd.read_csv(file_path, low_memory=False)

    if "Label" not in df.columns:
        raise ValueError(
            f"Cannot find the Label column in {file_path}"
        )

    # Use the feature order established by the training dataset.
    if feature_columns is None:
        feature_columns = [
            column for column in df.columns
            if column != "Label"
        ]
    else:
        missing = [
            column for column in feature_columns
            if column not in df.columns
        ]

        unexpected = [
            column for column in df.columns
            if column != "Label" and column not in feature_columns
        ]

        if missing or unexpected:
            raise ValueError(
                f"Feature mismatch in {file_path.name}\n"
                f"Missing: {missing}\n"
                f"Unexpected: {unexpected}"
            )

    X = df[feature_columns].to_numpy(dtype=np.float32)
    y = df["Label"].to_numpy(dtype=np.int8)

    if not np.isfinite(X).all():
        raise ValueError(
            f"{file_path.name} contains NaN or Inf."
        )

    if not set(np.unique(y).tolist()).issubset({0, 1}):
        raise ValueError(
            f"{file_path.name} contains labels outside {{0, 1}}."
        )

    if len(X) == 0:
        raise ValueError(f"{file_path.name} is empty.")

    return X, y, feature_columns


def evaluate_model(model, X, y, dataset_name):
    """Evaluate binary classification performance."""

    predictions = model.predict(X)

    # Confusion matrix:
    #                 Predicted
    #                 0       1
    # Actual 0       TN      FP
    # Actual 1       FN      TP

    tn, fp, fn, tp = confusion_matrix(
        y,
        predictions,
        labels=[0, 1],
    ).ravel()

    accuracy = accuracy_score(y, predictions)

    attack_precision = precision_score(
        y, predictions, pos_label=1, zero_division=0
    )

    attack_recall = recall_score(
        y, predictions, pos_label=1, zero_division=0
    )

    attack_f1 = f1_score(
        y, predictions, pos_label=1, zero_division=0
    )

    benign_f1 = f1_score(
        y, predictions, pos_label=0, zero_division=0
    )

    macro_f1 = f1_score(
        y, predictions, average="macro", zero_division=0
    )

    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    metrics = {
        "dataset": dataset_name,
        "sample_count": int(len(y)),
        "accuracy": float(accuracy),
        "attack_precision": float(attack_precision),
        "attack_recall": float(attack_recall),
        "attack_f1": float(attack_f1),
        "benign_f1": float(benign_f1),
        "macro_f1": float(macro_f1),
        "fnr": float(fnr),
        "fpr": float(fpr),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    print("\n" + "=" * 70)
    print(dataset_name)
    print("=" * 70)

    print(f"Accuracy         : {accuracy:.6f}")
    print(f"Attack Precision : {attack_precision:.6f}")
    print(f"Attack Recall    : {attack_recall:.6f}")
    print(f"Attack F1        : {attack_f1:.6f}")
    print(f"Benign F1        : {benign_f1:.6f}")
    print(f"Macro-F1         : {macro_f1:.6f}")
    print(f"FNR              : {fnr:.6f}")
    print(f"FPR              : {fpr:.6f}")

    print("\nConfusion Matrix:")
    print(f"TN = {tn:,}")
    print(f"FP = {fp:,}")
    print(f"FN = {fn:,}")
    print(f"TP = {tp:,}")

    print("\nClassification Report:")
    print(
        classification_report(
            y,
            predictions,
            labels=[0, 1],
            target_names=["Benign", "Attack"],
            digits=6,
            zero_division=0,
        )
    )

    return metrics, predictions


# ============================================================
# 4. MAIN TRAINING PIPELINE
# ============================================================

def main():

    PROJECT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("AD1v2 - LIGHTWEIGHT EDGE ANOMALY DETECTOR")
    print("=" * 70)

    print(f"Project directory : {PROJECT_DIR}")
    print(f"Dataset directory : {DATA_DIR}")
    print(f"Scaler file       : {SCALER_FILE}")
    print(f"Training ratio    : {TRAIN_RATIO:.0%}")
    print(f"Hidden layers     : {HIDDEN_LAYERS}")
    print(f"Random state      : {RANDOM_STATE}")

    # --------------------------------------------------------
    # Step 1: Check files
    # --------------------------------------------------------

    print("\n[1/7] Checking input files...")

    for file_path in [TRAIN_FILE, VAL_FILE, TEST_FILE]:
        if not file_path.exists():
            raise FileNotFoundError(
                f"Dataset file not found:\n{file_path}"
            )

    if not SCALER_FILE.exists():
        raise FileNotFoundError(
            f"Saved StandardScaler not found:\n{SCALER_FILE}"
        )

    # --------------------------------------------------------
    # Step 2: Load datasets
    # --------------------------------------------------------

    print("\n[2/7] Loading datasets...")

    X_train, y_train, feature_columns = load_dataset(
        TRAIN_FILE
    )

    X_val, y_val, _ = load_dataset(
        VAL_FILE,
        feature_columns,
    )

    X_test, y_test, _ = load_dataset(
        TEST_FILE,
        feature_columns,
    )

    if len(feature_columns) != 77:
        raise ValueError(
            f"Expected 77 input features, got {len(feature_columns)}."
        )

    # Verify the saved scaler matches the feature schema.
    scaler = joblib.load(SCALER_FILE)

    if not hasattr(scaler, "n_features_in_"):
        raise ValueError("The saved scaler is not fitted correctly.")

    if scaler.n_features_in_ != len(feature_columns):
        raise ValueError(
            f"Scaler expects {scaler.n_features_in_} features, "
            f"but the dataset contains {len(feature_columns)}."
        )

    if hasattr(scaler, "feature_names_in_"):
        scaler_features = list(scaler.feature_names_in_)

        if scaler_features != feature_columns:
            raise ValueError(
                "Feature order does not match the saved scaler. "
                "Check Preprocessing2.py outputs."
            )

    print(f"\nTraining   : {X_train.shape}")
    print(f"Validation : {X_val.shape}")
    print(f"Test       : {X_test.shape}")

    print_class_distribution("TRAIN", y_train)
    print_class_distribution("VALIDATION", y_val)
    print_class_distribution("TEST", y_test)

    # --------------------------------------------------------
    # Step 3: Prepare training data
    # --------------------------------------------------------

    print("\n[3/7] Preparing training data...")

    # Use all rows from CICIDS2017v2_train.csv.
    selected_train_samples = len(y_train)

    print(f"Training samples: {selected_train_samples:,}")
    print(f"Training ratio  : {TRAIN_RATIO:.0%}")

    print(
        "Input features are already standardized. "
        "No additional scaling will be performed."
    )

    # --------------------------------------------------------
    # Step 4: Build and train AD1v2
    # --------------------------------------------------------

    print("\n[4/7] Building and training AD1v2...")

    model = MLPClassifier(
        hidden_layer_sizes=HIDDEN_LAYERS,
        activation="relu",
        solver="adam",
        learning_rate_init=LEARNING_RATE_INIT,
        batch_size=BATCH_SIZE,
        max_iter=MAX_ITER,
        early_stopping=EARLY_STOPPING,
        validation_fraction=VALIDATION_FRACTION,
        n_iter_no_change=N_ITER_NO_CHANGE,
        random_state=RANDOM_STATE,
        verbose=True,
    )

    print("\nArchitecture:")
    print(f"Input layer   : {len(feature_columns)} neurons")
    print(f"Hidden layer 1: {HIDDEN_LAYERS[0]} neurons")
    print(f"Hidden layer 2: {HIDDEN_LAYERS[1]} neurons")
    print("Output layer  : 1 binary classification output")

    print("\nStarting training...")

    start_time = time.perf_counter()

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Stochastic Optimizer: Maximum iterations",
        )

        model.fit(X_train, y_train)

    training_time = time.perf_counter() - start_time

    print("\nTraining finished.")
    print(f"Training time : {training_time:.2f} seconds")
    print(f"Iterations    : {model.n_iter_}")
    print(f"Final loss    : {model.loss_:.8f}")

    # --------------------------------------------------------
    # Step 5: Evaluate validation and test
    # --------------------------------------------------------

    print("\n[5/7] Evaluating AD1v2...")

    validation_metrics, _ = evaluate_model(
        model,
        X_val,
        y_val,
        "AD1v2 - VALIDATION RESULTS",
    )

    test_metrics, test_predictions = evaluate_model(
        model,
        X_test,
        y_test,
        "AD1v2 - TEST RESULTS",
    )

    # --------------------------------------------------------
    # Step 6: Save model and evaluation results
    # --------------------------------------------------------

    print("\n[6/7] Saving model and evaluation results...")

    joblib.dump(model, MODEL_FILE, compress=3)

    model_size_bytes = MODEL_FILE.stat().st_size
    model_size_mb = model_size_bytes / (1024 * 1024)

    # Save validation and test metrics.
    metrics_df = pd.DataFrame([
        {
            "dataset": "validation",
            **{
                key: value
                for key, value in validation_metrics.items()
                if key != "dataset"
            },
        },
        {
            "dataset": "test",
            **{
                key: value
                for key, value in test_metrics.items()
                if key != "dataset"
            },
        },
    ])

    metrics_df.to_csv(
        METRICS_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    # Save detailed test classification report.
    report = classification_report(
        y_test,
        test_predictions,
        labels=[0, 1],
        target_names=["Benign", "Attack"],
        output_dict=True,
        zero_division=0,
    )

    with open(REPORT_FILE, "w", encoding="utf-8") as file:
        json.dump(
            report,
            file,
            indent=4,
            ensure_ascii=False,
        )

    # --------------------------------------------------------
    # Step 7: Save metadata
    # --------------------------------------------------------

    metadata = {
        "model_name": "AD1v2",
        "deployment_target": "Edge",
        "dataset": {
            "name": "CICIDS2017v2",
            "train_file": str(TRAIN_FILE),
            "validation_file": str(VAL_FILE),
            "test_file": str(TEST_FILE),
            "scaler_file": str(SCALER_FILE),
            "feature_count": len(feature_columns),
            "feature_columns": feature_columns,
            "labels": {
                "0": "Benign",
                "1": "Attack",
            },
            "scaling": (
                "StandardScaler fitted on training data "
                "in Preprocessing2.py; applied before training"
            ),
        },
        "architecture": {
            "input_features": len(feature_columns),
            "hidden_layers": list(HIDDEN_LAYERS),
            "activation": "relu",
            "output": "binary classification",
        },
        "configuration": {
            "random_state": RANDOM_STATE,
            "train_ratio": TRAIN_RATIO,
            "learning_rate_init": LEARNING_RATE_INIT,
            "batch_size": BATCH_SIZE,
            "max_iter": MAX_ITER,
            "early_stopping": EARLY_STOPPING,
            "validation_fraction_for_early_stopping": (
                VALIDATION_FRACTION
            ),
            "n_iter_no_change": N_ITER_NO_CHANGE,
        },
        "training": {
            "original_train_samples": int(len(y_train)),
            "selected_train_samples": int(selected_train_samples),
            "training_ratio": float(TRAIN_RATIO),
            "iterations": int(model.n_iter_),
            "final_loss": float(model.loss_),
            "training_time_seconds": float(training_time),
            "class_distribution": get_class_distribution(y_train),
        },
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "model": {
            "file": str(MODEL_FILE),
            "size_bytes": int(model_size_bytes),
            "size_mb": float(model_size_mb),
        },
        "inference_latency": None,
        "software": {
            "python": sys.version,
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }

    with open(METADATA_FILE, "w", encoding="utf-8") as file:
        json.dump(
            metadata,
            file,
            indent=4,
            ensure_ascii=False,
        )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n[7/7] AD1v2 completed.")

    print("\n" + "=" * 70)
    print("AD1v2 FINAL TEST RESULTS")
    print("=" * 70)

    print("Architecture     : 77 -> 16 -> 8 -> 1")
    print(f"Training samples : {selected_train_samples:,}")
    print(f"Training time    : {training_time:.2f} seconds")
    print(f"Iterations       : {model.n_iter_}")
    print(f"Final loss       : {model.loss_:.8f}")
    print(f"Model size       : {model_size_mb:.6f} MB")

    print(f"\nAccuracy         : {test_metrics['accuracy']:.6f}")
    print(f"Attack Precision : {test_metrics['attack_precision']:.6f}")
    print(f"Attack Recall    : {test_metrics['attack_recall']:.6f}")
    print(f"Attack F1        : {test_metrics['attack_f1']:.6f}")
    print(f"Benign F1        : {test_metrics['benign_f1']:.6f}")
    print(f"Macro-F1         : {test_metrics['macro_f1']:.6f}")
    print(f"FNR              : {test_metrics['fnr']:.6f}")
    print(f"FPR              : {test_metrics['fpr']:.6f}")

    print("\nSaved files:")
    print(f"Model       : {MODEL_FILE}")
    print(f"Metadata    : {METADATA_FILE}")
    print(f"Metrics CSV : {METRICS_FILE}")
    print(f"Report JSON : {REPORT_FILE}")
    print("=" * 70)


if __name__ == "__main__":
    main()
