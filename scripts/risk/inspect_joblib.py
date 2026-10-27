import joblib
from pathlib import Path

base = Path(r"D:\Documents\Course's Project\PRO-IDS")
for name in ["risk_predictor_edge.joblib", "risk_predictor_cloud.joblib"]:
    path = base / name
    print("=" * 60)
    print(f"File: {name}")
    print(f"Size: {path.stat().st_size / 1024 / 1024:.2f} MB")
    model = joblib.load(path)
    print(f"Type: {type(model).__name__}")
    print(f"Module: {type(model).__module__}")
    if hasattr(model, "get_params"):
        params = model.get_params()
        for k, v in params.items():
            print(f"  {k}: {v}")
    if hasattr(model, "n_features_in_"):
        print(f"n_features_in_: {model.n_features_in_}")
    if hasattr(model, "feature_importances_"):
        fi = model.feature_importances_
        print(f"feature_importances_ shape: {fi.shape}")
        print(f"  top 5 importance: {sorted(fi, reverse=True)[:5]}")
        print(f"  min: {fi.min():.6f}, max: {fi.max():.6f}")
    if hasattr(model, "n_outputs_"):
        print(f"n_outputs_: {model.n_outputs_}")
