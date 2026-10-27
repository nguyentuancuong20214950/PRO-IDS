import joblib
import numpy as np
from pathlib import Path

base = Path(r"D:\Documents\Course's Project\PRO-IDS")

for name in ["risk_predictor_edge.joblib", "risk_predictor_cloud.joblib"]:
    print("=" * 60)
    print(f"File: {name}")
    model = joblib.load(base / name)
    
    # 1 sample, 77 features, random
    X = np.random.randn(1, 77).astype(np.float32)
    pred = model.predict(X)
    print(f"Input shape: {X.shape}")
    print(f"Output type: {type(pred)}")
    print(f"Output shape: {np.asarray(pred).shape}")
    print(f"Output value: {pred}")
    print(f"Output dtype: {np.asarray(pred).dtype}")
    
    # Test 5 samples
    X5 = np.random.randn(5, 77).astype(np.float32)
    pred5 = model.predict(X5)
    print(f"5-sample output: {pred5}")
    print(f"Range: [{pred5.min():.4f}, {pred5.max():.4f}]")
