"""
train_model.py
Trains two RandomForestRegressor models:
  - model_turbidity.pkl  -> predicts Turbidity (NTU)
  - model_no3.pkl        -> predicts NO3 (mg/L)
Run once before launching app.py:
    python train_model.py
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

CSV_PATH = "Johnstone_river_coquette_point_joined.csv"
FEATURES  = ["Conductivity", "Temp", "Dayofweek", "Month"]

# 1. Load
print("Loading dataset ...")
df = pd.read_csv(CSV_PATH)
print(f"  Raw shape : {df.shape}")

# 2. Parse timestamp & derive hour
df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
df["Hour"]      = df["Timestamp"].dt.hour
ALL_FEATURES    = FEATURES + ["Hour"]

# 3. Mean-impute numeric columns
for col in df.select_dtypes(include=[np.number]).columns:
    df[col] = df[col].fillna(df[col].mean())

print(f"  After imputation shape: {df.shape}")

# 4. Helper: train + evaluate one model
def train_and_save(target: str, filename: str) -> None:
    sub = df[ALL_FEATURES + [target]].dropna()
    X   = sub[ALL_FEATURES]
    y   = sub[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print(f"\n  [{target}]")
    print(f"    MAE : {mean_absolute_error(y_test, preds):.4f}")
    print(f"    R2  : {r2_score(y_test, preds):.4f}")

    joblib.dump(model, filename)
    print(f"    Saved -> {filename}")

# 5. Train both models
train_and_save("Turbidity", "model_turbidity.pkl")
train_and_save("NO3",       "model_no3.pkl")

print("\nDone. Both models saved. You can now run:  streamlit run app.py")
