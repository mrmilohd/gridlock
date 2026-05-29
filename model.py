import pandas as pd
import numpy as np
import pygeohash as pgh
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
import warnings

warnings.filterwarnings("ignore")

print("Loading data...")
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

train["is_train"] = 1
test["is_train"] = 0
test["demand"] = np.nan

# We do NOT extract test_idx here anymore, we let 'Index' travel with the dataframe
df = pd.concat([train, test], axis=0, ignore_index=True)

print("Starting Advanced Feature Engineering (V4 Architecture)...")

# ---------------------------------------------------------
# A. Base Parsing & Spatial Features
# ---------------------------------------------------------
df["day"] = df["day"].astype(int)
df["latitude"] = df["geohash"].apply(
    lambda x: pgh.decode(x)[0] if pd.notnull(x) else np.nan
)
df["longitude"] = df["geohash"].apply(
    lambda x: pgh.decode(x)[1] if pd.notnull(x) else np.nan
)

# geohash_prefix acts as a proxy for the broader "Neighborhood"
df["geohash_prefix"] = df["geohash"].astype(str).str.slice(0, 5)

time_split = df["timestamp"].astype(str).str.split(":", expand=True)
df["hour"] = time_split[0].astype(int)
df["minute"] = time_split[1].astype(int)

df["minute_bin"] = (df["minute"] // 15) * 15
df["time_slot"] = (
    df["hour"].astype(str).str.zfill(2)
    + ":"
    + df["minute_bin"].astype(str).str.zfill(2)
)

# Absolute continuous time for chronological sorting
df["time_decimal"] = df["day"] * 24 + df["hour"] + (df["minute"] / 60.0)

# Cyclical Time
df["time_sin"] = np.sin(2 * np.pi * df["time_decimal"] / 24.0)
df["time_cos"] = np.cos(2 * np.pi * df["time_decimal"] / 24.0)

# ---------------------------------------------------------
# B. The Multi-Tier Interaction Unlock
# ---------------------------------------------------------
df["geohash_x_time"] = df["geohash"].astype(str) + "_" + df["time_slot"].astype(str)
df["weather_road"] = df["Weather"].astype(str) + "_" + df["RoadType"].astype(str)

# High-Order Interaction
df["neighborhood_weather_hour"] = (
    df["geohash_prefix"].astype(str)
    + "_"
    + df["Weather"].astype(str)
    + "_"
    + df["hour"].astype(str)
)

# ---------------------------------------------------------
# C. Strict Time-Series Dynamics (Momentum & EWMA)
# ---------------------------------------------------------
print(" -> Calculating Temporal Momentum (EWMA)...")
df = df.sort_values(["geohash", "time_decimal"]).reset_index(drop=True)
df["ewma_demand_3hr"] = df.groupby("geohash")["demand"].transform(
    lambda x: x.shift(1).ewm(alpha=0.3, min_periods=1).mean()
)

df = df.sort_values(["geohash_prefix", "time_decimal"]).reset_index(drop=True)
df["neighborhood_momentum"] = df.groupby("geohash_prefix")["demand"].transform(
    lambda x: x.shift(1).rolling(window=4, min_periods=1).mean()
)

# ---------------------------------------------------------
# D. Expanding Window Target Encoding (Leakage-Free)
# ---------------------------------------------------------
print(" -> Applying Expanding Window Target Encoding...")
df = df.sort_values("time_decimal").reset_index(drop=True)

target_encode_cols = ["geohash_x_time", "neighborhood_weather_hour"]

for col in target_encode_cols:
    df[col] = df.groupby(col)["demand"].transform(
        lambda x: x.shift(1).expanding().mean()
    )

global_median = df[df["is_train"] == 1]["demand"].median()
df["ewma_demand_3hr"] = df["ewma_demand_3hr"].fillna(global_median)
df["neighborhood_momentum"] = df["neighborhood_momentum"].fillna(global_median)
for col in target_encode_cols:
    df[col] = df[col].fillna(global_median)

# ---------------------------------------------------------
# E. Encoding & Cleanup
# ---------------------------------------------------------
categorical_cols = [
    "RoadType",
    "Weather",
    "geohash",
    "geohash_prefix",
    "time_slot",
    "weather_road",
    "neighborhood_weather_hour",
    "geohash_x_time",
]

for col in categorical_cols:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

binary_cols = ["LargeVehicles", "Landmarks"]
for col in binary_cols:
    df[col] = df[col].map({"Yes": 1, "No": 0}).fillna(0).astype(int)

# CRITICAL FIX: Do NOT drop 'Index'
df = df.drop(["timestamp", "minute_bin", "latitude", "longitude"], axis=1)

# ---------------------------------------------------------
# 3. Validation Split (Strictly Chronological)
# ---------------------------------------------------------
print(" -> Splitting Data...")
train_df = df[df["is_train"] == 1].sort_values("time_decimal").reset_index(drop=True)
test_df = df[df["is_train"] == 0].reset_index(drop=True)

split_idx = int(len(train_df) * 0.8)

# Drop Index strictly from the feature matrices, keep it in the main test_df
X_train = train_df.iloc[:split_idx].drop(
    ["demand", "is_train", "time_decimal", "Index"], axis=1
)
y_train = train_df.iloc[:split_idx]["demand"]

X_val = train_df.iloc[split_idx:].drop(
    ["demand", "is_train", "time_decimal", "Index"], axis=1
)
y_val = train_df.iloc[split_idx:]["demand"]

test_processed = test_df.drop(["demand", "is_train", "time_decimal", "Index"], axis=1)

# ---------------------------------------------------------
# 4. Model Training
# ---------------------------------------------------------
print("Training Model...")

lgb_params = {
    "n_estimators": 3000,
    "learning_rate": 0.015,
    "max_depth": 12,
    "num_leaves": 127,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.5,
    "random_state": 42,
    "n_jobs": -1,
    "metric": "rmse",
}

model = lgb.LGBMRegressor(**lgb_params)

model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(stopping_rounds=200, verbose=True)],
)

# ---------------------------------------------------------
# 5. Evaluation & Submission (Index Fixed)
# ---------------------------------------------------------
val_preds = model.predict(X_val)
val_preds = np.clip(val_preds, 0, 1)

r2 = r2_score(y_val, val_preds)
print(f"\n--- Results ---")
print(f"STRICT CHRONOLOGICAL Validation R2 Score: {r2:.6f}")
print(f"Expected Hackathon Score: {max(0, 100 * r2):.4f}")

test_preds = model.predict(test_processed)
test_preds = np.clip(test_preds, 0, 1)

# CRITICAL FIX: Map predictions exactly to the traveling Index column
submission = pd.DataFrame({"Index": test_df["Index"], "demand": test_preds})

# Sort numerically back to original hackathon format before saving
submission = submission.sort_values("Index").reset_index(drop=True)

submission.to_csv("submission_v4_fixed.csv", index=False)
print("Saved to submission_v4_fixed.csv successfully!")
