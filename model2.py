import pandas as pd
import numpy as np
import pygeohash as pgh
import lightgbm as lgb
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
from sklearn.cluster import KMeans
import warnings

warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# 1. Initialization & Data Loading
# ---------------------------------------------------------
print("Loading data...")
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

train["is_train"] = 1
test["is_train"] = 0
test["demand"] = np.nan

df = pd.concat([train, test], axis=0, ignore_index=True)

# ---------------------------------------------------------
# 2. Base Projections & Parsing
# ---------------------------------------------------------
print("Starting Advanced Feature Engineering...")
df["day"] = df["day"].astype(int)
df["latitude"] = df["geohash"].apply(
    lambda x: pgh.decode(x)[0] if pd.notnull(x) else np.nan
)
df["longitude"] = df["geohash"].apply(
    lambda x: pgh.decode(x)[1] if pd.notnull(x) else np.nan
)
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
df["time_decimal"] = df["day"] * 24 + df["hour"] + (df["minute"] / 60.0)

df["time_sin"] = np.sin(2 * np.pi * df["time_decimal"] / 24.0)
df["time_cos"] = np.cos(2 * np.pi * df["time_decimal"] / 24.0)

# ---------------------------------------------------------
# 3. Volatility & Cluster Fallbacks
# ---------------------------------------------------------
print(" -> Building Clusters and Volatility Profiles...")
coords = df[["latitude", "longitude"]].fillna(df[["latitude", "longitude"]].mean())
kmeans = KMeans(n_clusters=60, random_state=42, n_init=10)
df["spatial_cluster"] = kmeans.fit_predict(coords)

day48 = df[(df["day"] == 48) & (df["is_train"] == 1)]

geo_stats = day48.groupby("geohash")["demand"].agg(["mean", "std", "max"]).reset_index()
geo_stats.columns = ["geohash", "geo_mean", "geo_std", "geo_max"]
df = pd.merge(df, geo_stats, on="geohash", how="left")

cluster_means = day48.groupby("spatial_cluster")["demand"].mean()
df["geo_mean"] = (
    df["geo_mean"]
    .fillna(df["spatial_cluster"].map(cluster_means))
    .fillna(day48["demand"].median())
)
df["geo_std"] = df["geo_std"].fillna(0)
df["geo_max"] = df["geo_max"].fillna(df["geo_mean"])

# ---------------------------------------------------------
# 4. Sequential Momentum (EWMA)
# ---------------------------------------------------------
print(" -> Calculating Temporal Momentum...")
df = df.sort_values(["geohash", "time_decimal"]).reset_index(drop=True)
df["ewma_demand_3hr"] = df.groupby("geohash")["demand"].transform(
    lambda x: x.shift(1).ewm(alpha=0.3, min_periods=1).mean()
)

df = df.sort_values(["geohash_prefix", "time_decimal"]).reset_index(drop=True)
df["neighborhood_momentum"] = df.groupby("geohash_prefix")["demand"].transform(
    lambda x: x.shift(1).rolling(window=4, min_periods=1).mean()
)

# ---------------------------------------------------------
# 5. Expanding Window Target Encoding
# ---------------------------------------------------------
print(" -> Applying Expanding Target Encodings...")
df = df.sort_values("time_decimal").reset_index(drop=True)
df["geohash_x_time"] = df["geohash"].astype(str) + "_" + df["time_slot"].astype(str)
df["neighborhood_weather_hour"] = (
    df["geohash_prefix"].astype(str)
    + "_"
    + df["Weather"].astype(str)
    + "_"
    + df["hour"].astype(str)
)

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
# 6. Categorical Transformations & Cleanup
# ---------------------------------------------------------
print(" -> Encoding Categoricals...")
df["weather_road"] = df["Weather"].astype(str) + "_" + df["RoadType"].astype(str)
categorical_cols = [
    "RoadType",
    "Weather",
    "geohash",
    "geohash_prefix",
    "time_slot",
    "weather_road",
]
for col in categorical_cols:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

binary_cols = ["LargeVehicles", "Landmarks"]
for col in binary_cols:
    df[col] = df[col].map({"Yes": 1, "No": 0}).fillna(0).astype(int)

# CRITICAL FIX MAINTAINED: Do NOT drop 'Index'
df = df.drop(["timestamp", "minute_bin", "latitude", "longitude"], axis=1)

# ---------------------------------------------------------
# 7. Model Execution Pipeline (DEEP ENSEMBLE MODE)
# ---------------------------------------------------------
print(" -> Splitting Data Chronologically...")
train_df = df[df["is_train"] == 1].sort_values("time_decimal").reset_index(drop=True)
test_df = df[df["is_train"] == 0].reset_index(drop=True)

split_idx = int(len(train_df) * 0.8)

# Drop Index strictly from the feature matrices
X_train = train_df.iloc[:split_idx].drop(
    ["demand", "is_train", "time_decimal", "Index"], axis=1
)
y_train = train_df.iloc[:split_idx]["demand"]

X_val = train_df.iloc[split_idx:].drop(
    ["demand", "is_train", "time_decimal", "Index"], axis=1
)
y_val = train_df.iloc[split_idx:]["demand"]

test_processed = test_df.drop(["demand", "is_train", "time_decimal", "Index"], axis=1)

# --- MODEL 1: LightGBM (Deep Convergence) ---
print("\n[1/2] Training LightGBM (Deep Run)...")
lgb_params = {
    "n_estimators": 10000,  # Massively increased
    "learning_rate": 0.005,  # Lowered for micro-precision
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

model_lgb = lgb.LGBMRegressor(**lgb_params)
model_lgb.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[
        lgb.early_stopping(stopping_rounds=500, verbose=True)
    ],  # Much longer patience
)
lgb_val_preds = np.clip(model_lgb.predict(X_val), 0, 1)
lgb_test_preds = np.clip(model_lgb.predict(test_processed), 0, 1)

# --- MODEL 2: XGBoost (Deep Convergence) ---
print("\n[2/2] Training XGBoost (Deep Run)...")
xgb_params = {
    "n_estimators": 6000,  # Doubled
    "learning_rate": 0.008,  # Halved for stability
    "max_depth": 8,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_jobs": -1,
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "tree_method": "hist",
    "early_stopping_rounds": 400,  # Added explicit XGBoost patience
}

model_xgb = xgb.XGBRegressor(**xgb_params)
model_xgb.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    verbose=100,  # Print progress every 100 trees
)
xgb_val_preds = np.clip(model_xgb.predict(X_val), 0, 1)
xgb_test_preds = np.clip(model_xgb.predict(test_processed), 0, 1)

# ---------------------------------------------------------
# 8. Blending & Safe Submission
# ---------------------------------------------------------
print("\n -> Blending Predictions...")

# 60/40 Weighted Blend
final_val_preds = (0.6 * lgb_val_preds) + (0.4 * xgb_val_preds)
final_test_preds = (0.6 * lgb_test_preds) + (0.4 * xgb_test_preds)

r2_lgb = r2_score(y_val, lgb_val_preds)
r2_xgb = r2_score(y_val, xgb_val_preds)
r2_final = r2_score(y_val, final_val_preds)

print(f"\n--- Final Output State ---")
print(f"LightGBM Solo R2: {r2_lgb:.6f}")
print(f"XGBoost Solo R2:  {r2_xgb:.6f}")
print(f"BLENDED ENSEMBLE R2: {r2_final:.6f}")
print(f"Expected Hackathon Score: {max(0, 100 * r2_final):.4f}")

# SECURE ID MAPPING
submission = pd.DataFrame({"Index": test_df["Index"], "demand": final_test_preds})

# Sort the submission back to its original numerical sequence
submission = submission.sort_values("Index").reset_index(drop=True)
submission.to_csv("submission_ensemble_deep.csv", index=False)
print("Saved to submission_ensemble_deep.csv successfully!")
