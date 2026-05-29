import pandas as pd
import numpy as np
import pygeohash as pgh
import lightgbm as lgb
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
import warnings

warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# 1. Load Data & Inject the 87-Scoring Pseudo-Labels
# ---------------------------------------------------------
print("Loading data and injecting 87-Scoring Pseudo-Labels...")
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Fixed: Loading your best live leaderboard submission
best_sub = pd.read_csv("submission_final.csv")

# Merge the highly accurate predictions into the test set
test = test.merge(best_sub, on="Index", how="left")

train["is_train"] = 1
train["is_pseudo"] = 0

test["is_train"] = 0
test["is_pseudo"] = 1  # Mark these as our artificial training rows

# Combine into one massive, continuous timeline
df = pd.concat([train, test], axis=0, ignore_index=True)

# ---------------------------------------------------------
# 2. Grandmaster Feature Engineering (V6 Architecture)
# ---------------------------------------------------------
print("Starting Feature Engineering on full timeline...")
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

print(" -> Calculating Temporal Momentum (Bridging the Gap)...")
df = df.sort_values(["geohash", "time_decimal"]).reset_index(drop=True)
df["ewma_demand_3hr"] = df.groupby("geohash")["demand"].transform(
    lambda x: x.shift(1).ewm(alpha=0.3, min_periods=1).mean()
)

df = df.sort_values(["geohash_prefix", "time_decimal"]).reset_index(drop=True)
df["neighborhood_momentum"] = df.groupby("geohash_prefix")["demand"].transform(
    lambda x: x.shift(1).rolling(window=4, min_periods=1).mean()
)

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

# CRITICAL FIX: Keep Index for perfect mapping
df = df.drop(["timestamp", "minute_bin", "latitude", "longitude"], axis=1)

# ---------------------------------------------------------
# 3. Full Data Preparation (No Validation Split)
# ---------------------------------------------------------
print(" -> Preparing Data for Full Training...")

# Train on BOTH original train data AND the highly accurate pseudo-labeled test data
train_and_pseudo = (
    df[(df["is_train"] == 1) | (df["is_pseudo"] == 1)]
    .sort_values("time_decimal")
    .reset_index(drop=True)
)

# The target we actually want to predict and submit
test_processed = df[df["is_pseudo"] == 1].reset_index(drop=True)

X_train_full = train_and_pseudo.drop(
    ["demand", "is_train", "is_pseudo", "time_decimal", "Index"], axis=1
)
y_train_full = train_and_pseudo["demand"]

X_test = test_processed.drop(
    ["demand", "is_train", "is_pseudo", "time_decimal", "Index"], axis=1
)

# ---------------------------------------------------------
# 4. Ensemble Training (Full Data Mode)
# ---------------------------------------------------------
print("Training Final LightGBM on Full Dataset + Pseudo-Labels...")
# We use fixed, robust estimators since we can't use early stopping here
lgb_params = {
    "n_estimators": 4500,
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
}

model_lgb = lgb.LGBMRegressor(**lgb_params)
model_lgb.fit(X_train_full, y_train_full)
lgb_test_preds = np.clip(model_lgb.predict(X_test), 0, 1)

print("Training Final XGBoost on Full Dataset + Pseudo-Labels...")
xgb_params = {
    "n_estimators": 2500,
    "learning_rate": 0.015,
    "max_depth": 8,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_jobs": -1,
    "objective": "reg:squarederror",
    "tree_method": "hist",
}

model_xgb = xgb.XGBRegressor(**xgb_params)
model_xgb.fit(X_train_full, y_train_full, verbose=False)
xgb_test_preds = np.clip(model_xgb.predict(X_test), 0, 1)

# ---------------------------------------------------------
# 5. Blending & Submission
# ---------------------------------------------------------
print(" -> Blending Pseudo-Labeled Predictions...")
final_test_preds = (0.6 * lgb_test_preds) + (0.4 * xgb_test_preds)

submission = pd.DataFrame(
    {"Index": test_processed["Index"], "demand": final_test_preds}
)

submission = submission.sort_values("Index").reset_index(drop=True)
submission.to_csv("submission_ultimate.csv", index=False)
print("Saved to submission_ultimate.csv successfully! Time to claim the top spot.")
