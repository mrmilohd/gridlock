import pandas as pd
import numpy as np
import pygeohash as pgh
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score
import warnings

warnings.filterwarnings("ignore")

print("🔥 Initializing V14: The Baseline Restored (No SVD, Pure Interactions)...")

# ---------------------------------------------------------
# 1. Load Data & Strict Chronological Split
# ---------------------------------------------------------
print(" -> Loading and Slicing 80/20 Chronological Validation Split...")
train_raw = pd.read_csv("train.csv")

train_raw["day"] = train_raw["day"].astype(int)
time_split = train_raw["timestamp"].astype(str).str.split(":", expand=True)
train_raw["hour"] = time_split[0].astype(int)
train_raw["minute"] = time_split[1].astype(int)
train_raw["time_decimal"] = (
    train_raw["day"] * 24 + train_raw["hour"] + (train_raw["minute"] / 60.0)
)

# Sort strictly by time to prevent future leakage
train_raw = train_raw.sort_values("time_decimal").reset_index(drop=True)

split_idx = int(len(train_raw) * 0.80)
train = train_raw.iloc[:split_idx].copy()
val = train_raw.iloc[split_idx:].copy()

train["is_train"] = 1
val["is_train"] = 0

df = pd.concat([train, val], axis=0, ignore_index=True)

# ---------------------------------------------------------
# 2. Base Spatial & Temporal Engineering
# ---------------------------------------------------------
print(" -> Engineering Temporal and Spatial Foundations...")
df["latitude"] = df["geohash"].apply(
    lambda x: pgh.decode(x)[0] if pd.notnull(x) else np.nan
)
df["longitude"] = df["geohash"].apply(
    lambda x: pgh.decode(x)[1] if pd.notnull(x) else np.nan
)
df["geohash_prefix"] = df["geohash"].astype(str).str.slice(0, 5)
df["minute_bin"] = (df["minute"] // 15) * 15
df["time_slot"] = (
    df["hour"].astype(str).str.zfill(2)
    + ":"
    + df["minute_bin"].astype(str).str.zfill(2)
)

# The Kaggle Post Secret: Cyclical Time
df["time_sin"] = np.sin(2 * np.pi * df["time_decimal"] / 24.0)
df["time_cos"] = np.cos(2 * np.pi * df["time_decimal"] / 24.0)

# Spatial Clustering (Macro-neighborhoods)
coords_train = df[df["is_train"] == 1][["latitude", "longitude"]].fillna(
    df[["latitude", "longitude"]].mean()
)
coords_all = df[["latitude", "longitude"]].fillna(df[["latitude", "longitude"]].mean())
kmeans = KMeans(n_clusters=60, random_state=42, n_init=10)
kmeans.fit(coords_train)
df["spatial_cluster"] = kmeans.predict(coords_all)

# ---------------------------------------------------------
# 3. The Grandmaster Interaction Features
# ---------------------------------------------------------
print(" -> Applying Geohash x Time Interactions...")

df["geo_x_time"] = df["geohash"].astype(str) + "_" + df["time_slot"].astype(str)
df["weather_road"] = df["Weather"].astype(str) + "_" + df["RoadType"].astype(str)
df["neighborhood_weather_hour"] = (
    df["geohash_prefix"].astype(str)
    + "_"
    + df["Weather"].astype(str)
    + "_"
    + df["hour"].astype(str)
)

# Target Encoding (Strictly mapped from the training split)
train_te = df[df["is_train"] == 1]
global_median = train_te["demand"].median()

# Feature 1: How busy is this exact intersection at this exact 15-min slot?
te_geo_time = train_te.groupby("geo_x_time")["demand"].mean()
df["te_geo_time"] = df["geo_x_time"].map(te_geo_time).fillna(global_median)

# Feature 2: How does this neighborhood react to this weather at this hour?
te_neigh_weath = train_te.groupby("neighborhood_weather_hour")["demand"].mean()
df["te_neighborhood_weather_hour"] = (
    df["neighborhood_weather_hour"].map(te_neigh_weath).fillna(global_median)
)

# Encode all string categoricals
categorical_cols = [
    "RoadType",
    "Weather",
    "geohash",
    "geohash_prefix",
    "time_slot",
    "weather_road",
    "geo_x_time",
    "neighborhood_weather_hour",
]
for col in categorical_cols:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

df["LargeVehicles"] = df["LargeVehicles"].map({"Yes": 1, "No": 0}).fillna(0).astype(int)
df["Landmarks"] = df["Landmarks"].map({"Yes": 1, "No": 0}).fillna(0).astype(int)

# Drop redundant geometric data
df = df.drop(["timestamp", "minute_bin", "latitude", "longitude"], axis=1)

# ---------------------------------------------------------
# 4. Prepare Final Tensors
# ---------------------------------------------------------
print(" -> Slicing Data for Validation Pipeline...")
train_clean = df[df["is_train"] == 1].sort_values("time_decimal").reset_index(drop=True)
val_clean = df[df["is_train"] == 0].sort_values("time_decimal").reset_index(drop=True)

drop_cols = ["demand", "is_train", "time_decimal", "Index", "day", "hour", "minute"]

X_train = train_clean.drop(drop_cols, axis=1, errors="ignore")
y_train = np.log1p(train_clean["demand"])  # Log Transform Target to handle spikes

X_val = val_clean.drop(drop_cols, axis=1, errors="ignore")
y_val = val_clean["demand"]  # Keep absolute for the R2 score

# ---------------------------------------------------------
# 5. Training The Log-Scale Trifecta
# ---------------------------------------------------------
print("\n[1/3] Training Interactions-Powered LightGBM...")
lgb_model = lgb.LGBMRegressor(
    n_estimators=3000,
    learning_rate=0.015,
    max_depth=12,
    num_leaves=127,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)
lgb_model.fit(X_train, y_train)
lgb_preds = np.clip(np.expm1(lgb_model.predict(X_val)), 0, 1)

print("[2/3] Training Interactions-Powered XGBoost...")
xgb_model = xgb.XGBRegressor(
    n_estimators=1500,
    learning_rate=0.015,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    tree_method="hist",
)
xgb_model.fit(X_train, y_train, verbose=False)
xgb_preds = np.clip(np.expm1(xgb_model.predict(X_val)), 0, 1)

print("[3/3] Training Interactions-Powered CatBoost...")
cat_model = CatBoostRegressor(
    iterations=2000,
    learning_rate=0.02,
    depth=8,
    random_seed=42,
    verbose=0,
    task_type="CPU",
)
cat_model.fit(X_train, y_train)
cat_preds = np.clip(np.expm1(cat_model.predict(X_val)), 0, 1)

# ---------------------------------------------------------
# 6. Evaluation
# ---------------------------------------------------------
print("\n -> Blending the Validation Predictions...")
final_val_preds = (0.40 * lgb_preds) + (0.30 * xgb_preds) + (0.30 * cat_preds)

r2 = r2_score(y_val, final_val_preds)

print("\n" + "=" * 50)
print(f"📊 TRUE LOCAL R2 SCORE (80/20 Chrono Split): {r2:.6f}")
print("=" * 50)
