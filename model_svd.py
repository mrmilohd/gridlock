import pandas as pd
import numpy as np
import pygeohash as pgh
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import TruncatedSVD
import warnings

warnings.filterwarnings("ignore")

print("🧊 Initializing V11: 3D Spatiotemporal Matrix Factorization...")

# ---------------------------------------------------------
# 1. Load Clean Data (No Pseudo-Labels)
# ---------------------------------------------------------
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

train["is_train"] = 1
test["is_train"] = 0
test["demand"] = np.nan

df = pd.concat([train, test], axis=0, ignore_index=True)

# ---------------------------------------------------------
# 2. Base Features & Grid Setup
# ---------------------------------------------------------
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

# ---------------------------------------------------------
# 3. CONSTRUCTING THE 3D MAP (Matrix Factorization)
# ---------------------------------------------------------
print(" -> Constructing the Space-Time Matrix...")

# FIX: We filter the combined 'df' for the training rows so 'time_slot' exists
train_for_svd = df[df["is_train"] == 1]
pivot_space = train_for_svd.pivot_table(
    index="geohash", columns="time_slot", values="demand", aggfunc="mean"
).fillna(0)

print(" -> Extracting Latent Spatial Embeddings via SVD...")
# Compress the 96 daily 15-minute slots into 8 core "personality traits" for each geohash
n_components = 8
svd = TruncatedSVD(n_components=n_components, random_state=42)
space_embeddings = svd.fit_transform(pivot_space)

# Create a dataframe of these new 3D coordinates
svd_cols = [f"svd_space_{i}" for i in range(n_components)]
df_space_emb = pd.DataFrame(
    space_embeddings, columns=svd_cols, index=pivot_space.index
).reset_index()

# Merge the fundamental coordinates back into the main timeline
df = df.merge(df_space_emb, on="geohash", how="left")

# If any geohash in the test set never existed in training, fill its SVD with 0
for col in svd_cols:
    df[col] = df[col].fillna(0)

# ---------------------------------------------------------
# 4. Standard Categoricals & Clean Up
# ---------------------------------------------------------
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

df["LargeVehicles"] = df["LargeVehicles"].map({"Yes": 1, "No": 0}).fillna(0).astype(int)
df["Landmarks"] = df["Landmarks"].map({"Yes": 1, "No": 0}).fillna(0).astype(int)

df = df.drop(["timestamp", "minute_bin", "latitude", "longitude"], axis=1)

# ---------------------------------------------------------
# 5. Split & Train (Log-Transformed)
# ---------------------------------------------------------
print(" -> Slicing Data and Initiating Log-Scale Training...")
train_clean = df[df["is_train"] == 1].sort_values("time_decimal").reset_index(drop=True)
test_clean = df[df["is_train"] == 0].sort_values("time_decimal").reset_index(drop=True)

X_train = train_clean.drop(["demand", "is_train", "time_decimal", "Index"], axis=1)
y_train = np.log1p(train_clean["demand"])  # Still using the log shield against spikes

X_test = test_clean.drop(["demand", "is_train", "time_decimal", "Index"], axis=1)

# --- 3-Model Baseline ---
print("Training SVD-Powered LightGBM...")
lgb_model = lgb.LGBMRegressor(
    n_estimators=3500,
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
lgb_preds = np.clip(np.expm1(lgb_model.predict(X_test)), 0, 1)

print("Training SVD-Powered XGBoost...")
xgb_model = xgb.XGBRegressor(
    n_estimators=2000,
    learning_rate=0.015,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    tree_method="hist",
)
xgb_model.fit(X_train, y_train, verbose=False)
xgb_preds = np.clip(np.expm1(xgb_model.predict(X_test)), 0, 1)

print("Training SVD-Powered CatBoost...")
cat_model = CatBoostRegressor(
    iterations=2500,
    learning_rate=0.02,
    depth=8,
    random_seed=42,
    verbose=0,
    task_type="CPU",
)
cat_model.fit(X_train, y_train)
cat_preds = np.clip(np.expm1(cat_model.predict(X_test)), 0, 1)

# ---------------------------------------------------------
# 6. Final Blend
# ---------------------------------------------------------
print("\n -> Blending SVD Predictions...")
final_test_preds = (0.40 * lgb_preds) + (0.30 * xgb_preds) + (0.30 * cat_preds)

submission = pd.DataFrame({"Index": test_clean["Index"], "demand": final_test_preds})

submission = submission.sort_values("Index").reset_index(drop=True)
submission.to_csv("submission_v11_svd_3map.csv", index=False)
print("✅ Saved to submission_v11_svd_3map.csv successfully! The map is complete.")
