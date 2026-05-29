import pandas as pd
import numpy as np
import pygeohash as pgh
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings('ignore')

print("🎯 Initializing V10: The Confidence Sniper...")

# =========================================================
# PHASE 1: THE SCOUT (Finding the Safe Rows)
# =========================================================
print("\n[Phase 1] Scouting for high-confidence predictions...")
train_raw = pd.read_csv('train.csv')
test_raw = pd.read_csv('test.csv')

# We need basic features for the scout (No time-series leakage yet)
def basic_features(df):
    df['day'] = df['day'].astype(int)
    time_split = df['timestamp'].astype(str).str.split(':', expand=True)
    df['hour'] = time_split[0].astype(int)
    df['minute'] = time_split[1].astype(int)
    df['time_decimal'] = df['day'] * 24 + df['hour'] + (df['minute'] / 60.0)
    
    df['geohash_prefix'] = df['geohash'].astype(str).str.slice(0, 5)
    for col in ['RoadType', 'Weather', 'geohash', 'geohash_prefix']:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    
    df['LargeVehicles'] = df['LargeVehicles'].map({'Yes': 1, 'No': 0}).fillna(0).astype(int)
    df['Landmarks'] = df['Landmarks'].map({'Yes': 1, 'No': 0}).fillna(0).astype(int)
    return df.drop(['timestamp', 'minute'], axis=1)

scout_train = basic_features(train_raw.copy())
scout_test = basic_features(test_raw.copy())

X_scout = scout_train.drop(['demand', 'Index', 'time_decimal'], axis=1)
y_scout = scout_train['demand']
X_scout_test = scout_test.drop(['Index', 'time_decimal'], axis=1, errors='ignore')

# Train 3 quick models to check for variance
scout_preds = []
for seed in [42, 1337, 2026]:
    scout_model = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.03, random_state=seed, n_jobs=-1, verbose=-1)
    scout_model.fit(X_scout, y_scout)
    scout_preds.append(np.clip(scout_model.predict(X_scout_test), 0, 1))

# Calculate Mean and Standard Deviation (Variance)
scout_preds_matrix = np.vstack(scout_preds)
test_means = np.mean(scout_preds_matrix, axis=0)
test_stds = np.std(scout_preds_matrix, axis=0)

# ---------------------------------------------------------
# 🎯 THE THRESHOLD: Only trust rows where models strongly agree
# ---------------------------------------------------------
CONFIDENCE_THRESHOLD = 0.015 # If models disagree by more than 1.5%, discard the pseudo-label
confident_mask = test_stds < CONFIDENCE_THRESHOLD

print(f" -> Scout found {np.sum(confident_mask)} high-confidence rows out of {len(test_raw)}.")
print(f" -> Discarding {np.sum(~confident_mask)} hallucinated/conflicted rows.")

# Create the purified pseudo-label dataframe
safe_pseudo_labels = test_raw.copy()
safe_pseudo_labels['demand'] = np.where(confident_mask, test_means, np.nan)

# =========================================================
# PHASE 2: THE SNIPER (Deep Training on Purified Data)
# =========================================================
print("\n[Phase 2] Injecting Safe Labels & Bridging the Gap...")

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Inject ONLY the safe predictions back into the test set
test['demand'] = safe_pseudo_labels['demand']

train['is_train'] = 1
train['is_pseudo'] = 0
test['is_train'] = 0 
# Only flag as pseudo if it actually contains a confident prediction
test['is_pseudo'] = np.where(test['demand'].notna(), 1, 0) 

df = pd.concat([train, test], axis=0, ignore_index=True)

# --- V8/V9 Feature Engineering (Now protected from hallucinations) ---
df['day'] = df['day'].astype(int)
df['geohash_prefix'] = df['geohash'].astype(str).str.slice(0, 5)
time_split = df['timestamp'].astype(str).str.split(':', expand=True)
df['hour'] = time_split[0].astype(int)
df['minute'] = time_split[1].astype(int)
df['minute_bin'] = (df['minute'] // 15) * 15
df['time_slot'] = df['hour'].astype(str).str.zfill(2) + ":" + df['minute_bin'].astype(str).str.zfill(2)
df['time_decimal'] = df['day'] * 24 + df['hour'] + (df['minute'] / 60.0)

# Temporal & Spatial Features
df = df.sort_values(['geohash', 'time_decimal']).reset_index(drop=True)
df['ewma_demand_3hr'] = df.groupby('geohash')['demand'].transform(lambda x: x.shift(1).ewm(alpha=0.3, min_periods=1).mean())

df = df.sort_values(['geohash_prefix', 'time_decimal']).reset_index(drop=True)
df['neighborhood_momentum'] = df.groupby('geohash_prefix')['demand'].transform(lambda x: x.shift(1).rolling(window=4, min_periods=1).mean())
df['neighborhood_shockwave_15m'] = df.groupby(['geohash_prefix', 'time_decimal'])['demand'].transform('max')

df = df.sort_values(['geohash', 'time_decimal']).reset_index(drop=True)
df['lagged_shockwave'] = df.groupby('geohash')['neighborhood_shockwave_15m'].shift(1)

# Categoricals
df['geohash_x_time'] = df['geohash'].astype(str) + "_" + df['time_slot'].astype(str)
df['neighborhood_weather_hour'] = df['geohash_prefix'].astype(str) + "_" + df['Weather'].astype(str) + "_" + df['hour'].astype(str)
df['geo_prefix_weather_road'] = df['geohash_prefix'].astype(str) + "_" + df['Weather'].astype(str) + "_" + df['RoadType'].astype(str)
df['weather_road'] = df['Weather'].astype(str) + "_" + df['RoadType'].astype(str)

for col in ['geohash_x_time', 'neighborhood_weather_hour']:
    df[col] = df.groupby(col)['demand'].transform(lambda x: x.shift(1).expanding().mean())

cat_cols = ['RoadType', 'Weather', 'geohash', 'geohash_prefix', 'time_slot', 'weather_road', 'geo_prefix_weather_road']
for col in cat_cols:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

df['LargeVehicles'] = df['LargeVehicles'].map({'Yes': 1, 'No': 0}).fillna(0).astype(int)
df['Landmarks'] = df['Landmarks'].map({'Yes': 1, 'No': 0}).fillna(0).astype(int)

df = df.drop(['timestamp', 'minute_bin', 'neighborhood_shockwave_15m'], axis=1)

# Split data
train_and_pseudo = df[(df['is_train'] == 1) | (df['is_pseudo'] == 1)].sort_values('time_decimal').reset_index(drop=True)
test_processed = df[df['is_train'] == 0].sort_values('time_decimal').reset_index(drop=True)

X_train_full = train_and_pseudo.drop(['demand', 'is_train', 'is_pseudo', 'time_decimal', 'Index'], axis=1)
# Log-Transformed Target
y_train_full = np.log1p(train_and_pseudo['demand']) 
X_test = test_processed.drop(['demand', 'is_train', 'is_pseudo', 'time_decimal', 'Index'], axis=1)

# --- V9 Log-Scale Trifecta Training ---
print("\n[Phase 2] Training Log-Scale Trifecta on Purified Data...")
SEEDS = [42, 1337, 2026]
lgb_preds, xgb_preds, cat_preds = 0, 0, 0

for seed in SEEDS:
    lgb_model = lgb.LGBMRegressor(n_estimators=3500, learning_rate=0.015, max_depth=12, num_leaves=127, subsample=0.8, colsample_bytree=0.8, random_state=seed, n_jobs=-1, verbose=-1)
    lgb_model.fit(X_train_full, y_train_full)
    lgb_preds += np.clip(np.expm1(lgb_model.predict(X_test)), 0, 1) / len(SEEDS)
    
    xgb_model = xgb.XGBRegressor(n_estimators=2000, learning_rate=0.015, max_depth=8, subsample=0.8, colsample_bytree=0.8, random_state=seed, n_jobs=-1, tree_method='hist')
    xgb_model.fit(X_train_full, y_train_full, verbose=False)
    xgb_preds += np.clip(np.expm1(xgb_model.predict(X_test)), 0, 1) / len(SEEDS)
    
    cat_model = CatBoostRegressor(iterations=2500, learning_rate=0.02, depth=8, random_seed=seed, verbose=0, task_type='CPU')
    cat_model.fit(X_train_full, y_train_full)
    cat_preds += np.clip(np.expm1(cat_model.predict(X_test)), 0, 1) / len(SEEDS)

print("\n -> Blending Final Sniped Predictions...")
final_test_preds = (0.40 * lgb_preds) + (0.30 * xgb_preds) + (0.30 * cat_preds)

submission = pd.DataFrame({
    'Index': test_processed['Index'],
    'demand': final_test_preds
})

submission = submission.sort_values('Index').reset_index(drop=True)
submission.to_csv('submission_v10_sniper.csv', index=False)
print("✅ Saved to submission_v10_sniper.csv successfully!")
