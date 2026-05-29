
"""
OPTIMIZED LIGHTGBM TRAFFIC DEMAND PIPELINE
Best stable version
"""

import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder

import lightgbm as lgb

# ============================================================
# 1. LOAD DATA
# ============================================================

print("=" * 60)
print("STEP 1: Loading data")
print("=" * 60)

train = pd.read_csv("train_cleaned.csv")
test  = pd.read_csv("test.csv")

print(f"Train shape : {train.shape}")
print(f"Test shape  : {test.shape}")

# ============================================================
# 2. TIMESTAMP FEATURES
# ============================================================

def parse_timestamp(df):

    ts = df["timestamp"].str.split(":", expand=True).astype(int)

    df["hour"] = ts[0]
    df["minute"] = ts[1]

    df["time_slot"] = (
        df["hour"] * 4
        + df["minute"] // 15
    )

    return df

train = parse_timestamp(train)
test  = parse_timestamp(test)

# ============================================================
# 3. GEOHASH PREFIX FEATURES
# ============================================================

for df in [train, test]:

    df["geo4"] = df["geohash"].str[:4]
    df["geo5"] = df["geohash"].str[:5]

# ============================================================
# 4. TEST IMPUTATION
# ============================================================

print("\nSTEP 2: Imputing test set")

def fill_test(df, ref):

    # ========================================================
    # ROADTYPE
    # ========================================================

    geo_road = (
        ref.groupby("geohash")["RoadType"]
        .agg(lambda x: x.mode()[0])
    )

    geo4_road = (
        ref.groupby("geo4")["RoadType"]
        .agg(lambda x: x.mode()[0])
    )

    df["RoadType"] = (

        df["RoadType"]

        .fillna(df["geohash"].map(geo_road))

        .fillna(df["geo4"].map(geo4_road))

        .fillna(ref["RoadType"].mode()[0])
    )

    # ========================================================
    # WEATHER
    # ========================================================

    geo_weather = (
        ref.groupby("geohash")["Weather"]
        .agg(lambda x: x.mode()[0])
    )

    df["Weather"] = (
        df["Weather"]
        .fillna(df["geohash"].map(geo_weather))
    )

    # infer weather from temp

    mask = df["Weather"].isna()

    def infer_weather(temp):

        if pd.isna(temp):
            return np.nan

        if temp < 2:
            return "Snowy"

        elif temp < 10:
            return "Rainy"

        elif temp < 20:
            return "Cloudy"

        return "Sunny"

    df.loc[mask, "Weather"] = (
        df.loc[mask, "Temperature"]
        .apply(infer_weather)
    )

    df["Weather"] = (
        df["Weather"]
        .fillna(ref["Weather"].mode()[0])
    )

    # ========================================================
    # TEMPERATURE
    # ========================================================

    geo_temp = (
        ref.groupby(
            ["geohash", "day", "hour"]
        )["Temperature"]
        .median()
    )

    geo_temp2 = (
        ref.groupby(
            ["geohash", "hour"]
        )["Temperature"]
        .median()
    )

    global_temp = ref["Temperature"].median()

    mask = df["Temperature"].isna()

    df.loc[mask, "Temperature"] = (

        df.loc[mask, ["geohash", "day", "hour"]]

        .apply(

            lambda r: geo_temp.get(
                (r["geohash"], r["day"], r["hour"]),
                np.nan
            ),

            axis=1
        )
    )

    mask = df["Temperature"].isna()

    df.loc[mask, "Temperature"] = (

        df.loc[mask, ["geohash", "hour"]]

        .apply(

            lambda r: geo_temp2.get(
                (r["geohash"], r["hour"]),
                global_temp
            ),

            axis=1
        )
    )

    df["Temperature"] = (
        df["Temperature"]
        .fillna(global_temp)
    )

    return df

test = fill_test(test, train)

print(f"Train nulls : {train.isnull().sum().sum()}")
print(f"Test  nulls : {test.isnull().sum().sum()}")

# ============================================================
# 5. CYCLIC FEATURES
# ============================================================

def cyclic(df, col, mx):

    df[f"{col}_sin"] = np.sin(
        2 * np.pi * df[col] / mx
    )

    df[f"{col}_cos"] = np.cos(
        2 * np.pi * df[col] / mx
    )

    return df

for col, mx in [

    ("hour", 24),
    ("minute", 60),
    ("time_slot", 96),

]:

    train = cyclic(train, col, mx)
    test  = cyclic(test, col, mx)

# ============================================================
# 6. TEMPORAL FEATURES
# ============================================================

print("\nSTEP 3: Feature engineering")

day48 = train[
    train["day"] == 48
].copy()

day49 = train[
    train["day"] == 49
].copy()

# ============================================================
# SAME SLOT PREVIOUS DAY
# ============================================================

lag_exact = (

    day48[
        ["geohash", "time_slot", "demand"]
    ]

    .rename(columns={
        "demand": "lag_exact"
    })
)

train = train.merge(
    lag_exact,
    on=["geohash", "time_slot"],
    how="left"
)

test = test.merge(
    lag_exact,
    on=["geohash", "time_slot"],
    how="left"
)

# ============================================================
# TEMPORAL NEIGHBOR LAGS
# ============================================================

for shift in [-2, -1, 1, 2]:

    tmp = lag_exact.copy()

    tmp["time_slot"] += shift

    tmp.rename(
        columns={
            "lag_exact": f"lag_{shift}"
        },
        inplace=True
    )

    train = train.merge(
        tmp,
        on=["geohash", "time_slot"],
        how="left"
    )

    test = test.merge(
        tmp,
        on=["geohash", "time_slot"],
        how="left"
    )

lag_cols = [

    "lag_exact",

    "lag_-2",
    "lag_-1",

    "lag_1",
    "lag_2",
]

for col in lag_cols:

    train[col] = (
        train[col]
        .fillna(train["lag_exact"])
        .fillna(0)
    )

    test[col] = (
        test[col]
        .fillna(test["lag_exact"])
        .fillna(0)
    )

# ============================================================
# TEMPORAL SMOOTHING
# ============================================================

train["lag_mean_5"] = (
    train[lag_cols]
    .mean(axis=1)
)

test["lag_mean_5"] = (
    test[lag_cols]
    .mean(axis=1)
)

train["lag_std_5"] = (
    train[lag_cols]
    .std(axis=1)
)

test["lag_std_5"] = (
    test[lag_cols]
    .std(axis=1)
)

# ============================================================
# DAY49 CONTEXT
# ============================================================

context = (

    day49.groupby("geohash")

    .agg(
        d49_mean=("demand", "mean"),
        d49_last=("demand", "last"),
    )

    .reset_index()
)

train = train.merge(
    context,
    on="geohash",
    how="left"
)

test = test.merge(
    context,
    on="geohash",
    how="left"
)

# ============================================================
# GEOHASH STATS
# ============================================================

geo_stats = (

    day48.groupby("geohash")["demand"]

    .agg(
        gh_mean="mean",
        gh_std="std",
        gh_max="max",
        gh_min="min",
    )

    .reset_index()
)

train = train.merge(
    geo_stats,
    on="geohash",
    how="left"
)

test = test.merge(
    geo_stats,
    on="geohash",
    how="left"
)

# ============================================================
# GEOHASH × HOUR
# ============================================================

gh_hour = (

    day48.groupby(
        ["geohash", "hour"]
    )["demand"]

    .mean()

    .rename("gh_hour_mean")

    .reset_index()
)

train = train.merge(
    gh_hour,
    on=["geohash", "hour"],
    how="left"
)

test = test.merge(
    gh_hour,
    on=["geohash", "hour"],
    how="left"
)

# ============================================================
# ROADTYPE × HOUR
# ============================================================

road_hour = (

    day48.groupby(
        ["RoadType", "hour"]
    )["demand"]

    .mean()

    .rename("road_hour_mean")

    .reset_index()
)

train = train.merge(
    road_hour,
    on=["RoadType", "hour"],
    how="left"
)

test = test.merge(
    road_hour,
    on=["RoadType", "hour"],
    how="left"
)

# ============================================================
# SPATIAL PREFIX FEATURES
# ============================================================

geo4_mean = (
    day48.groupby("geo4")["demand"]
    .mean()
)

geo5_mean = (
    day48.groupby("geo5")["demand"]
    .mean()
)

train["geo4_mean"] = (
    train["geo4"]
    .map(geo4_mean)
)

test["geo4_mean"] = (
    test["geo4"]
    .map(geo4_mean)
)

train["geo5_mean"] = (
    train["geo5"]
    .map(geo5_mean)
)

test["geo5_mean"] = (
    test["geo5"]
    .map(geo5_mean)
)

# ============================================================
# FIX NaNs
# ============================================================

fill_cols = [

    "d49_mean",
    "d49_last",

    "gh_mean",
    "gh_std",
    "gh_max",
    "gh_min",

    "gh_hour_mean",
    "road_hour_mean",

    "geo4_mean",
    "geo5_mean",
]

for col in fill_cols:

    train[col] = (
        train[col]
        .fillna(train[col].median())
    )

    test[col] = (
        test[col]
        .fillna(train[col].median())
    )

# ============================================================
# ENCODING
# ============================================================

cat_cols = [

    "RoadType",
    "LargeVehicles",
    "Landmarks",
    "Weather",
]

for col in cat_cols:

    le = LabelEncoder()

    combined = pd.concat([
        train[col],
        test[col]
    ]).astype(str)

    le.fit(combined)

    train[col + "_enc"] = le.transform(
        train[col].astype(str)
    )

    test[col + "_enc"] = le.transform(
        test[col].astype(str)
    )

# geohash

le_geo = LabelEncoder()

combined_geo = pd.concat([
    train["geohash"],
    test["geohash"]
])

le_geo.fit(combined_geo)

train["geohash_enc"] = le_geo.transform(
    train["geohash"]
)

test["geohash_enc"] = le_geo.transform(
    test["geohash"]
)

# ============================================================
# FEATURES
# ============================================================

FEATURES = [

    "geohash_enc",

    "hour",
    "minute",
    "time_slot",

    "hour_sin",
    "hour_cos",

    "minute_sin",
    "minute_cos",

    "time_slot_sin",
    "time_slot_cos",

    "RoadType_enc",
    "LargeVehicles_enc",
    "Landmarks_enc",
    "Weather_enc",

    "NumberofLanes",
    "Temperature",

    "lag_exact",

    "lag_-2",
    "lag_-1",

    "lag_1",
    "lag_2",

    "lag_mean_5",
    "lag_std_5",

    "d49_mean",
    "d49_last",

    "gh_mean",
    "gh_std",
    "gh_max",
    "gh_min",

    "gh_hour_mean",
    "road_hour_mean",

    "geo4_mean",
    "geo5_mean",
]

TARGET = "demand"

# ============================================================
# TEMPORAL SPLIT
# ============================================================

train_df = train[
    train["day"] == 48
].copy()

valid_df = train[
    train["day"] == 49
].copy()

X_train = train_df[FEATURES]
y_train = train_df[TARGET]

X_valid = valid_df[FEATURES]
y_valid = valid_df[TARGET]

X_test = test[FEATURES]

print(f"\nTrain shape : {X_train.shape}")
print(f"Valid shape : {X_valid.shape}")
print(f"Test shape  : {X_test.shape}")

# ============================================================
# LIGHTGBM
# ============================================================

print("\nSTEP 4: Training LightGBM")

params = {

    "objective": "regression",
    "metric": "rmse",

    "learning_rate": 0.025,

    "num_leaves": 64,

    "max_depth": 10,

    "min_child_samples": 40,

    "feature_fraction": 0.85,
    "bagging_fraction": 0.85,
    "bagging_freq": 1,

    "reg_alpha": 0.15,
    "reg_lambda": 0.15,

    "seed": 42,

    "verbose": -1,
}

train_ds = lgb.Dataset(
    X_train,
    label=y_train
)

valid_ds = lgb.Dataset(
    X_valid,
    label=y_valid
)

model = lgb.train(

    params,

    train_ds,

    num_boost_round=3000,

    valid_sets=[valid_ds],

    callbacks=[

        lgb.early_stopping(150),

        lgb.log_evaluation(100),
    ]
)

# ============================================================
# PREDICTIONS
# ============================================================

valid_pred = model.predict(X_valid)
test_pred  = model.predict(X_test)

valid_pred = np.clip(
    valid_pred,
    0,
    1
)

test_pred = np.clip(
    test_pred,
    0,
    1
)

# ============================================================
# EVALUATION
# ============================================================

val_r2 = r2_score(
    y_valid,
    valid_pred
)

print("\n" + "=" * 60)
print("FINAL RESULTS")
print("=" * 60)

print(f"Validation R² : {val_r2:.5f}")

# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({

    "feature": FEATURES,

    "importance": model.feature_importance()
})

importance = importance.sort_values(
    by="importance",
    ascending=False
)

print("\nTop 15 Features:")
print(importance.head(15))

# ============================================================
# SAVE SUBMISSION
# ============================================================

submission = pd.DataFrame({

    "Index": test["Index"],
    "demand": test_pred,
})

submission.to_csv(
    "submission.csv",
    index=False
)

print("\nSubmission saved successfully!")

