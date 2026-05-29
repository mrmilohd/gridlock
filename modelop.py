import pandas as pd
import numpy as np
import pygeohash as pgh
import lightgbm as lgb
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
from sklearn.cluster import KMeans
import optuna
import warnings

warnings.filterwarnings("ignore")


# ---------------------------------------------------------
# 1. Feature Engineering (The Grandmaster Setup)
# ---------------------------------------------------------
def prepare_data():
    print("Loading and engineering data for Joint Optuna...")
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")

    train["is_train"] = 1
    test["is_train"] = 0
    test["demand"] = np.nan

    df = pd.concat([train, test], axis=0, ignore_index=True)

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

    coords = df[["latitude", "longitude"]].fillna(df[["latitude", "longitude"]].mean())
    kmeans = KMeans(n_clusters=60, random_state=42, n_init=10)
    df["spatial_cluster"] = kmeans.fit_predict(coords)

    day48 = df[(df["day"] == 48) & (df["is_train"] == 1)]
    geo_stats = (
        day48.groupby("geohash")["demand"].agg(["mean", "std", "max"]).reset_index()
    )
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

    df = df.sort_values(["geohash", "time_decimal"]).reset_index(drop=True)
    df["ewma_demand_3hr"] = df.groupby("geohash")["demand"].transform(
        lambda x: x.shift(1).ewm(alpha=0.3, min_periods=1).mean()
    )

    df = df.sort_values(["geohash_prefix", "time_decimal"]).reset_index(drop=True)
    df["neighborhood_momentum"] = df.groupby("geohash_prefix")["demand"].transform(
        lambda x: x.shift(1).rolling(window=4, min_periods=1).mean()
    )

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

    df = df.drop(["timestamp", "minute_bin", "latitude", "longitude"], axis=1)

    train_df = (
        df[df["is_train"] == 1].sort_values("time_decimal").reset_index(drop=True)
    )
    test_df = df[df["is_train"] == 0].reset_index(drop=True)

    split_idx = int(len(train_df) * 0.8)

    X_train = train_df.iloc[:split_idx].drop(
        ["demand", "is_train", "time_decimal", "Index"], axis=1
    )
    y_train = train_df.iloc[:split_idx]["demand"]

    X_val = train_df.iloc[split_idx:].drop(
        ["demand", "is_train", "time_decimal", "Index"], axis=1
    )
    y_val = train_df.iloc[split_idx:]["demand"]

    test_processed = test_df.drop(
        ["demand", "is_train", "time_decimal", "Index"], axis=1
    )

    return X_train, y_train, X_val, y_val, test_processed, test_df


# Load data into memory once
X_train, y_train, X_val, y_val, test_processed, test_df = prepare_data()


# ---------------------------------------------------------
# 2. Optuna Objective (Joint Optimization)
# ---------------------------------------------------------
def objective(trial):
    # --- LightGBM Hyperparameters ---
    lgb_params = {
        "n_estimators": 1500,  # Capped for tuning speed
        "learning_rate": trial.suggest_float("lgb_lr", 0.005, 0.05, log=True),
        "max_depth": trial.suggest_int("lgb_depth", 6, 15),
        "num_leaves": trial.suggest_int("lgb_leaves", 31, 200),
        "min_child_samples": trial.suggest_int("lgb_min_child", 5, 50),
        "subsample": trial.suggest_float("lgb_subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("lgb_colsample", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("lgb_alpha", 1e-4, 5.0, log=True),
        "reg_lambda": trial.suggest_float("lgb_lambda", 1e-4, 5.0, log=True),
        "random_state": 42,
        "n_jobs": -1,
        "metric": "rmse",
    }

    # --- XGBoost Hyperparameters ---
    xgb_params = {
        "n_estimators": 1000,  # Capped for tuning speed
        "learning_rate": trial.suggest_float("xgb_lr", 0.005, 0.05, log=True),
        "max_depth": trial.suggest_int(
            "xgb_depth", 4, 10
        ),  # XGBoost needs shallower trees
        "subsample": trial.suggest_float("xgb_subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("xgb_colsample", 0.6, 1.0),
        "random_state": 42,
        "n_jobs": -1,
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
    }

    # --- The Blend Weight ---
    # Optuna will figure out exactly how much to trust LGBM vs XGB
    lgb_weight = trial.suggest_float("lgb_weight", 0.1, 0.9)

    # Train LightGBM
    model_lgb = lgb.LGBMRegressor(**lgb_params)
    model_lgb.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    lgb_preds = np.clip(model_lgb.predict(X_val), 0, 1)

    # Train XGBoost
    model_xgb = xgb.XGBRegressor(**xgb_params, early_stopping_rounds=50)
    model_xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    xgb_preds = np.clip(model_xgb.predict(X_val), 0, 1)

    # Blend and Evaluate
    final_preds = (lgb_weight * lgb_preds) + ((1.0 - lgb_weight) * xgb_preds)
    r2 = r2_score(y_val, final_preds)

    return r2


# ---------------------------------------------------------
# 3. Execution & Auto-Submission
# ---------------------------------------------------------
if __name__ == "__main__":
    print("\n🚀 Initializing Optuna Joint Hyperparameter Hunt (30 Trials)...")

    # We want to MAXIMIZE the blended R2 score
    study = optuna.create_study(direction="maximize")
    study.optimize(
        objective, n_trials=30
    )  # 30 trials is a sweet spot for time vs performance

    print("\n" + "=" * 50)
    print("🏆 OPTIMIZATION COMPLETE 🏆")
    print(f"Best Validation R2: {study.best_value:.6f}")

    # Extract the winning parameters
    best = study.best_trial.params
    lgb_weight = best.pop("lgb_weight")

    lgb_best_params = {
        k.replace("lgb_", ""): v for k, v in best.items() if k.startswith("lgb_")
    }
    xgb_best_params = {
        k.replace("xgb_", ""): v for k, v in best.items() if k.startswith("xgb_")
    }

    # Add static params back in for the final massive run
    lgb_best_params.update(
        {"n_estimators": 8000, "random_state": 42, "n_jobs": -1, "metric": "rmse"}
    )
    xgb_best_params.update(
        {
            "n_estimators": 5000,
            "random_state": 42,
            "n_jobs": -1,
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "tree_method": "hist",
        }
    )

    print("\n⚙️ Automatically Training Final Deep Models with Winning Parameters...")

    print(f" -> Training LightGBM (Weight: {lgb_weight:.2f})")
    final_lgb = lgb.LGBMRegressor(**lgb_best_params)
    final_lgb.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(300, verbose=False)],
    )
    lgb_test_preds = np.clip(final_lgb.predict(test_processed), 0, 1)

    print(f" -> Training XGBoost (Weight: {(1.0 - lgb_weight):.2f})")
    final_xgb = xgb.XGBRegressor(**xgb_best_params, early_stopping_rounds=300)
    final_xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    xgb_test_preds = np.clip(final_xgb.predict(test_processed), 0, 1)

    print(" -> Blending and Saving...")
    final_test_preds = (lgb_weight * lgb_test_preds) + (
        (1.0 - lgb_weight) * xgb_test_preds
    )

    submission = pd.DataFrame({"Index": test_df["Index"], "demand": final_test_preds})
    submission = submission.sort_values("Index").reset_index(drop=True)
    submission.to_csv("submission_optuna_optimized.csv", index=False)

    print("✅ Saved to submission_optuna_optimized.csv successfully! Good luck!")
