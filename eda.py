import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")


def run_eda():
    print("=" * 60)
    print("🚦 GRIDLOCK HACKATHON: SPATIO-TEMPORAL EDA REPORT 🚦")
    print("=" * 60 + "\n")

    # 1. Load Data
    print("Loading datasets...")
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")

    train["is_train"] = 1
    test["is_train"] = 0
    test["demand"] = np.nan
    df = pd.concat([train, test], axis=0, ignore_index=True)

    # Convert day to int and parse time safely
    df["day"] = df["day"].astype(int)
    time_split = df["timestamp"].astype(str).str.split(":", expand=True)
    df["hour"] = time_split[0].astype(int)
    df["minute"] = time_split[1].astype(int)
    df["time_decimal"] = df["day"] * 24 + df["hour"] + (df["minute"] / 60.0)

    # ---------------------------------------------------------
    print("\n[1] BASIC SHAPES & SPATIAL OVERLAP")
    print("-" * 40)
    print(f"Train rows: {len(train):,}")
    print(f"Test rows:  {len(test):,}")

    train_geo = set(train["geohash"].unique())
    test_geo = set(test["geohash"].unique())

    print(f"Unique Geohashes in Train: {len(train_geo)}")
    print(f"Unique Geohashes in Test:  {len(test_geo)}")

    missing_in_train = test_geo - train_geo
    if len(missing_in_train) > 0:
        print(
            f"🚨 WARNING: {len(missing_in_train)} geohashes in Test are COMPLETELY UNSEEN in Train."
        )
    else:
        print("✅ All test geohashes exist in the training set.")

    # ---------------------------------------------------------
    print("\n[2] TIMELINE MAPPING & GAPS")
    print("-" * 40)
    days = sorted(df["day"].unique())
    for d in days:
        d_train = df[(df["day"] == d) & (df["is_train"] == 1)]
        d_test = df[(df["day"] == d) & (df["is_train"] == 0)]

        train_times = d_train["timestamp"].nunique()
        test_times = d_test["timestamp"].nunique()

        train_min = d_train["timestamp"].min() if not d_train.empty else "N/A"
        train_max = d_train["timestamp"].max() if not d_train.empty else "N/A"

        test_min = d_test["timestamp"].min() if not d_test.empty else "N/A"
        test_max = d_test["timestamp"].max() if not d_test.empty else "N/A"

        print(f"Day {d}:")
        print(
            f"  -> Train: {train_times} distinct times | Range: {train_min} to {train_max}"
        )
        print(
            f"  -> Test:  {test_times} distinct times | Range: {test_min} to {test_max}"
        )

    # ---------------------------------------------------------
    print("\n[3] SPATIO-TEMPORAL GRID SPARSITY")
    print("-" * 40)
    # Check if every geohash has an entry for every timestamp, or if rows are missing
    max_possible_rows = (
        len(train_geo) * train["timestamp"].nunique() * train["day"].nunique()
    )
    actual_rows = len(train)
    sparsity = 100 - (actual_rows / max_possible_rows * 100)
    print(
        f"Train Grid Sparsity: {sparsity:.2f}% (0% means perfect regular grid, >0% means missing time-slots per geohash)"
    )

    # ---------------------------------------------------------
    print("\n[4] TARGET ('demand') PROFILING")
    print("-" * 40)
    d_stats = train["demand"].describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99])
    print(d_stats[["min", "1%", "50%", "99%", "max"]].to_string())

    zeros = (train["demand"] == 0).sum()
    print(f"Exact Zeros: {zeros} rows ({(zeros/len(train))*100:.2f}%)")

    # ---------------------------------------------------------
    print("\n[5] FEATURE CONSISTENCY (TRICK IDENTIFIER)")
    print("-" * 40)
    # Do static features change for the same geohash?
    inconsistent_lanes = (df.groupby("geohash")["NumberofLanes"].nunique() > 1).sum()
    inconsistent_road = (df.groupby("geohash")["RoadType"].nunique() > 1).sum()

    print(f"Geohashes with changing NumberofLanes: {inconsistent_lanes}")
    print(f"Geohashes with changing RoadTypes: {inconsistent_road}")

    if inconsistent_lanes > 0 or inconsistent_road > 0:
        print(
            "🚨 WARNING: Static geography features are changing over time. Data may be noisy or moving."
        )

    print("\n" + "=" * 60)
    print("EDA COMPLETE. PLEASE COPY THIS OUTPUT TO YOUR ASSISTANT.")
    print("=" * 60)


if __name__ == "__main__":
    run_eda()
