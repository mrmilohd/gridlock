import pandas as pd
import numpy as np

def main():
    # Load dataset
    print("Loading data...")
    df = pd.read_csv('train_expanded_geohash.csv')

    # Get original row count
    original_row_count = len(df)

    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("PRE CHECK:")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("Shape before encoding:", df.shape)
    print("Dtypes:\n", df.dtypes)
    print("Missing values:\n", df.isnull().sum())
    assert df.isnull().sum().sum() == 0, "ERROR: missing values exist, handle them first"
    print("PRE CHECK PASSED")

    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("STEP 1 — LargeVehicles (binary):")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    df['LargeVehicles'] = df['LargeVehicles'].map({
        'Allowed'    : 1,
        'Not Allowed': 0
    })

    assert df['LargeVehicles'].isnull().sum() == 0, "ERROR: LargeVehicles has nulls after mapping"
    assert set(df['LargeVehicles'].unique()) == {0, 1}, "ERROR: LargeVehicles has unexpected values"
    print("STEP 1 PASSED — LargeVehicles")
    print("Value counts:", df['LargeVehicles'].value_counts().to_dict())
    print("Shape:", df.shape)
    assert len(df) == original_row_count

    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("STEP 2 — Landmarks (binary):")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    df['Landmarks'] = df['Landmarks'].map({
        'Yes': 1,
        'No' : 0
    })

    assert df['Landmarks'].isnull().sum() == 0, "ERROR: Landmarks has nulls after mapping"
    assert set(df['Landmarks'].unique()) == {0, 1}, "ERROR: Landmarks has unexpected values"
    print("STEP 2 PASSED — Landmarks")
    print("Value counts:", df['Landmarks'].value_counts().to_dict())
    print("Shape:", df.shape)
    assert len(df) == original_row_count

    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("STEP 3 — timestamp (cyclical encoding):")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    df['hour']   = df['timestamp'].apply(lambda x: int(str(x).split(':')[0]))
    df['minute'] = df['timestamp'].apply(lambda x: int(str(x).split(':')[1]))

    assert df['hour'].min() >= 0
    assert df['hour'].max() <= 23
    assert df['minute'].min() >= 0
    assert df['minute'].max() <= 45
    print("Hour range:", df['hour'].min(), "to", df['hour'].max())
    print("Minute values:", sorted(df['minute'].unique()))

    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['min_sin']  = np.sin(2 * np.pi * df['minute'] / 60)
    df['min_cos']  = np.cos(2 * np.pi * df['minute'] / 60)

    df['time_index'] = df['hour'] * 4 + df['minute'] // 15

    assert df['time_index'].min() == 0
    assert df['time_index'].max() == 95
    assert df['time_index'].nunique() == 96

    df = df.drop(columns=['timestamp'])

    assert 'timestamp' not in df.columns
    assert 'hour' in df.columns
    assert 'hour_sin' in df.columns
    assert 'hour_cos' in df.columns
    assert 'min_sin' in df.columns
    assert 'min_cos' in df.columns
    assert 'time_index' in df.columns
    print("STEP 3 PASSED — timestamp cyclical encoding")
    print("Shape:", df.shape)
    assert len(df) == original_row_count

    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("STEP 4 — RoadType (one hot, 3 classes):")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    known_roadtypes = {'Residential', 'Street', 'Highway'}
    actual_roadtypes = set(df['RoadType'].unique())
    unexpected_rt = actual_roadtypes - known_roadtypes
    if unexpected_rt:
        print("WARNING unexpected RoadType:", unexpected_rt)
        df['RoadType'] = df['RoadType'].apply(
            lambda x: x if x in known_roadtypes 
            else 'Residential'
        )

    roadtype_dummies = pd.get_dummies(df['RoadType'], prefix='RoadType', dtype=int)

    expected_rt_cols = ['RoadType_Residential', 'RoadType_Street', 'RoadType_Highway']
    for col in expected_rt_cols:
        if col not in roadtype_dummies.columns:
            roadtype_dummies[col] = 0
            print(f"WARNING: {col} was missing, added as 0")

    roadtype_dummies = roadtype_dummies[expected_rt_cols]

    df = pd.concat([df, roadtype_dummies], axis=1)
    df = df.drop(columns=['RoadType'])

    assert 'RoadType' not in df.columns
    for col in expected_rt_cols:
        assert col in df.columns, f"ERROR: {col} missing"
        assert df[col].isnull().sum() == 0
        assert set(df[col].unique()).issubset({0,1})

    rt_sum = roadtype_dummies.sum(axis=1)
    assert (rt_sum == 1).all(), "ERROR: RoadType one hot rows dont sum to 1"

    print("STEP 4 PASSED — RoadType one hot")
    print("RoadType column counts:")
    for col in expected_rt_cols:
        print(f"  {col}:", df[col].sum())
    print("Shape:", df.shape)
    assert len(df) == original_row_count

    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("STEP 5 — Weather (one hot, 5 classes):")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    known_weather = {'Sunny','Rainy','Foggy','Snowy','Cloudy'}
    actual_weather = set(df['Weather'].unique())
    unexpected_w = actual_weather - known_weather
    if unexpected_w:
        print("WARNING unexpected Weather:", unexpected_w)
        df['Weather'] = df['Weather'].apply(
            lambda x: x if x in known_weather else 'Sunny'
        )

    weather_dummies = pd.get_dummies(df['Weather'], prefix='Weather', dtype=int)

    expected_w_cols = [
        'Weather_Sunny', 'Weather_Rainy', 'Weather_Foggy',
        'Weather_Snowy', 'Weather_Cloudy'
    ]
    for col in expected_w_cols:
        if col not in weather_dummies.columns:
            weather_dummies[col] = 0
            print(f"WARNING: {col} was missing, added as 0")

    weather_dummies = weather_dummies[expected_w_cols]

    df = pd.concat([df, weather_dummies], axis=1)
    df = df.drop(columns=['Weather'])

    assert 'Weather' not in df.columns
    for col in expected_w_cols:
        assert col in df.columns, f"ERROR: {col} missing"
        assert df[col].isnull().sum() == 0
        assert set(df[col].unique()).issubset({0,1})

    w_sum = weather_dummies.sum(axis=1)
    assert (w_sum == 1).all(), "ERROR: Weather one hot rows dont sum to 1"

    print("STEP 5 PASSED — Weather one hot")
    print("Weather column counts:")
    for col in expected_w_cols:
        print(f"  {col}:", df[col].sum())
    print("Shape:", df.shape)
    assert len(df) == original_row_count

    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("FINAL VERIFICATION:")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    expected_final_cols = [
        'geohash', 'day', 'demand',
        'NumberofLanes', 'Temperature',
        'latitude', 'longitude',
        'LargeVehicles', 'Landmarks',
        'hour', 'minute',
        'hour_sin', 'hour_cos',
        'min_sin', 'min_cos',
        'time_index',
        'RoadType_Residential', 'RoadType_Street', 'RoadType_Highway',
        'Weather_Sunny', 'Weather_Rainy',
        'Weather_Foggy', 'Weather_Snowy', 'Weather_Cloudy'
    ]

    for col in expected_final_cols:
        assert col in df.columns, f"ERROR: expected column {col} is missing"

    extra_cols = [c for c in df.columns if c not in expected_final_cols]
    if extra_cols:
        print("WARNING extra columns found:", extra_cols)
        print("Dropping extra columns...")
        df = df.drop(columns=extra_cols)

    object_cols = [c for c in df.columns if df[c].dtype == 'object' and c != 'geohash']
    assert len(object_cols) == 0, f"ERROR: object dtype columns remain: {object_cols}"

    assert df.isnull().sum().sum() == 0, "ERROR: missing values exist in final dataframe"
    assert len(df) == original_row_count, f"ERROR: row count changed from {original_row_count} to {len(df)}"

    print("Column count:", len(df.columns))
    print("All columns:", df.columns.tolist())
    print("All dtypes:\n", df.dtypes)
    print("Missing values:", df.isnull().sum().sum())
    print("Final shape:", df.shape)
    print("========== VERIFICATION PASSED ==========")

    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("SAVE:")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    df.to_csv('train_encoded.csv', index=False)
    print("Saved: train_encoded.csv")
    print("Final shape:", df.shape)

if __name__ == "__main__":
    main()