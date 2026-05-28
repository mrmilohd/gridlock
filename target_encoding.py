import pandas as pd
import numpy as np
import pygeohash as pgh
from sklearn.model_selection import train_test_split

def main():
    print("Preprocessing test data...")
    df_test = pd.read_csv('test.csv')

    # Imputations since test.csv has nulls
    df_test['Temperature'] = df_test['Temperature'].fillna(16.3803)

    # 1. Decode exactly to append latitude and longitude
    print("Decoding geohashes...")
    unique_geohashes = df_test['geohash'].unique()
    records = []
    for gh in unique_geohashes:
        lat, lon, lat_err, lon_err = pgh.decode_exactly(gh)
        records.append({
            'geohash': gh,
            'latitude': lat,
            'longitude': lon
        })
    mapping_df = pd.DataFrame(records)
    df_test = df_test.merge(mapping_df, on='geohash', how='left')

    # 2. Apply same mappings
    print("Applying mappings...")
    df_test['LargeVehicles'] = df_test['LargeVehicles'].map({'Allowed': 1, 'Not Allowed': 0})
    df_test['Landmarks'] = df_test['Landmarks'].map({'Yes': 1, 'No': 0})

    df_test['hour'] = df_test['timestamp'].apply(lambda x: int(str(x).split(':')[0]))
    df_test['minute'] = df_test['timestamp'].apply(lambda x: int(str(x).split(':')[1]))
    df_test['hour_sin'] = np.sin(2 * np.pi * df_test['hour'] / 24)
    df_test['hour_cos'] = np.cos(2 * np.pi * df_test['hour'] / 24)
    df_test['min_sin']  = np.sin(2 * np.pi * df_test['minute'] / 60)
    df_test['min_cos']  = np.cos(2 * np.pi * df_test['minute'] / 60)
    df_test['time_index'] = df_test['hour'] * 4 + df_test['minute'] // 15
    df_test = df_test.drop(columns=['timestamp'])

    known_roadtypes = {'Residential', 'Street', 'Highway'}
    df_test['RoadType'] = df_test['RoadType'].apply(lambda x: x if x in known_roadtypes else 'Residential')
    roadtype_dummies = pd.get_dummies(df_test['RoadType'], prefix='RoadType', dtype=int)
    expected_rt_cols = ['RoadType_Residential', 'RoadType_Street', 'RoadType_Highway']
    for col in expected_rt_cols:
        if col not in roadtype_dummies.columns:
            roadtype_dummies[col] = 0
    roadtype_dummies = roadtype_dummies[expected_rt_cols]
    df_test = pd.concat([df_test, roadtype_dummies], axis=1)
    df_test = df_test.drop(columns=['RoadType'])

    known_weather = {'Sunny','Rainy','Foggy','Snowy','Cloudy'}
    df_test['Weather'] = df_test['Weather'].apply(lambda x: x if x in known_weather else 'Sunny')
    weather_dummies = pd.get_dummies(df_test['Weather'], prefix='Weather', dtype=int)
    expected_w_cols = ['Weather_Sunny', 'Weather_Rainy', 'Weather_Foggy', 'Weather_Snowy', 'Weather_Cloudy']
    for col in expected_w_cols:
        if col not in weather_dummies.columns:
            weather_dummies[col] = 0
    weather_dummies = weather_dummies[expected_w_cols]
    df_test = pd.concat([df_test, weather_dummies], axis=1)
    df_test = df_test.drop(columns=['Weather'])

    # Drop Index if exists
    if 'Index' in df_test.columns:
        df_test = df_test.drop(columns=['Index'])

    # 3. Name it test_encoded.csv
    df_test.to_csv('test_encoded.csv', index=False)
    print("Saved test_encoded.csv")


    print("Loading encoded training data...")
    df_train = pd.read_csv('train_encoded.csv')

    # Step 1: Convert `day` to `is_day_49`.
    print("Step 1: Convert day to is_day_49")
    df_train['is_day_49'] = (df_train['day'] == 49).astype(int)
    df_train = df_train.drop(columns=['day'])
    df_test['is_day_49'] = (df_test['day'] == 49).astype(int)
    df_test = df_test.drop(columns=['day'])

    # Step 2: Train/Val split (80/20).
    print("Step 2: Train/Val split (80/20)")
    X = df_train.drop(columns=['demand'])
    y = df_train['demand']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 3: Compute geohash mean & median strictly on `X_train`.
    print("Step 3: Compute geohash mean & median strictly on X_train")
    train_tmp = X_train.copy()
    train_tmp['demand'] = y_train
    geohash_stats = train_tmp.groupby('geohash')['demand'].agg(['mean', 'median']).reset_index()
    geohash_stats = geohash_stats.rename(columns={'mean': 'geohash_target_mean', 'median': 'geohash_target_median'})

    # Step 4: Map geohash embeddings to `X_train`, `X_val`, and `df_test`.
    print("Step 4: Map geohash embeddings")
    global_mean = train_tmp['demand'].mean()
    global_median = train_tmp['demand'].median()

    X_train = X_train.merge(geohash_stats, on='geohash', how='left')
    X_train['geohash_target_mean'] = X_train['geohash_target_mean'].fillna(global_mean)
    X_train['geohash_target_median'] = X_train['geohash_target_median'].fillna(global_median)

    X_val = X_val.merge(geohash_stats, on='geohash', how='left')
    X_val['geohash_target_mean'] = X_val['geohash_target_mean'].fillna(global_mean)
    X_val['geohash_target_median'] = X_val['geohash_target_median'].fillna(global_median)

    df_test = df_test.merge(geohash_stats, on='geohash', how='left')
    df_test['geohash_target_mean'] = df_test['geohash_target_mean'].fillna(global_mean)
    df_test['geohash_target_median'] = df_test['geohash_target_median'].fillna(global_median)


    # Step 5: Drop `geohash`.
    print("Step 5: Drop geohash")
    X_train = X_train.drop(columns=['geohash'])
    X_val = X_val.drop(columns=['geohash'])
    df_test = df_test.drop(columns=['geohash'])

    # Asserts
    print("Running asserts...")
    assert 'geohash' not in X_train.columns
    assert 'geohash' not in X_val.columns
    assert 'geohash' not in df_test.columns
    assert 'day' not in X_train.columns
    assert 'is_day_49' in X_train.columns
    assert 'geohash_target_mean' in X_train.columns
    assert 'geohash_target_median' in X_train.columns
    assert X_train.isnull().sum().sum() == 0, "Nulls found in X_train"
    assert X_val.isnull().sum().sum() == 0, "Nulls found in X_val"
    assert df_test.isnull().sum().sum() == 0, "Nulls found in df_test"

    # Export `X_train.csv`, `X_val.csv`, `y_train.csv`, `y_val.csv`, and `test_final.csv`.
    print("Exporting...")
    X_train.to_csv('X_train.csv', index=False)
    X_val.to_csv('X_val.csv', index=False)
    y_train.to_csv('y_train.csv', index=False)
    y_val.to_csv('y_val.csv', index=False)
    df_test.to_csv('test_final.csv', index=False)
    print("Done.")

if __name__ == '__main__':
    main()
