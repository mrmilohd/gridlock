import pandas as pd
import numpy as np
import geohash2
import warnings
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')

df_test = pd.read_csv('test.csv')

print("Shape:", df_test.shape)
print("Columns:", df_test.columns.tolist())
print("Dtypes:\n", df_test.dtypes)
print("Missing values:\n", df_test.isnull().sum())
print("Day unique values:", df_test['day'].unique())
print("RoadType unique:", df_test['RoadType'].unique())
print("LargeVehicles unique:", df_test['LargeVehicles'].unique())
print("Landmarks unique:", df_test['Landmarks'].unique())
print("Weather unique:", df_test['Weather'].unique())
print("Timestamp sample:", df_test['timestamp'].head(5).tolist())
print("Temperature sample:", df_test['Temperature'].head(5).tolist())

# ─── STEP 0 — SAVE INDEX ────────────────────────────────────────────────────

test_index = df_test['Index'].copy()

assert len(test_index) == len(df_test), \
    "ERROR: Index length does not match dataframe"
assert test_index.isnull().sum() == 0, \
    "ERROR: Index has missing values"

print("Index saved successfully")
print("Index length:", len(test_index))
print("Index sample:", test_index.head(5).tolist())
print("Shape:", df_test.shape)

# ─── STEP 1 — HANDLE MISSING VALUES ─────────────────────────────────────────

# --- Temperature ---

print("Temperature missing BEFORE:", df_test['Temperature'].isnull().sum())

df_test['_hour'] = df_test['timestamp'].apply(
    lambda x: int(str(x).split(':')[0])
)

fill1 = df_test.groupby(['geohash', 'day', '_hour'])['Temperature'].transform(
    lambda x: x.fillna(x.median())
)
df_test['Temperature'] = df_test['Temperature'].fillna(fill1)

fill2 = df_test.groupby(['geohash', '_hour'])['Temperature'].transform(
    lambda x: x.fillna(x.median())
)
df_test['Temperature'] = df_test['Temperature'].fillna(fill2)

fill3 = df_test.groupby('geohash')['Temperature'].transform(
    lambda x: x.fillna(x.median())
)
df_test['Temperature'] = df_test['Temperature'].fillna(fill3)

fill4 = df_test.groupby('Weather')['Temperature'].transform(
    lambda x: x.fillna(x.median())
)
df_test['Temperature'] = df_test['Temperature'].fillna(fill4)

global_temp = df_test['Temperature'].median()
df_test['Temperature'] = df_test['Temperature'].fillna(global_temp)

df_test = df_test.drop(columns=['_hour'])

assert df_test['Temperature'].isnull().sum() == 0, \
    "ERROR: Temperature still has nulls"
print("Temperature missing AFTER:", df_test['Temperature'].isnull().sum())
print("Temperature stats:", df_test['Temperature'].describe().to_dict())

# --- Weather ---

print("Weather missing BEFORE:", df_test['Weather'].isnull().sum())

fill1 = df_test.groupby(['geohash', 'day'])['Weather'].transform(
    lambda x: x.fillna(x.mode()[0] if len(x.mode()) > 0 else np.nan)
)
df_test['Weather'] = df_test['Weather'].fillna(fill1)

fill2 = df_test.groupby('geohash')['Weather'].transform(
    lambda x: x.fillna(x.mode()[0] if len(x.mode()) > 0 else np.nan)
)
df_test['Weather'] = df_test['Weather'].fillna(fill2)

def infer_weather(row):
    if pd.notnull(row['Weather']):
        return row['Weather']
    temp = row['Temperature']
    if temp < 2:    return 'Snowy'
    elif temp < 10: return 'Rainy'
    elif temp < 20: return 'Cloudy'
    else:           return 'Sunny'

still_null = df_test['Weather'].isnull()
if still_null.sum() > 0:
    df_test.loc[still_null, 'Weather'] = df_test[still_null].apply(infer_weather, axis=1)

global_weather = df_test['Weather'].mode()[0]
df_test['Weather'] = df_test['Weather'].fillna(global_weather)

assert df_test['Weather'].isnull().sum() == 0, \
    "ERROR: Weather still has nulls"

known_weather = {'Sunny', 'Rainy', 'Foggy', 'Snowy', 'Cloudy'}
unexpected = set(df_test['Weather'].unique()) - known_weather
if unexpected:
    print("WARNING unexpected weather:", unexpected)
    df_test['Weather'] = df_test['Weather'].apply(
        lambda x: x if x in known_weather else 'Sunny'
    )

print("Weather missing AFTER:", df_test['Weather'].isnull().sum())
print("Weather counts:", df_test['Weather'].value_counts().to_dict())

# --- RoadType ---

print("RoadType missing BEFORE:", df_test['RoadType'].isnull().sum())

fill1 = df_test.groupby('geohash')['RoadType'].transform(
    lambda x: x.fillna(x.mode()[0] if len(x.mode()) > 0 else np.nan)
)
df_test['RoadType'] = df_test['RoadType'].fillna(fill1)

df_test['_geo4'] = df_test['geohash'].str[:4]
fill2 = df_test.groupby('_geo4')['RoadType'].transform(
    lambda x: x.fillna(x.mode()[0] if len(x.mode()) > 0 else np.nan)
)
df_test['RoadType'] = df_test['RoadType'].fillna(fill2)
df_test = df_test.drop(columns=['_geo4'])

global_road = df_test['RoadType'].mode()[0]
df_test['RoadType'] = df_test['RoadType'].fillna(global_road)

assert df_test['RoadType'].isnull().sum() == 0, \
    "ERROR: RoadType still has nulls"

known_roadtypes = {'Residential', 'Street', 'Highway'}
unexpected_rt = set(df_test['RoadType'].unique()) - known_roadtypes
if unexpected_rt:
    print("WARNING unexpected RoadType:", unexpected_rt)
    df_test['RoadType'] = df_test['RoadType'].apply(
        lambda x: x if x in known_roadtypes else 'Residential'
    )

print("RoadType missing AFTER:", df_test['RoadType'].isnull().sum())
print("RoadType counts:", df_test['RoadType'].value_counts().to_dict())

print("\nAll missing values after STEP 1:")
print(df_test.isnull().sum())
assert df_test[['Temperature', 'Weather', 'RoadType']].isnull().sum().sum() == 0, \
    "ERROR: still has missing values after all rules"
print("STEP 1 PASSED — all missing values handled")
print("Shape:", df_test.shape)

# ─── STEP 2 — DECODE GEOHASH TO LAT/LON ─────────────────────────────────────

df_test['latitude']  = df_test['geohash'].apply(lambda x: float(geohash2.decode(x)[0]))
df_test['longitude'] = df_test['geohash'].apply(lambda x: float(geohash2.decode(x)[1]))

assert df_test['latitude'].isnull().sum() == 0, \
    "ERROR: latitude has nulls"
assert df_test['longitude'].isnull().sum() == 0, \
    "ERROR: longitude has nulls"
assert df_test['latitude'].between(-90, 90).all(), \
    "ERROR: latitude out of valid range"
assert df_test['longitude'].between(-180, 180).all(), \
    "ERROR: longitude out of valid range"

print("STEP 2 PASSED — lat/lon decoded")
print("Latitude range:", df_test['latitude'].min(), "to", df_test['latitude'].max())
print("Longitude range:", df_test['longitude'].min(), "to", df_test['longitude'].max())
print("Shape:", df_test.shape)

# ─── STEP 3 — ENCODE LargeVehicles ──────────────────────────────────────────

df_test['LargeVehicles'] = df_test['LargeVehicles'].map({'Allowed': 1, 'Not Allowed': 0})

assert df_test['LargeVehicles'].isnull().sum() == 0, \
    "ERROR: LargeVehicles has nulls after mapping"
assert set(df_test['LargeVehicles'].unique()).issubset({0, 1}), \
    "ERROR: LargeVehicles unexpected values"

print("STEP 3 PASSED — LargeVehicles encoded")
print("LargeVehicles counts:", df_test['LargeVehicles'].value_counts().to_dict())
print("Shape:", df_test.shape)

# ─── STEP 4 — ENCODE Landmarks ───────────────────────────────────────────────

df_test['Landmarks'] = df_test['Landmarks'].map({'Yes': 1, 'No': 0})

assert df_test['Landmarks'].isnull().sum() == 0, \
    "ERROR: Landmarks has nulls after mapping"
assert set(df_test['Landmarks'].unique()).issubset({0, 1}), \
    "ERROR: Landmarks unexpected values"

print("STEP 4 PASSED — Landmarks encoded")
print("Landmarks counts:", df_test['Landmarks'].value_counts().to_dict())
print("Shape:", df_test.shape)

# ─── STEP 5 — ENCODE TIMESTAMP ───────────────────────────────────────────────

df_test['hour']   = df_test['timestamp'].apply(lambda x: int(str(x).split(':')[0]))
df_test['minute'] = df_test['timestamp'].apply(lambda x: int(str(x).split(':')[1]))

assert df_test['hour'].between(0, 23).all(), \
    "ERROR: hour out of 0-23 range"
assert df_test['minute'].isin([0, 15, 30, 45]).all(), \
    "ERROR: minute has values other than 0,15,30,45"
assert df_test['hour'].isnull().sum() == 0, \
    "ERROR: hour has nulls"
assert df_test['minute'].isnull().sum() == 0, \
    "ERROR: minute has nulls"

print("Hour range:", df_test['hour'].min(), "to", df_test['hour'].max())
print("Minute values:", sorted(df_test['minute'].unique()))

df_test['hour_sin'] = np.sin(2 * np.pi * df_test['hour'] / 24)
df_test['hour_cos'] = np.cos(2 * np.pi * df_test['hour'] / 24)
df_test['min_sin']  = np.sin(2 * np.pi * df_test['minute'] / 60)
df_test['min_cos']  = np.cos(2 * np.pi * df_test['minute'] / 60)

assert df_test['hour_sin'].isnull().sum() == 0
assert df_test['hour_cos'].isnull().sum() == 0
assert df_test['min_sin'].isnull().sum() == 0
assert df_test['min_cos'].isnull().sum() == 0

df_test = df_test.drop(columns=['hour', 'minute', 'timestamp'])

assert 'timestamp' not in df_test.columns
assert 'hour'      not in df_test.columns
assert 'minute'    not in df_test.columns
assert 'hour_sin'  in df_test.columns
assert 'hour_cos'  in df_test.columns
assert 'min_sin'   in df_test.columns
assert 'min_cos'   in df_test.columns

print("STEP 5 PASSED — timestamp cyclical encoded")
print("Shape:", df_test.shape)

# ─── STEP 6 — ENCODE DAY TO is_day_49 ────────────────────────────────────────

df_test['is_day_49'] = (df_test['day'] == 49).astype(int)
df_test = df_test.drop(columns=['day'])

assert 'day'       not in df_test.columns, "ERROR: day still exists"
assert 'is_day_49' in df_test.columns,     "ERROR: is_day_49 missing"
assert set(df_test['is_day_49'].unique()).issubset({0, 1}), \
    "ERROR: is_day_49 unexpected values"

print("STEP 6 PASSED — day encoded to is_day_49")
print("is_day_49 counts:", df_test['is_day_49'].value_counts().to_dict())
print("Shape:", df_test.shape)

# ─── STEP 7 — ONE HOT ENCODE RoadType ────────────────────────────────────────

expected_rt_cols = ['RoadType_Residential', 'RoadType_Street', 'RoadType_Highway']

roadtype_dummies = pd.get_dummies(df_test['RoadType'], prefix='RoadType', dtype=int)

for col in expected_rt_cols:
    if col not in roadtype_dummies.columns:
        roadtype_dummies[col] = 0
        print(f"WARNING: {col} missing, added as 0")

roadtype_dummies = roadtype_dummies[expected_rt_cols]

df_test = pd.concat([df_test, roadtype_dummies], axis=1)
df_test = df_test.drop(columns=['RoadType'])

assert 'RoadType' not in df_test.columns
for col in expected_rt_cols:
    assert col in df_test.columns,              f"ERROR: {col} missing"
    assert df_test[col].isnull().sum() == 0,    f"ERROR: {col} has nulls"
    assert set(df_test[col].unique()).issubset({0, 1}), f"ERROR: {col} has non binary values"

rt_row_sum = roadtype_dummies.sum(axis=1)
assert (rt_row_sum == 1).all(), "ERROR: RoadType rows dont sum to 1"

print("STEP 7 PASSED — RoadType one hot encoded")
for col in expected_rt_cols:
    print(f"  {col}:", df_test[col].sum())
print("Shape:", df_test.shape)

# ─── STEP 8 — ONE HOT ENCODE Weather ─────────────────────────────────────────

expected_w_cols = [
    'Weather_Sunny', 'Weather_Rainy', 'Weather_Foggy',
    'Weather_Snowy', 'Weather_Cloudy'
]

weather_dummies = pd.get_dummies(df_test['Weather'], prefix='Weather', dtype=int)

for col in expected_w_cols:
    if col not in weather_dummies.columns:
        weather_dummies[col] = 0
        print(f"WARNING: {col} missing, added as 0")

weather_dummies = weather_dummies[expected_w_cols]

df_test = pd.concat([df_test, weather_dummies], axis=1)
df_test = df_test.drop(columns=['Weather'])

assert 'Weather' not in df_test.columns
for col in expected_w_cols:
    assert col in df_test.columns,              f"ERROR: {col} missing"
    assert df_test[col].isnull().sum() == 0,    f"ERROR: {col} has nulls"
    assert set(df_test[col].unique()).issubset({0, 1}), f"ERROR: {col} has non binary values"

w_row_sum = weather_dummies.sum(axis=1)
assert (w_row_sum == 1).all(), "ERROR: Weather rows dont sum to 1"

print("STEP 8 PASSED — Weather one hot encoded")
for col in expected_w_cols:
    print(f"  {col}:", df_test[col].sum())
print("Shape:", df_test.shape)

# ─── STEP 9 — GEOHASH TARGET ENCODING ────────────────────────────────────────

df_train_raw = pd.read_csv('train.csv')

X_raw = df_train_raw.drop(columns=['demand'])
y_raw = df_train_raw['demand']

X_tr, X_v, y_tr, y_v = train_test_split(
    X_raw, y_raw,
    test_size=0.2,
    random_state=42,
    shuffle=True
)
X_tr = X_tr.reset_index(drop=True)
y_tr = y_tr.reset_index(drop=True)

train_temp = X_tr[['geohash']].copy()
train_temp['demand'] = y_tr

geo_mean   = train_temp.groupby('geohash')['demand'].mean()
geo_median = train_temp.groupby('geohash')['demand'].median()

global_mean   = y_tr.mean()
global_median = y_tr.median()

print("Geohash stats computed on X_train")
print("Unique geohashes in train:", len(geo_mean))
print("Unique geohashes in test:", df_test['geohash'].nunique())

unseen = set(df_test['geohash'].unique()) - set(geo_mean.index)
print("Unseen geohashes in test:", len(unseen))
if len(unseen) > 0:
    print("Will fill with global mean/median")

df_test['geohash_target_mean']   = df_test['geohash'].map(geo_mean)
df_test['geohash_target_median'] = df_test['geohash'].map(geo_median)

df_test['geohash_target_mean']   = df_test['geohash_target_mean'].fillna(global_mean)
df_test['geohash_target_median'] = df_test['geohash_target_median'].fillna(global_median)

assert df_test['geohash_target_mean'].isnull().sum() == 0, \
    "ERROR: geohash_target_mean has nulls"
assert df_test['geohash_target_median'].isnull().sum() == 0, \
    "ERROR: geohash_target_median has nulls"

print("STEP 9 PASSED — geohash target encoded")
print("geo_target_mean stats:", df_test['geohash_target_mean'].describe().to_dict())
print("Shape:", df_test.shape)

# ─── STEP 10 — DROP REMAINING UNUSED COLUMNS ─────────────────────────────────

cols_to_drop = []

if 'geohash' in df_test.columns:
    cols_to_drop.append('geohash')
if 'Index' in df_test.columns:
    cols_to_drop.append('Index')

df_test = df_test.drop(columns=cols_to_drop)

assert 'geohash' not in df_test.columns, "ERROR: geohash still in dataframe"
assert 'Index'   not in df_test.columns, "ERROR: Index still in dataframe"

print("STEP 10 PASSED — unused columns dropped")
print("Dropped:", cols_to_drop)
print("Shape:", df_test.shape)

# ─── STEP 11 — ALIGN TO EXACT FEATURE ORDER ──────────────────────────────────

feature_cols = [
    'NumberofLanes', 'Temperature',
    'latitude', 'longitude',
    'hour_sin', 'hour_cos',
    'min_sin', 'min_cos',
    'geohash_target_mean', 'geohash_target_median',
    'LargeVehicles', 'Landmarks',
    'is_day_49',
    'RoadType_Residential', 'RoadType_Street', 'RoadType_Highway',
    'Weather_Sunny', 'Weather_Rainy', 'Weather_Foggy',
    'Weather_Snowy', 'Weather_Cloudy'
]

for col in feature_cols:
    assert col in df_test.columns, f"ERROR: expected column {col} is missing"

extra = [c for c in df_test.columns if c not in feature_cols]
if extra:
    print("WARNING extra columns found:", extra)
    df_test = df_test.drop(columns=extra)

df_test = df_test[feature_cols]

assert df_test.shape[1] == 21, \
    f"ERROR: expected 21 cols got {df_test.shape[1]}"
assert df_test.columns.tolist() == feature_cols, \
    "ERROR: column order does not match"

print("STEP 11 PASSED — columns aligned")
print("Final columns:", df_test.columns.tolist())
print("Shape:", df_test.shape)

# ─── FINAL VERIFICATION ───────────────────────────────────────────────────────

print("\n========== FINAL VERIFICATION ==========")

print("Final shape:", df_test.shape)
assert df_test.shape[1] == 21, "ERROR: wrong number of columns"

assert df_test.columns.tolist() == feature_cols, "ERROR: column order mismatch"
print("Column order check PASSED")

assert df_test.isnull().sum().sum() == 0, \
    "ERROR: missing values exist in final dataframe"
print("Missing values check PASSED: 0 missing")

obj_cols = [c for c in df_test.columns if df_test[c].dtype == object]
assert len(obj_cols) == 0, f"ERROR: object dtype columns remain: {obj_cols}"
print("Dtype check PASSED: no object columns")

binary_check = [
    'LargeVehicles', 'Landmarks', 'is_day_49',
    'RoadType_Residential', 'RoadType_Street', 'RoadType_Highway',
    'Weather_Sunny', 'Weather_Rainy', 'Weather_Foggy',
    'Weather_Snowy', 'Weather_Cloudy'
]
for col in binary_check:
    assert set(df_test[col].unique()).issubset({0, 1}), \
        f"ERROR: {col} has non binary values"
print("Binary columns check PASSED")

assert len(test_index) == len(df_test), \
    "ERROR: Index length does not match dataframe"
print("Index length check PASSED:", len(test_index), "rows")

print("\nFeature statistics:")
print(df_test.describe().to_string())

print("\n========== ALL CHECKS PASSED ==========")
print("Ready for model prediction")

# ─── SAVE ─────────────────────────────────────────────────────────────────────

df_test.to_csv('test_final.csv', index=False)
test_index.to_csv('test_index.csv', index=False)

print("Saved: test_final.csv")
print("Saved: test_index.csv")
print("Final shape:", df_test.shape)
print("Index shape:", test_index.shape)
