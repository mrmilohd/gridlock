import pandas as pd
import numpy as np
import pygeohash as pgh
from sklearn.model_selection import train_test_split

print("1. Loading test.csv...")
test_df = pd.read_csv('test.csv')

print("2. Extracting Index...")
test_df[['Index']].to_csv('test_index.csv', index=False)
indices = test_df['Index'].copy()

print("3. Imputation (TM-1 to 5, WE-1 to 4, RT-1 to 3)...")
def get_mode(x):
    m = x.mode()
    return m.iloc[0] if len(m) > 0 else np.nan

# Helper
test_df['_hour'] = test_df['timestamp'].apply(lambda x: int(str(x).split(':')[0]))
test_df['_geo4'] = test_df['geohash'].str[:4]
test_df['_geo3'] = test_df['geohash'].str[:3]

# RoadType (Mode)
for grp, rule in [('geohash', 'RT-1'), ('_geo4', 'RT-2'), ('_geo3', 'RT-3')]:
    null_mask = test_df['RoadType'].isnull()
    fill_vals = test_df.groupby(grp)['RoadType'].transform(get_mode)
    test_df.loc[null_mask & fill_vals.notnull(), 'RoadType'] = fill_vals[null_mask & fill_vals.notnull()]
test_df['RoadType'] = test_df['RoadType'].fillna(test_df['RoadType'].mode()[0])

# Temperature (Median)
for grp, rule in [(['geohash', '_hour'], 'TM-1'), ('geohash', 'TM-2'), ('_geo4', 'TM-3'), ('_hour', 'TM-4')]:
    null_mask = test_df['Temperature'].isnull()
    fill_vals = test_df.groupby(grp)['Temperature'].transform('median')
    test_df.loc[null_mask & fill_vals.notnull(), 'Temperature'] = fill_vals[null_mask & fill_vals.notnull()]
test_df['Temperature'] = test_df['Temperature'].fillna(test_df['Temperature'].median())

# Weather (Mode)
for grp, rule in [(['geohash', '_hour'], 'WE-1'), ('geohash', 'WE-2'), ('_hour', 'WE-3')]:
    null_mask = test_df['Weather'].isnull()
    fill_vals = test_df.groupby(grp)['Weather'].transform(get_mode)
    test_df.loc[null_mask & fill_vals.notnull(), 'Weather'] = fill_vals[null_mask & fill_vals.notnull()]
test_df['Weather'] = test_df['Weather'].fillna(test_df['Weather'].mode()[0])

test_df.drop(columns=['_hour', '_geo4', '_geo3'], inplace=True)

print("4. Geohash Decoded Lat/Lon...")
test_df['latitude'] = test_df['geohash'].apply(pgh.decode_exact).apply(lambda x: x[0])
test_df['longitude'] = test_df['geohash'].apply(pgh.decode_exact).apply(lambda x: x[1])

print("5. Categorical/Bool Encoding...")
bool_cols = ['LargeVehicles', 'Landmarks']
for b in bool_cols:
    if b in test_df.columns:
        test_df[b] = test_df[b].map({'Yes': 1, 'No': 0}).fillna(0).astype(int)
        
if 'is_day_49' in test_df.columns:
    test_df['is_day_49'] = test_df['is_day_49'].astype(int)
else:
    test_df['is_day_49'] = (test_df['day'] == 49).astype(int)

print("6. Cyclic mapping for timestamp...")
test_df['hour_val'] = test_df['timestamp'].apply(lambda x: int(str(x).split(':')[0]))
test_df['minute_val'] = test_df['timestamp'].apply(lambda x: int(str(x).split(':')[1]))
test_df['time_min'] = test_df['hour_val'] * 60 + test_df['minute_val']
test_df['time_sin'] = np.sin(2 * np.pi * test_df['time_min'] / 1440.0)
test_df['time_cos'] = np.cos(2 * np.pi * test_df['time_min'] / 1440.0)
test_df.drop(columns=['timestamp', 'hour_val', 'minute_val', 'time_min'], inplace=True)

print("7. One Hot Encoding RoadType and Weather...")
rt_categories = ['Arterial', 'Highway', 'Street']
we_categories = ['Clear', 'Rainy', 'Snowy']

for cat in rt_categories:
    if f'RoadType_{cat}' not in test_df.columns:
        test_df[f'RoadType_{cat}'] = (test_df['RoadType'] == cat).astype(int)
for cat in we_categories:
    if f'Weather_{cat}' not in test_df.columns:
        test_df[f'Weather_{cat}'] = (test_df['Weather'] == cat).astype(int)

if 'RoadType' in test_df.columns: test_df.drop(columns=['RoadType'], inplace=True)
if 'Weather' in test_df.columns:  test_df.drop(columns=['Weather'], inplace=True)

print("8. Target Encoding (mean/median). Reloading train.csv...")
train = pd.read_csv('train.csv')
X = train.copy()
y = X.pop('TrafficVolume') if 'TrafficVolume' in X.columns else train['TrafficVolume']

X_tr, _, y_tr, _ = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
enc_df = pd.DataFrame({'geohash': X_tr['geohash'], 'target': y_tr})

print("Computing target encodings...")
stats = enc_df.groupby('geohash')['target'].agg(['mean', 'median'])
stats.columns = ['geohash_target_mean', 'geohash_target_median']

test_df = test_df.merge(stats, on='geohash', how='left')
test_df['geohash_target_mean'] = test_df['geohash_target_mean'].fillna(y_tr.mean())
test_df['geohash_target_median'] = test_df['geohash_target_median'].fillna(y_tr.median())

print("9. Drop geohash and Index, finalize columns...")
if 'geohash' in test_df.columns:
    test_df.drop('geohash', axis=1, inplace=True)
if 'Index' in test_df.columns:
    test_df.drop('Index', axis=1, inplace=True)

print("Final Columns:", test_df.columns.tolist())
print(f"Num Columns: {len(test_df.columns)}")

print("Saving test_final.csv...")
test_df.to_csv('test_final.csv', index=False)
print("Done!")
