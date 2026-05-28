import pandas as pd
import numpy as np

# LOAD DATA
df = pd.read_csv('train.csv')
original_row_count = len(df)

# INITIAL COUNTS
roadtype_null_idx    = df[df['RoadType'].isnull()].index.tolist()
temperature_null_idx = df[df['Temperature'].isnull()].index.tolist()
weather_null_idx     = df[df['Weather'].isnull()].index.tolist()

track_records = []
for idx in roadtype_null_idx:
    track_records.append({
        'original_row_index': idx, 'column': 'RoadType',
        'geohash': df.loc[idx, 'geohash'], 'day': df.loc[idx, 'day'], 'timestamp': df.loc[idx, 'timestamp'],
        'filled_value': None, 'rule_used': None, 'fill_successful': False
    })
for idx in temperature_null_idx:
    track_records.append({
        'original_row_index': idx, 'column': 'Temperature',
        'geohash': df.loc[idx, 'geohash'], 'day': df.loc[idx, 'day'], 'timestamp': df.loc[idx, 'timestamp'],
        'filled_value': None, 'rule_used': None, 'fill_successful': False
    })
for idx in weather_null_idx:
    track_records.append({
        'original_row_index': idx, 'column': 'Weather',
        'geohash': df.loc[idx, 'geohash'], 'day': df.loc[idx, 'day'], 'timestamp': df.loc[idx, 'timestamp'],
        'filled_value': None, 'rule_used': None, 'fill_successful': False
    })

tracker = pd.DataFrame(track_records)
print("Total missing values to fill:", len(tracker))
print("RoadType to fill:", len(roadtype_null_idx))
print("Temperature to fill:", len(temperature_null_idx))
print("Weather to fill:", len(weather_null_idx))

# HELPER
hour_series = df['timestamp'].apply(lambda x: int(str(x).split(':')[0]))
df['_hour_temp'] = hour_series

# COLUMN 1 - ROADTYPE
def fill_roadtype(df, tracker):
    def get_mode(x):
        m = x.mode()
        return m.iloc[0] if len(m) > 0 else np.nan

    null_mask = df['RoadType'].isnull()
    
    # RT-1
    fill_rt1 = df.groupby('geohash')['RoadType'].transform(get_mode)
    mask_rt1 = null_mask & fill_rt1.notnull()
    df.loc[mask_rt1, 'RoadType'] = fill_rt1[mask_rt1]
    
    idx_rt1 = df[mask_rt1].index
    tracker.loc[(tracker['column'] == 'RoadType') & (tracker['original_row_index'].isin(idx_rt1)), 'filled_value'] = fill_rt1[mask_rt1].values
    tracker.loc[(tracker['column'] == 'RoadType') & (tracker['original_row_index'].isin(idx_rt1)), 'rule_used'] = "RT-1: geohash mode"
    
    null_mask = df['RoadType'].isnull()
    
    # RT-2
    df['_geo4'] = df['geohash'].str[:4]
    fill_rt2 = df.groupby('_geo4')['RoadType'].transform(get_mode)
    mask_rt2 = null_mask & fill_rt2.notnull()
    df.loc[mask_rt2, 'RoadType'] = fill_rt2[mask_rt2]
    
    idx_rt2 = df[mask_rt2].index
    tracker.loc[(tracker['column'] == 'RoadType') & (tracker['original_row_index'].isin(idx_rt2)), 'filled_value'] = fill_rt2[mask_rt2].values
    tracker.loc[(tracker['column'] == 'RoadType') & (tracker['original_row_index'].isin(idx_rt2)), 'rule_used'] = "RT-2: geohash4 prefix mode"
    df.drop(columns=['_geo4'], inplace=True)
    
    null_mask = df['RoadType'].isnull()
    
    # RT-3
    global_mode = df['RoadType'].mode()[0]
    mask_rt3 = null_mask
    df.loc[mask_rt3, 'RoadType'] = global_mode
    
    idx_rt3 = df[mask_rt3].index
    tracker.loc[(tracker['column'] == 'RoadType') & (tracker['original_row_index'].isin(idx_rt3)), 'filled_value'] = global_mode
    tracker.loc[(tracker['column'] == 'RoadType') & (tracker['original_row_index'].isin(idx_rt3)), 'rule_used'] = "RT-3: global mode fallback"
    
    # Update fill_successful
    rt_mask = (tracker['column'] == 'RoadType') & tracker['filled_value'].notnull()
    tracker.loc[rt_mask, 'fill_successful'] = True
    
    print("RoadType missing remaining:", df['RoadType'].isnull().sum())
    print("\nRoadType fill rule breakdown:")
    rt_tracker = tracker[tracker['column']=='RoadType']
    print(rt_tracker['rule_used'].value_counts())
    
    return df, tracker

# COLUMN 2 - TEMPERATURE
def fill_temperature(df, tracker):
    null_mask = df['Temperature'].isnull()
    
    # TM-1
    grouped_tm1 = df.groupby(['geohash', 'day', '_hour_temp'])['Temperature']
    median_tm1 = grouped_tm1.transform('median')
    count_tm1 = grouped_tm1.transform('count')
    mask_tm1 = null_mask & median_tm1.notnull() & (count_tm1 >= 2)
    df.loc[mask_tm1, 'Temperature'] = median_tm1[mask_tm1]
    
    idx_tm1 = df[mask_tm1].index
    tracker.loc[(tracker['column'] == 'Temperature') & (tracker['original_row_index'].isin(idx_tm1)), 'filled_value'] = median_tm1[mask_tm1].values
    tracker.loc[(tracker['column'] == 'Temperature') & (tracker['original_row_index'].isin(idx_tm1)), 'rule_used'] = "TM-1: geohash+day+hour median"
    
    null_mask = df['Temperature'].isnull()
    
    # TM-2
    median_tm2 = df.groupby(['geohash', '_hour_temp'])['Temperature'].transform('median')
    mask_tm2 = null_mask & median_tm2.notnull()
    df.loc[mask_tm2, 'Temperature'] = median_tm2[mask_tm2]
    
    idx_tm2 = df[mask_tm2].index
    tracker.loc[(tracker['column'] == 'Temperature') & (tracker['original_row_index'].isin(idx_tm2)), 'filled_value'] = median_tm2[mask_tm2].values
    tracker.loc[(tracker['column'] == 'Temperature') & (tracker['original_row_index'].isin(idx_tm2)), 'rule_used'] = "TM-2: geohash+hour median"
    
    null_mask = df['Temperature'].isnull()
    
    # TM-3
    median_tm3 = df.groupby(['geohash'])['Temperature'].transform('median')
    mask_tm3 = null_mask & median_tm3.notnull()
    df.loc[mask_tm3, 'Temperature'] = median_tm3[mask_tm3]
    
    idx_tm3 = df[mask_tm3].index
    tracker.loc[(tracker['column'] == 'Temperature') & (tracker['original_row_index'].isin(idx_tm3)), 'filled_value'] = median_tm3[mask_tm3].values
    tracker.loc[(tracker['column'] == 'Temperature') & (tracker['original_row_index'].isin(idx_tm3)), 'rule_used'] = "TM-3: geohash median"
    
    null_mask = df['Temperature'].isnull()
    
    # TM-4
    median_tm4 = df.groupby(['Weather'])['Temperature'].transform('median')
    mask_tm4 = null_mask & median_tm4.notnull()
    df.loc[mask_tm4, 'Temperature'] = median_tm4[mask_tm4]
    
    idx_tm4 = df[mask_tm4].index
    tracker.loc[(tracker['column'] == 'Temperature') & (tracker['original_row_index'].isin(idx_tm4)), 'filled_value'] = median_tm4[mask_tm4].values
    tracker.loc[(tracker['column'] == 'Temperature') & (tracker['original_row_index'].isin(idx_tm4)), 'rule_used'] = "TM-4: weather median"
    
    null_mask = df['Temperature'].isnull()
    
    # TM-5
    global_median = df['Temperature'].median()
    mask_tm5 = null_mask
    df.loc[mask_tm5, 'Temperature'] = global_median
    
    idx_tm5 = df[mask_tm5].index
    tracker.loc[(tracker['column'] == 'Temperature') & (tracker['original_row_index'].isin(idx_tm5)), 'filled_value'] = global_median
    tracker.loc[(tracker['column'] == 'Temperature') & (tracker['original_row_index'].isin(idx_tm5)), 'rule_used'] = "TM-5: global median fallback"
    
    tm_mask = (tracker['column'] == 'Temperature') & tracker['filled_value'].notnull()
    tracker.loc[tm_mask, 'fill_successful'] = True
    
    print("\nTemperature missing remaining:", df['Temperature'].isnull().sum())
    print("\nTemperature fill rule breakdown:")
    tm_tracker = tracker[tracker['column']=='Temperature']
    print(tm_tracker['rule_used'].value_counts())
    print("\nTemperature stats after fill:")
    print(df['Temperature'].describe())
    
    return df, tracker

# COLUMN 3 - WEATHER
def fill_weather(df, tracker):
    def get_mode(x):
        m = x.mode()
        return m.iloc[0] if len(m) > 0 else np.nan

    null_mask = df['Weather'].isnull()
    
    # WE-1
    fill_we1 = df.groupby(['geohash', 'day'])['Weather'].transform(get_mode)
    mask_we1 = null_mask & fill_we1.notnull()
    df.loc[mask_we1, 'Weather'] = fill_we1[mask_we1]
    
    idx_we1 = df[mask_we1].index
    tracker.loc[(tracker['column'] == 'Weather') & (tracker['original_row_index'].isin(idx_we1)), 'filled_value'] = fill_we1[mask_we1].values
    tracker.loc[(tracker['column'] == 'Weather') & (tracker['original_row_index'].isin(idx_we1)), 'rule_used'] = "WE-1: geohash+day mode"
    
    null_mask = df['Weather'].isnull()
    
    # WE-2
    fill_we2 = df.groupby(['geohash'])['Weather'].transform(get_mode)
    mask_we2 = null_mask & fill_we2.notnull()
    df.loc[mask_we2, 'Weather'] = fill_we2[mask_we2]
    
    idx_we2 = df[mask_we2].index
    tracker.loc[(tracker['column'] == 'Weather') & (tracker['original_row_index'].isin(idx_we2)), 'filled_value'] = fill_we2[mask_we2].values
    tracker.loc[(tracker['column'] == 'Weather') & (tracker['original_row_index'].isin(idx_we2)), 'rule_used'] = "WE-2: geohash mode"
    
    null_mask = df['Weather'].isnull()
    
    # WE-3 (Temperature inference)
    def infer_weather(temp):
        if pd.isna(temp): return np.nan
        if temp < 2: return "Snowy"
        elif temp < 10: return "Rainy"
        elif temp < 20: return "Cloudy"
        else: return "Sunny"
        
    inferred_we3 = df['Temperature'].apply(infer_weather)
    mask_we3 = null_mask & inferred_we3.notnull()
    df.loc[mask_we3, 'Weather'] = inferred_we3[mask_we3]
    
    idx_we3 = df[mask_we3].index
    tracker.loc[(tracker['column'] == 'Weather') & (tracker['original_row_index'].isin(idx_we3)), 'filled_value'] = inferred_we3[mask_we3].values
    tracker.loc[(tracker['column'] == 'Weather') & (tracker['original_row_index'].isin(idx_we3)), 'rule_used'] = "WE-3: temperature inference"
    
    null_mask = df['Weather'].isnull()
    
    # WE-4
    global_mode = df['Weather'].mode()[0]
    mask_we4 = null_mask
    df.loc[mask_we4, 'Weather'] = global_mode
    
    idx_we4 = df[mask_we4].index
    tracker.loc[(tracker['column'] == 'Weather') & (tracker['original_row_index'].isin(idx_we4)), 'filled_value'] = global_mode
    tracker.loc[(tracker['column'] == 'Weather') & (tracker['original_row_index'].isin(idx_we4)), 'rule_used'] = "WE-4: global mode fallback"
    
    we_mask = (tracker['column'] == 'Weather') & tracker['filled_value'].notnull()
    tracker.loc[we_mask, 'fill_successful'] = True
    
    print("\nWeather missing remaining:", df['Weather'].isnull().sum())
    print("\nWeather fill rule breakdown:")
    we_tracker = tracker[tracker['column']=='Weather']
    print(we_tracker['rule_used'].value_counts())
    print("\nWeather value counts after fill:")
    print(df['Weather'].value_counts())
    
    return df, tracker

# EXECUTION
df, tracker = fill_roadtype(df, tracker)
df, tracker = fill_temperature(df, tracker)
df, tracker = fill_weather(df, tracker)

# CLEANUP
df = df.drop(columns=['_hour_temp'])
df = df.drop(columns=['Index'])

# VERIFICATION
print("\n========== FINAL VERIFICATION ==========")
assert len(df) == original_row_count, f"ERROR: Row count is {len(df)}, expected {original_row_count}"
print("Row count check PASSED:", len(df), "rows")

assert df['RoadType'].isnull().sum() == 0, f"ERROR: RoadType still has {df['RoadType'].isnull().sum()} nulls"
assert df['Temperature'].isnull().sum() == 0, f"ERROR: Temperature still has {df['Temperature'].isnull().sum()} nulls"
assert df['Weather'].isnull().sum() == 0, f"ERROR: Weather still has {df['Weather'].isnull().sum()} nulls"
print("Missing value check PASSED: all 0")

assert df['NumberofLanes'].isnull().sum() == 0
assert df['LargeVehicles'].isnull().sum() == 0
assert df['Landmarks'].isnull().sum() == 0
assert df['geohash'].isnull().sum() == 0
assert df['demand'].isnull().sum() == 0
print("Untouched columns check PASSED")

total_filled = tracker['fill_successful'].sum()
total_expected = 600 + 2495 + 797
assert total_filled == total_expected, f"ERROR: Only {total_filled} filled, expected {total_expected}"
print("Tracker completeness check PASSED:", total_filled, "values tracked")

unfilled = tracker[tracker['fill_successful'] == False]
if len(unfilled) > 0:
    print("WARNING: These rows were NOT filled:")
    print(unfilled)
else:
    print("All missing values successfully filled and tracked")

assert 'Index' not in df.columns, "ERROR: Index column still exists"
print("Index column drop check PASSED")

expected_columns = ['geohash','day','timestamp','demand','RoadType','NumberofLanes','LargeVehicles','Landmarks','Temperature','Weather']
actual_columns = df.columns.tolist()
extra_columns = [c for c in actual_columns if c not in expected_columns]
if extra_columns:
    print("WARNING: Extra columns found:", extra_columns)
    print("Dropping extra columns...")
    df = df.drop(columns=extra_columns)
else:
    print("Column check PASSED: no extra columns")

print("Final columns:", df.columns.tolist())
print("Final shape:", df.shape)
print("========== VERIFICATION COMPLETE ==========")

# SAVE
df.to_csv('train_cleaned.csv', index=False)
print("Saved: train_cleaned.csv")
tracker.to_csv('missing_value_tracking.csv', index=False)
print("Saved: missing_value_tracking.csv")

print("\n========== TRACKING SUMMARY ==========")
print("\nRoadType rules used:")
print(tracker[tracker['column']=='RoadType']['rule_used'].value_counts())
print("\nTemperature rules used:")
print(tracker[tracker['column']=='Temperature']['rule_used'].value_counts())
print("\nWeather rules used:")
print(tracker[tracker['column']=='Weather']['rule_used'].value_counts())
print("\nOverall fill success rate:")
print(tracker['fill_successful'].value_counts())
print("========================================")
