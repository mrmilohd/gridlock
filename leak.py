import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import warnings

warnings.filterwarnings('ignore')

print("🕵️ Initializing Data Leakage Hunter...\n")

# Load data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Extract time components for both
for df in [train, test]:
    time_split = df['timestamp'].astype(str).str.split(':', expand=True)
    df['hour'] = time_split[0].astype(int)
    df['minute'] = time_split[1].astype(int)

# =========================================================
# TEST 1: The ID Shuffle (Index Leakage)
# =========================================================
print("--- TEST 1: ID Correlation Leakage ---")
# If the Index is highly correlated with demand, the data was sorted before splitting.
id_corr, p_value = spearmanr(train['Index'], train['demand'])

print(f"Spearman Rank Correlation (Index vs Demand): {id_corr:.4f}")
if abs(id_corr) > 0.05:
    print("🚨 FATAL LEAK DETECTED: The 'Index' column contains predictive signal!")
    print("Fix: You must include 'Index' as a feature in your LightGBM model.")
else:
    print("✅ Safe: No obvious ID leakage detected.")
print("")

# =========================================================
# TEST 2: The Groundhog Day Leak (Day Clones)
# =========================================================
print("--- TEST 2: Historical Day Clones ---")
# Check if the total daily demand is identical across any days, implying copy-pasted data
daily_demand = train.groupby('day')['demand'].sum().reset_index()
duplicates = daily_demand[daily_demand.duplicated('demand', keep=False)]

if not duplicates.empty:
    print("🚨 FATAL LEAK DETECTED: Exact daily demand duplicates found!")
    print(duplicates.sort_values('demand'))
    print("Fix: Check if the test set features exactly match these duplicate days.")
else:
    print("✅ Safe: Every day in the training set has a unique total demand.")
print("")

# =========================================================
# TEST 3: The Exact Match Override (Feature Copying)
# =========================================================
print("--- TEST 3: Exact Feature Match Exploitation ---")
# Can we find rows in the test set that are 100% identical to the train set?
# If geohash, hour, minute, weather, and road type are identical, does demand repeat?

# Create a fingerprint for every row
cols_to_match = ['geohash', 'hour', 'minute', 'Weather', 'RoadType', 'LargeVehicles', 'Landmarks']

# Find standard deviations of demand for these exact feature combinations in training
train_fingerprints = train.groupby(cols_to_match)['demand'].agg(['mean', 'std', 'count']).reset_index()

# Find combinations where the demand is EXACTLY the same every single time it happens (std == 0)
perfect_repeats = train_fingerprints[(train_fingerprints['std'] == 0) & (train_fingerprints['count'] > 2)]

print(f"Found {len(perfect_repeats)} unique feature combinations in training data that ALWAYS yield the exact same demand.")

# Merge these "perfect repeating" rules onto the test set
test_merged = test.merge(perfect_repeats, on=cols_to_match, how='left')
exploitable_rows = test_merged['mean'].notna().sum()

print(f"You can explicitly override {exploitable_rows} rows in your test set with 100% historical accuracy.")

if exploitable_rows > 0:
    print("\n🚨 LEAK EXPLOIT GENERATOR:")
    print("To use this, load your 87-scoring submission CSV in Python and do this:")
    print("  sub = pd.read_csv('submission_ultimate.csv')")
    print("  sub.loc[test_merged['mean'].notna(), 'demand'] = test_merged.loc[test_merged['mean'].notna(), 'mean']")
    print("  sub.to_csv('hacked_submission.csv', index=False)")
else:
    print("✅ Safe: No reliable 1:1 feature-to-target mapping found for the test set.")
print("")

# =========================================================
# TEST 4: The Midnight Grid Lock (Zero-Variance Geohashes)
# =========================================================
print("--- TEST 4: Zero-Variance Geohashes ---")
# Do certain geohashes just literally turn off (demand = 0) at night?
zero_demand_geos = train[train['demand'] == 0].groupby(['geohash', 'hour']).size().reset_index(name='zero_count')
total_geos = train.groupby(['geohash', 'hour']).size().reset_index(name='total_count')

geo_stats = zero_demand_geos.merge(total_geos, on=['geohash', 'hour'])
geo_stats['zero_ratio'] = geo_stats['zero_count'] / geo_stats['total_count']

# Find geohashes that are 0.0 demand 100% of the time for a given hour
dead_zones = geo_stats[geo_stats['zero_ratio'] == 1.0]

print(f"Found {len(dead_zones)} Geohash+Hour combinations that NEVER have traffic.")
print("If your model is predicting > 0 for these, you are bleeding points.")
