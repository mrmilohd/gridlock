import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

continuous_cols = [
    'NumberofLanes', 'Temperature',
    'latitude', 'longitude',
    'hour_sin', 'hour_cos',
    'min_sin', 'min_cos',
    'geohash_target_mean', 'geohash_target_median'
]

binary_cols = [
    'LargeVehicles', 'Landmarks', 'is_day_49',
    'RoadType_Residential', 'RoadType_Street', 'RoadType_Highway',
    'Weather_Sunny', 'Weather_Rainy', 'Weather_Foggy',
    'Weather_Snowy', 'Weather_Cloudy'
]

feature_cols = continuous_cols + binary_cols

# ── Refit scaler on X_train (same as code.py) ────────────────────────────────
print("Fitting scaler on X_train...")
X_train = pd.read_csv('X_train.csv')
if 'time_index' in X_train.columns:
    X_train = X_train.drop(columns=['time_index'])

scaler = StandardScaler()
scaler.fit(X_train[continuous_cols])
print(f"Scaler fitted on {len(X_train)} training rows")

# ── Load test data ────────────────────────────────────────────────────────────
print("Loading test_final.csv and test_index.csv...")
df_test    = pd.read_csv('test_final.csv')
test_index = pd.read_csv('test_index.csv')

assert df_test.shape[1] == 21,              "ERROR: test_final must have 21 columns"
assert df_test.columns.tolist() == feature_cols, "ERROR: column order mismatch"
assert len(df_test) == len(test_index),     "ERROR: test row count != index row count"

print(f"Test rows: {len(df_test)}")

# ── Scale continuous features ─────────────────────────────────────────────────
df_test[continuous_cols] = scaler.transform(df_test[continuous_cols])
print("Continuous features scaled")

X_test = df_test[feature_cols].values.astype(np.float32)

# ── Model definition (must match code.py exactly) ────────────────────────────
class TrafficMLP(nn.Module):
    def __init__(self, input_size):
        super(TrafficMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.network(x).squeeze(1)

# ── Load model ────────────────────────────────────────────────────────────────
device = torch.device('cpu')
model  = TrafficMLP(input_size=len(feature_cols)).to(device)
model.load_state_dict(torch.load('best_model.pt', map_location=device))
model.eval()
print("Model loaded from best_model.pt")

total_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {total_params:,}")

# ── Inference (batched to be safe with BatchNorm running stats) ───────────────
BATCH_SIZE = 512
all_preds  = []

X_tensor = torch.tensor(X_test, dtype=torch.float32)

with torch.no_grad():
    for i in range(0, len(X_tensor), BATCH_SIZE):
        batch = X_tensor[i : i + BATCH_SIZE].to(device)
        preds = model(batch)
        all_preds.append(preds.cpu().numpy())

predictions = np.concatenate(all_preds)
print(f"Inference complete — {len(predictions)} predictions")

# ── Prediction stats ──────────────────────────────────────────────────────────
print(f"  min:  {predictions.min():.6f}")
print(f"  max:  {predictions.max():.6f}")
print(f"  mean: {predictions.mean():.6f}")
print(f"  std:  {predictions.std():.6f}")
neg_count = (predictions < 0).sum()
if neg_count > 0:
    print(f"  WARNING: {neg_count} negative predictions — clipping to 0")
    predictions = np.clip(predictions, 0, None)

# ── Build submission ──────────────────────────────────────────────────────────
submission = pd.DataFrame({
    'Index':  test_index['Index'].values,
    'demand': predictions
})

assert len(submission) == len(df_test), "ERROR: submission row count mismatch"
assert submission['Index'].isnull().sum() == 0, "ERROR: Index has nulls"
assert submission['demand'].isnull().sum() == 0, "ERROR: demand has nulls"

submission.to_csv('submission.csv', index=False)

print("\nSaved: submission.csv")
print("Shape:", submission.shape)
print("\nSample:")
print(submission.head(10).to_string(index=False))
