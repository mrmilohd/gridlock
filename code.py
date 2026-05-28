import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LOAD DATA
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("Loading data...")
X_train = pd.read_csv('X_train.csv')
X_val   = pd.read_csv('X_val.csv')
y_train = pd.read_csv('y_train.csv').squeeze()
y_val   = pd.read_csv('y_val.csv').squeeze()

print("X_train shape:", X_train.shape)
print("X_val shape:",   X_val.shape)
print("y_train shape:", y_train.shape)
print("y_val shape:",   y_val.shape)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DEFINE FEATURE COLUMNS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

continuous_cols = [
    'NumberofLanes',
    'Temperature',
    'latitude',
    'longitude',
    'hour_sin',
    'hour_cos',
    'min_sin',
    'min_cos',
    'geohash_target_mean',
    'geohash_target_median'
]

binary_cols = [
    'LargeVehicles',
    'Landmarks',
    'is_day_49',
    'RoadType_Residential',
    'RoadType_Street',
    'RoadType_Highway',
    'Weather_Sunny',
    'Weather_Rainy',
    'Weather_Foggy',
    'Weather_Snowy',
    'Weather_Cloudy'
]

# Drop time_index if exists
if 'time_index' in X_train.columns:
    X_train = X_train.drop(columns=['time_index'])
    X_val   = X_val.drop(columns=['time_index'])
    print("Dropped time_index")

# Verify all columns exist
for col in continuous_cols + binary_cols:
    assert col in X_train.columns, \
        f"ERROR: {col} missing from X_train"
    assert col in X_val.columns, \
        f"ERROR: {col} missing from X_val"

print("All feature columns verified")
print("Total features:", 
      len(continuous_cols) + len(binary_cols))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SCALE CONTINUOUS FEATURES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("\nScaling continuous features...")
scaler = StandardScaler()

X_train[continuous_cols] = scaler.fit_transform(
    X_train[continuous_cols]
)
X_val[continuous_cols] = scaler.transform(
    X_val[continuous_cols]
)
print("Scaling done")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FINAL FEATURE ORDER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

feature_cols = continuous_cols + binary_cols
X_train      = X_train[feature_cols]
X_val        = X_val[feature_cols]

print("Final feature count:", len(feature_cols))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DATASET CLASS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TrafficDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(
            X.values, dtype=torch.float32
        )
        self.y = torch.tensor(
            y.values, dtype=torch.float32
        )

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = TrafficDataset(X_train, y_train)
val_dataset   = TrafficDataset(X_val,   y_val)

train_loader  = DataLoader(
    train_dataset,
    batch_size=512,
    shuffle=True
)
val_loader    = DataLoader(
    val_dataset,
    batch_size=512,
    shuffle=False
)

print("Train batches:", len(train_loader))
print("Val batches:",   len(val_loader))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MLP MODEL
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

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

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SETUP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

device     = torch.device('cpu')
input_size = len(feature_cols)
model      = TrafficMLP(input_size).to(device)

optimizer  = torch.optim.Adam(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-5
)
loss_fn    = nn.MSELoss()

scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',    # maximize R2 score
    factor=0.5,
    patience=5
)

print("\nModel architecture:")
print(model)
total_params = sum(
    p.numel() for p in model.parameters()
)
print(f"Total parameters: {total_params:,}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TRAINING LOOP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

N_EPOCHS       = 50
best_score     = float('-inf')
best_epoch     = 0
patience       = 10
patience_count = 0

print("\n========== TRAINING STARTED ==========")
print(f"{'Epoch':>6} | "
      f"{'Train RMSE':>10} | "
      f"{'Val RMSE':>10} | "
      f"{'Val MAE':>10} | "
      f"{'R2 Score':>10} | "
      f"{'LR':>8}")
print("-" * 70)

for epoch in range(1, N_EPOCHS + 1):

    # ── TRAIN ──
    model.train()
    train_loss = 0.0

    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        pred = model(X_batch)
        loss = loss_fn(pred, y_batch)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)
    train_rmse     = np.sqrt(avg_train_loss)

    # ── VALIDATE ──
    model.eval()
    val_loss       = 0.0
    val_preds_all  = []
    val_target_all = []

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch  = X_batch.to(device)
            y_batch  = y_batch.to(device)
            pred     = model(X_batch)
            loss     = loss_fn(pred, y_batch)
            val_loss += loss.item()

            val_preds_all.append(
                pred.cpu().numpy()
            )
            val_target_all.append(
                y_batch.cpu().numpy()
            )

    avg_val_loss   = val_loss / len(val_loader)
    val_rmse       = np.sqrt(avg_val_loss)
    val_preds_all  = np.concatenate(val_preds_all)
    val_target_all = np.concatenate(val_target_all)

    # ── METRICS ──
    val_mae = np.mean(
        np.abs(val_preds_all - val_target_all)
    )
    r2      = r2_score(val_target_all, val_preds_all)
    score   = max(0, 100 * r2)

    # Scheduler step on R2 score
    scheduler.step(score)
    current_lr = optimizer.param_groups[0]['lr']

    # Print every epoch
    print(f"{epoch:>6} | "
          f"{train_rmse:>10.6f} | "
          f"{val_rmse:>10.6f} | "
          f"{val_mae:>10.6f} | "
          f"{score:>10.4f} | "
          f"{current_lr:>8.6f}")

    # ── SAVE BEST MODEL ──
    if score > best_score:
        best_score     = score
        best_epoch     = epoch
        patience_count = 0
        torch.save(
            model.state_dict(), 
            'best_model.pt'
        )
        print(f"           New best score: {score:.4f}")
    else:
        patience_count += 1

    # ── EARLY STOPPING ──
    if patience_count >= patience:
        print(f"\nEarly stopping at epoch {epoch}")
        print(f"No improvement for {patience} epochs")
        break

print("\n========== TRAINING COMPLETE ==========")
print(f"Best Score: {best_score:.4f} at epoch {best_epoch}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FINAL EVAL WITH BEST MODEL
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("\nLoading best model for final evaluation...")
model.load_state_dict(torch.load('best_model.pt'))
model.eval()

val_preds_final  = []
val_target_final = []

with torch.no_grad():
    for X_batch, y_batch in val_loader:
        X_batch = X_batch.to(device)
        pred    = model(X_batch)
        val_preds_final.append(pred.cpu().numpy())
        val_target_final.append(y_batch.numpy())

val_preds_final  = np.concatenate(val_preds_final)
val_target_final = np.concatenate(val_target_final)

final_r2    = r2_score(val_target_final, val_preds_final)
final_score = max(0, 100 * final_r2)
final_rmse  = np.sqrt(
    np.mean(
        (val_preds_final - val_target_final) ** 2
    )
)
final_mae   = np.mean(
    np.abs(val_preds_final - val_target_final)
)

print("\n========== FINAL SCORES ==========")
print(f"Hackathon Score (R2×100): {final_score:.4f}")
print(f"R2:                       {final_r2:.6f}")
print(f"RMSE:                     {final_rmse:.6f}")
print(f"MAE:                      {final_mae:.6f}")
print("===================================")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SAVE VAL PREDICTIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

val_results = pd.DataFrame({
    'actual':    val_target_final,
    'predicted': val_preds_final,
    'error':     val_target_final - val_preds_final,
    'abs_error': np.abs(
        val_target_final - val_preds_final
    )
})
val_results.to_csv('val_predictions.csv', index=False)
print("\nSaved: val_predictions.csv")
print("\nSample predictions:")
print(val_results.head(10).to_string())