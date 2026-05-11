"""
Train LSTM v1, LSTM v2, TCN, and Quantile LSTM on the 30-day synthetic
dataset; save test-set predictions to data/model_predictions.csv so the
app's Methodology page can overlay them alongside the live
Persistence / Linear / GBM forecasts.

Hours column is aligned with the test split used by Methodology's
_compute_test_predictions, so a join on (bin index in test window) works.

Run: cd dc-scheduler-app && python3 scripts/generate_model_predictions.py
"""
import warnings; warnings.filterwarnings("ignore", category=UserWarning)
import os, sys, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

torch.manual_seed(42); np.random.seed(42)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")

# ─── Replicate the training pipeline exactly ─────────────────────────────────
HORIZON  = 16    # 4 hours ahead
LOOKBACK = 96    # 24 hours of history
BATCH    = 32
LR       = 1e-3

print("Loading synthetic data…", flush=True)
ts = pd.read_csv(os.path.join(DATA, "ts_flex_30day_synthetic.csv"))

df = ts.copy()
df["hour_of_day"]       = (df["hour"] % 24).round(2)
df["day_of_week"]       = (df["hour"] // 24).astype(int) % 7
df["is_business_hours"] = ((df["hour_of_day"] >= 9) & (df["hour_of_day"] <= 18)).astype(int)
df["job_arrival"]       = df["total_jobs"].diff().clip(lower=0).fillna(0)
df["job_completion"]    = df["total_jobs"].diff().clip(upper=0).abs().fillna(0)
df["flex3_ratio"]       = df["flex3_jobs"] / df["total_jobs"].replace(0, 1)
df["flex2_ratio"]       = df["flex2_jobs"] / df["total_jobs"].replace(0, 1)
for lag in [1, 4, 8, 32, 96]:
    df[f"flexible_mw_lag{lag}"] = df["flexible_mw"].shift(lag)
    df[f"total_jobs_lag{lag}"]  = df["total_jobs"].shift(lag)
df["flexible_mw_roll4_mean"]  = df["flexible_mw"].rolling(4).mean()
df["flexible_mw_roll4_std"]   = df["flexible_mw"].rolling(4).std()
df["flexible_mw_roll96_mean"] = df["flexible_mw"].rolling(96).mean()
df["target"] = df["flexible_mw"].shift(-HORIZON)
df = df.dropna().reset_index(drop=True)

drop_cols = ["target", "time_bin", "hour", "power_mw",
             "flex1_jobs", "flex2_jobs", "flex3_jobs", "flexible_mw"]
feature_cols = [c for c in df.columns if c not in drop_cols]

X = df[feature_cols].values
y = df["target"].values
split = int(len(df) * 0.8)

# Scaler fit on train only — same as the deployed GBM
scaler = StandardScaler().fit(X[:split])
X_all_sc = scaler.transform(X)  # apply to full series for sequence lookback
y_train, y_test = y[:split], y[split:]
n_features = X.shape[1]

# Build sequences. For each test bin t (absolute index `split + t`), the input
# sequence is X_all_sc[split + t - LOOKBACK : split + t]. This uses the tail
# of training data as lookback for the early test bins, so we have predictions
# for ALL 554 test bins (no padding gap).
def make_sequences(absolute_indices):
    Xs = []
    for ai in absolute_indices:
        Xs.append(X_all_sc[ai - LOOKBACK : ai])
    return np.array(Xs)

# Train sequences: from LOOKBACK to split (excludes the very early train bins
# that don't have enough lookback)
train_abs = np.arange(LOOKBACK, split)
X_seq_train = make_sequences(train_abs)
y_seq_train = y[train_abs]

# Test sequences: every test bin gets a prediction
test_abs = np.arange(split, len(df))
X_seq_test = make_sequences(test_abs)
y_seq_test = y[test_abs]

print(f"Train sequences: {X_seq_train.shape}", flush=True)
print(f"Test sequences:  {X_seq_test.shape}", flush=True)

X_tr = torch.FloatTensor(X_seq_train)
y_tr = torch.FloatTensor(y_seq_train)
X_te = torch.FloatTensor(X_seq_test)
y_te = torch.FloatTensor(y_seq_test)
loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=BATCH, shuffle=False)

# ─── Model definitions ────────────────────────────────────────────────────────
class LSTMForecaster(nn.Module):
    def __init__(self, input_size, hidden=64, layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, layers, batch_first=True, dropout=dropout)
        self.fc   = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, x):
        out, _ = self.lstm(x); return self.fc(out[:, -1, :]).squeeze(-1)

class TCNBlock(nn.Module):
    def __init__(self, in_c, out_c, ks, dil, dropout=0.2):
        super().__init__()
        pad = (ks - 1) * dil
        self.c1 = nn.Conv1d(in_c, out_c, ks, dilation=dil, padding=pad)
        self.c2 = nn.Conv1d(out_c, out_c, ks, dilation=dil, padding=pad)
        self.relu = nn.ReLU(); self.drop = nn.Dropout(dropout)
        self.ds   = nn.Conv1d(in_c, out_c, 1) if in_c != out_c else None
    def forward(self, x):
        out = self.drop(self.relu(self.c1(x)[:, :, :x.size(2)]))
        out = self.drop(self.relu(self.c2(out)[:, :, :x.size(2)]))
        return self.relu(out + (x if self.ds is None else self.ds(x)))

class TCNForecaster(nn.Module):
    def __init__(self, input_size, channels=(64, 64, 32), ks=3, dropout=0.2):
        super().__init__()
        layers = []; in_c = input_size
        for i, out_c in enumerate(channels):
            layers.append(TCNBlock(in_c, out_c, ks, 2**i, dropout)); in_c = out_c
        self.tcn = nn.Sequential(*layers); self.fc = nn.Linear(channels[-1], 1)
    def forward(self, x):
        x = x.permute(0, 2, 1); out = self.tcn(x)
        return self.fc(out[:, :, -1]).squeeze(-1)

class QuantileLSTM(nn.Module):
    def __init__(self, input_size, hidden=64, layers=2, dropout=0.2, q=(0.1, 0.5, 0.9)):
        super().__init__()
        self.q = q
        self.lstm = nn.LSTM(input_size, hidden, layers, batch_first=True, dropout=dropout)
        self.fc   = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, len(q)))
    def forward(self, x):
        out, _ = self.lstm(x); return self.fc(out[:, -1, :])

def pinball(preds, target, qs):
    losses = []
    for i, qv in enumerate(qs):
        err = target - preds[:, i]
        losses.append(torch.max(qv * err, (qv - 1) * err))
    return torch.mean(torch.stack(losses))

def evaluate(name, y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-6))) * 100
    print(f"  {name:30s} MAE: {mae:.4f}  RMSE: {rmse:.4f}  MAPE: {mape:.1f}%", flush=True)
    return mae

# ─── Train each model ─────────────────────────────────────────────────────────
criterion = nn.MSELoss()
results = {}

def train_simple(model, epochs, name, scheduler_on=False):
    t0 = time.time()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5) if scheduler_on else None
    for ep in range(epochs):
        model.train(); tot = 0
        for xb, yb in loader:
            opt.zero_grad(); pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward(); opt.step(); tot += loss.item()
        avg = tot / len(loader)
        if sched is not None:
            sched.step(avg)
        if (ep + 1) % 10 == 0:
            print(f"    [{name}] epoch {ep+1}/{epochs}  loss {avg:.6f}", flush=True)
    model.eval()
    with torch.no_grad():
        pred = model(X_te).numpy()
    print(f"  {name} trained in {(time.time()-t0)/60:.1f} min", flush=True)
    return pred

# LSTM v1 — 50 epochs, no scheduler
print("\n=== LSTM v1 (50 epochs) ===", flush=True)
m1 = LSTMForecaster(n_features); torch.manual_seed(42)
results["lstm_v1"] = train_simple(m1, 50, "LSTM v1", scheduler_on=False)
evaluate("LSTM v1", y_seq_test, results["lstm_v1"])

# LSTM v2 — 100 epochs + LR scheduler
print("\n=== LSTM v2 (100 epochs + LR scheduler) ===", flush=True)
m2 = LSTMForecaster(n_features); torch.manual_seed(43)
results["lstm_v2"] = train_simple(m2, 100, "LSTM v2", scheduler_on=True)
evaluate("LSTM v2", y_seq_test, results["lstm_v2"])

# TCN — 100 epochs + LR scheduler
print("\n=== TCN (100 epochs) ===", flush=True)
m3 = TCNForecaster(n_features); torch.manual_seed(44)
results["tcn"] = train_simple(m3, 100, "TCN", scheduler_on=True)
evaluate("TCN", y_seq_test, results["tcn"])

# Quantile LSTM — 100 epochs + pinball loss
print("\n=== Quantile LSTM (100 epochs) ===", flush=True)
torch.manual_seed(45)
mq = QuantileLSTM(n_features)
optq = torch.optim.Adam(mq.parameters(), lr=LR)
schedq = torch.optim.lr_scheduler.ReduceLROnPlateau(optq, patience=5, factor=0.5)
t0 = time.time()
for ep in range(100):
    mq.train(); tot = 0
    for xb, yb in loader:
        optq.zero_grad(); preds = mq(xb)
        loss = pinball(preds, yb, [0.1, 0.5, 0.9])
        loss.backward(); optq.step(); tot += loss.item()
    avg = tot / len(loader); schedq.step(avg)
    if (ep + 1) % 10 == 0:
        print(f"    [QLSTM] epoch {ep+1}/100  pinball {avg:.6f}", flush=True)
mq.eval()
with torch.no_grad():
    qpreds = mq(X_te).numpy()
print(f"  QLSTM trained in {(time.time()-t0)/60:.1f} min", flush=True)
results["qlstm_q10"] = qpreds[:, 0]
results["qlstm_q50"] = qpreds[:, 1]
results["qlstm_q90"] = qpreds[:, 2]
evaluate("Quantile LSTM (q50)", y_seq_test, results["qlstm_q50"])
coverage = np.mean((y_seq_test >= results["qlstm_q10"]) & (y_seq_test <= results["qlstm_q90"])) * 100
print(f"  Coverage [q10, q90]: {coverage:.1f}%  (target 80%)", flush=True)

# ─── Save predictions CSV aligned with Methodology's test window ─────────────
hours_in_test = df["hour"].values[test_abs] - df["hour"].values[test_abs[0]]
out = pd.DataFrame({
    "bin_offset_in_test": np.arange(len(test_abs)),
    "hours":              hours_in_test,
    "actual":             y_seq_test,
    "lstm_v1":            results["lstm_v1"],
    "lstm_v2":            results["lstm_v2"],
    "tcn":                results["tcn"],
    "qlstm_q10":          results["qlstm_q10"],
    "qlstm_q50":          results["qlstm_q50"],
    "qlstm_q90":          results["qlstm_q90"],
})
out_path = os.path.join(DATA, "model_predictions.csv")
out.to_csv(out_path, index=False)
print(f"\nSaved {len(out)} rows to {out_path}", flush=True)
print("DONE", flush=True)
