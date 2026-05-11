import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DC Workload Scheduler",
    page_icon="⚡",
    layout="wide",
)

# ── Load assets ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv("data/ts_flex_30day_synthetic.csv")

@st.cache_resource
def load_model():
    gbr         = joblib.load("models/gbr_model.joblib")
    scaler      = joblib.load("models/scaler.joblib")
    feature_cols = joblib.load("models/feature_cols.joblib")
    return gbr, scaler, feature_cols

ts           = load_data()
gbr, scaler, feature_cols = load_model()

P_IDLE, P_MAX, N_GPUS = 60, 300, 6500
HORIZON   = 16   # bins = 4 hours
BIN_MIN   = 15   # minutes per bin

# ── Feature engineering helper ────────────────────────────────────────────────
def build_features(ts_window: pd.DataFrame) -> pd.DataFrame:
    """Given a slice of ts (must have enough history), return feature rows."""
    df = ts_window.copy()
    df["hour_of_day"]        = (df["hour"] % 24).round(2)
    df["day_of_week"]        = (df["hour"] // 24).astype(int) % 7
    df["is_business_hours"]  = ((df["hour_of_day"] >= 9) & (df["hour_of_day"] <= 18)).astype(int)
    df["job_arrival"]        = df["total_jobs"].diff().clip(lower=0).fillna(0)
    df["job_completion"]     = df["total_jobs"].diff().clip(upper=0).abs().fillna(0)
    df["flex3_ratio"]        = df["flex3_jobs"] / df["total_jobs"].replace(0, 1)
    df["flex2_ratio"]        = df["flex2_jobs"] / df["total_jobs"].replace(0, 1)
    for lag in [1, 4, 8, 32, 96]:
        df[f"flexible_mw_lag{lag}"]   = df["flexible_mw"].shift(lag)
        df[f"total_jobs_lag{lag}"]    = df["total_jobs"].shift(lag)
    df["flexible_mw_roll4_mean"]  = df["flexible_mw"].rolling(4).mean()
    df["flexible_mw_roll4_std"]   = df["flexible_mw"].rolling(4).std()
    df["flexible_mw_roll96_mean"] = df["flexible_mw"].rolling(96).mean()
    return df.dropna()

def predict_horizon(now_bin: int, n_steps: int = 32) -> np.ndarray:
    """Predict flexible_mw for the next n_steps bins from now_bin.

    Keeps a raw-column buffer (only original ts columns) and rebuilds
    features fresh each step — avoids passing engineered rows back into
    build_features which breaks lag/rolling computations.
    """
    RAW_COLS = ["time_bin", "hour", "total_jobs", "flex1_jobs",
                "flex2_jobs", "flex3_jobs", "power_mw", "flexible_mw"]

    # Need at least 96 rows of history for lag-96
    history_start = max(0, now_bin - 110)
    raw_buf = ts.iloc[history_start : now_bin + 1][RAW_COLS].copy().reset_index(drop=True)

    preds = []
    for step in range(n_steps):
        feats = build_features(raw_buf)
        if len(feats) == 0:
            preds.append(float(ts["flexible_mw"].iloc[now_bin]))
            continue

        row    = feats.iloc[[-1]][feature_cols]
        scaled = scaler.transform(row)
        pred   = float(gbr.predict(scaled)[0])
        preds.append(pred)

        # Build next raw row — use real ts values if available, else carry forward
        next_bin = now_bin + step + 1
        if next_bin < len(ts):
            next_raw = ts.iloc[[next_bin]][RAW_COLS].copy()
        else:
            next_raw = raw_buf.iloc[[-1]][RAW_COLS].copy()
            next_raw["time_bin"] = next_bin
            next_raw["hour"]     = next_bin * BIN_MIN / 60

        next_raw = next_raw.copy()
        next_raw["flexible_mw"] = pred
        raw_buf = pd.concat([raw_buf, next_raw], ignore_index=True)

    return np.array(preds)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("⚡ DC Workload Scheduler")
st.sidebar.markdown("Carbon-aware job scheduling powered by GBM forecasting.")
st.sidebar.divider()

st.sidebar.subheader("🕐 Simulate Current Time")
max_bin   = len(ts) - HORIZON - 10
now_bin   = st.sidebar.slider(
    "Current time (15-min bins into 30-day window)",
    min_value=120, max_value=max_bin, value=1440, step=1,
)
now_hour  = now_bin * BIN_MIN / 60
now_day   = int(now_hour // 24) + 1
now_hhmm  = f"{int(now_hour % 24):02d}:{int((now_hour % 1)*60):02d}"
st.sidebar.markdown(f"**Day {now_day}, {now_hhmm}**")

st.sidebar.divider()
st.sidebar.subheader("📋 Job Submission")
gpu_count    = st.sidebar.slider("GPU count", 1, 500, 64, step=1)
duration_hr  = st.sidebar.slider("Estimated duration (hours)", 0.5, 48.0, 4.0, step=0.5)
flex_type    = st.sidebar.radio(
    "Job flexibility",
    ["Flex 1 — Real-time (no deferral)", "Flex 2 — Checkpointable (50%)", "Flex 3 — Batch (100% deferrable)"],
    index=1,
)
submit_btn = st.sidebar.button("Find optimal submission window", type="primary")

# ── Main ──────────────────────────────────────────────────────────────────────
st.title("GPU Cluster Carbon-Aware Scheduler")
st.caption("Alibaba PAI trace · 30-day synthetic workload · GBM 4-hour forecasting")

# History (last 24 hr = 96 bins)
hist_start  = max(0, now_bin - 96)
hist_bins   = ts.iloc[hist_start : now_bin + 1]
hist_hours  = hist_bins["hour"].values - now_hour   # relative to now

# Forecast (next 8 hr = 32 bins)
N_FORECAST  = 32
with st.spinner("Running GBM forecast..."):
    forecast = predict_horizon(now_bin, n_steps=N_FORECAST)

# Rolling std as simple uncertainty band
recent_std   = ts["flexible_mw"].iloc[hist_start:now_bin].std()
forecast_lo  = (forecast - 0.8 * recent_std).clip(0)
forecast_hi  = forecast + 0.8 * recent_std
forecast_hours = np.arange(1, N_FORECAST + 1) * BIN_MIN / 60  # positive = future

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7), sharex=False)

# Panel 1: Flexible capacity
ax1.plot(hist_hours, hist_bins["flexible_mw"].values,
         color="black", linewidth=1.0, label="Historical (actual)")
ax1.plot(forecast_hours, forecast,
         color="steelblue", linewidth=1.2, linestyle="--", label="GBM forecast")
ax1.fill_between(forecast_hours, forecast_lo, forecast_hi,
                 alpha=0.25, color="steelblue", label="±0.8σ interval")
ax1.axvline(0, color="red", linewidth=1.2, linestyle=":", label="Now")
ax1.set_ylabel("Flexible Capacity (MW)", fontsize=10)
ax1.set_title(f"Cluster Flexible Capacity — Day {now_day}, {now_hhmm}  (negative x = past, positive = future)",
              fontsize=10)
ax1.legend(fontsize=8, ncol=4)
ax1.grid(alpha=0.3)
ax1.set_xlim(-24, 8)

# Panel 2: Total jobs (cluster load)
ax2.fill_between(hist_hours, hist_bins["total_jobs"].values,
                 alpha=0.4, color="coral", label="Historical jobs")
future_jobs = ts["total_jobs"].iloc[now_bin+1 : now_bin+N_FORECAST+1].values
if len(future_jobs) == N_FORECAST:
    ax2.fill_between(forecast_hours, future_jobs,
                     alpha=0.25, color="steelblue", label="Future jobs (actual synthetic)")
ax2.axvline(0, color="red", linewidth=1.2, linestyle=":")
ax2.set_ylabel("Active Jobs", fontsize=10)
ax2.set_xlabel("Hours relative to now", fontsize=10)
ax2.set_title("Cluster Load", fontsize=10)
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)
ax2.set_xlim(-24, 8)

plt.tight_layout()
st.pyplot(fig)
plt.close()

# ── Scheduling recommendation ─────────────────────────────────────────────────
st.divider()
col1, col2, col3 = st.columns(3)
col1.metric("Current flexible capacity", f"{ts['flexible_mw'].iloc[now_bin]:.3f} MW")
col2.metric("Current active jobs", f"{ts['total_jobs'].iloc[now_bin]:,}")
col3.metric("Forecast horizon", "8 hours")

if submit_btn:
    st.divider()
    st.subheader("📌 Scheduling Recommendation")

    flex_level = int(flex_type[5])  # "Flex 1" → 1, etc.

    if flex_level == 1:
        st.info("**Flex 1 job — submit immediately.** Real-time jobs cannot be deferred. There is no benefit to waiting.")
    else:
        deferral_fraction = 0.5 if flex_level == 2 else 1.0

        # Find the best window in the next 8 hours (lowest predicted load)
        best_step  = int(np.argmin(forecast))
        best_delay = (best_step + 1) * BIN_MIN  # minutes
        best_cap   = forecast[best_step]
        now_cap    = ts["flexible_mw"].iloc[now_bin]
        cap_gain   = best_cap - now_cap

        # Power contribution of this job
        job_mw = gpu_count * (P_MAX - P_IDLE) / 1e6

        if best_delay == BIN_MIN:
            st.success(f"**Submit now.** The current window is already the lowest-load period in the next 8 hours.")
        else:
            st.success(
                f"**Defer by {best_delay} minutes** (submit at "
                f"Day {now_day}, {int((now_hour + best_delay/60) % 24):02d}:{int(((now_hour + best_delay/60) % 1)*60):02d})."
            )

        rc1, rc2, rc3, rc4 = st.columns(4)
        rc1.metric("Recommended delay",    f"{best_delay} min")
        rc2.metric("Capacity at best window", f"{best_cap:.3f} MW")
        rc3.metric("Capacity gain vs now",  f"{cap_gain:+.3f} MW")
        rc4.metric("Job power footprint",   f"{job_mw*1000:.1f} kW")

        # Highlight best window on a mini-plot
        fig2, ax = plt.subplots(figsize=(10, 3))
        ax.plot(forecast_hours, forecast, color="steelblue", linewidth=1.2, label="Forecast")
        ax.fill_between(forecast_hours, forecast_lo, forecast_hi, alpha=0.2, color="steelblue")
        ax.axvline(forecast_hours[best_step], color="green", linewidth=1.5,
                   linestyle="--", label=f"Best window (+{best_delay} min)")
        ax.scatter([forecast_hours[best_step]], [forecast[best_step]],
                   color="green", zorder=5, s=60)
        ax.axhline(now_cap, color="red", linewidth=0.8, linestyle=":", label="Current capacity")
        ax.set_xlabel("Hours from now"); ax.set_ylabel("Flexible Capacity (MW)")
        ax.set_title("Forecast — Optimal Submission Window")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

        st.markdown(
            f"> **Why this window?** GBM predicts flexible capacity peaks at **{best_cap:.3f} MW** "
            f"in {best_delay} minutes — **{abs(cap_gain)*100/max(now_cap,0.001):.1f}% "
            f"{'higher' if cap_gain > 0 else 'lower'}** than now. "
            f"Deferring your {gpu_count}-GPU job ({job_mw*1000:.1f} kW) reduces peak cluster load during the current high-demand window."
        )

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("Data: Alibaba PAI GPU cluster trace (2020) · Model: Gradient Boosting Regressor · "
           "Synthetic 30-day extension calibrated to real 7-day statistics")
