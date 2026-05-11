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

    # Job parameters → scheduler constraints
    duration_bins = max(1, int(np.ceil(duration_hr * 60 / BIN_MIN)))     # e.g. 4hr → 16 bins
    job_mw        = gpu_count * (P_MAX - P_IDLE) / 1e6                    # job's max-load power footprint
    if flex_level == 1:
        max_deferral_bins = 0
    elif flex_level == 2:
        max_deferral_bins = 8     # 2 hours
    else:
        max_deferral_bins = 32    # 8 hours (full forecast)

    if flex_level == 1:
        st.info("**Flex 1 job — submit immediately.** Real-time jobs cannot be deferred. There is no benefit to waiting.")
    else:
        # For a job that runs for `duration_bins`, evaluate each possible START bin
        # by the MEAN predicted cluster load over its full runtime → sustained low-load window
        N_FC = len(forecast)
        # Valid starts: 0 .. min(max_deferral, N_FC - 1)
        max_start = min(max_deferral_bins, N_FC - 1)
        candidate_starts = np.arange(0, max_start + 1)

        # Mean predicted flexible_mw over [start, start + duration_bins)
        window_means = []
        for s in candidate_starts:
            end = min(s + duration_bins, N_FC)
            window_means.append(forecast[s:end].mean())
        window_means = np.array(window_means)

        best_step = int(candidate_starts[np.argmin(window_means)])
        best_delay = (best_step + 1) * BIN_MIN      # +1 because step=0 → next bin (15 min from now)
        if best_step == 0:
            best_delay = 0  # submit now

        # Reference: avg load if we started right now
        now_window_mean = forecast[0:min(duration_bins, N_FC)].mean()
        best_window_mean = window_means.min()
        load_reduction = now_window_mean - best_window_mean        # MW saved per bin (averaged over duration)
        energy_shifted_mwh = load_reduction * duration_hr           # MW × hours = MW·h

        # Recommendation banner
        if best_delay == 0:
            st.success(
                f"**Submit now.** Among all deferral options for a {duration_hr:.1f}-hr job "
                f"with {flex_type.split(' — ')[0]} flexibility, the current window has the lowest sustained load."
            )
        else:
            target_hour = (now_hour + best_delay / 60) % 24
            target_hhmm = f"{int(target_hour):02d}:{int((target_hour % 1)*60):02d}"
            st.success(
                f"**Defer by {best_delay} minutes** (start at Day {now_day}, {target_hhmm}). "
                f"This gives the lowest average cluster load over the job's {duration_hr:.1f}-hr runtime, "
                f"within the {max_deferral_bins*BIN_MIN}-minute deferral budget for {flex_type.split(' — ')[0]}."
            )

        # Metric row
        rc1, rc2, rc3, rc4 = st.columns(4)
        rc1.metric("Recommended delay",       f"{best_delay} min")
        rc2.metric("Avg load if submit now",  f"{now_window_mean:.3f} MW")
        rc3.metric("Avg load if deferred",    f"{best_window_mean:.3f} MW", delta=f"{-load_reduction:+.3f} MW")
        rc4.metric("Energy shifted from peak", f"{energy_shifted_mwh:.3f} MW·h")

        rc5, rc6, rc7 = st.columns(3)
        rc5.metric("Job duration",          f"{duration_hr:.1f} hr ({duration_bins} bins)")
        rc6.metric("Job power footprint",   f"{job_mw*1000:.1f} kW @ peak")
        rc7.metric("Max allowed deferral",  f"{max_deferral_bins*BIN_MIN} min")

        # Highlight best window + job span on the forecast plot
        fig2, ax = plt.subplots(figsize=(10, 3.2))
        ax.plot(forecast_hours, forecast, color="steelblue", linewidth=1.2, label="Forecast")
        ax.fill_between(forecast_hours, forecast_lo, forecast_hi, alpha=0.2, color="steelblue")

        # Span of "submit now" scenario (red)
        now_span_end = min(duration_bins, N_FC)
        ax.axvspan(forecast_hours[0] - BIN_MIN/60, forecast_hours[now_span_end - 1],
                   alpha=0.15, color="red", label=f"If submit now (load={now_window_mean:.3f} MW)")

        # Span of "deferred" scenario (green)
        if best_step > 0:
            best_span_end = min(best_step + duration_bins, N_FC)
            ax.axvspan(forecast_hours[best_step], forecast_hours[best_span_end - 1],
                       alpha=0.25, color="green",
                       label=f"If deferred (load={best_window_mean:.3f} MW)")

        ax.set_xlabel("Hours from now")
        ax.set_ylabel("Flexible Capacity (MW)")
        ax.set_title(f"Optimal Window for a {duration_hr:.1f}-hr Job — {flex_type.split(' — ')[0]}")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

        if best_step == 0:
            st.markdown(
                f"> **Why submit now?** No future window in the next {max_deferral_bins*BIN_MIN} min has "
                f"lower sustained load over a {duration_hr:.1f}-hr runtime. "
                f"Your {gpu_count}-GPU job ({job_mw*1000:.1f} kW) won't reduce peak load by waiting."
            )
        else:
            st.markdown(
                f"> **Why this window?** Averaged across the full {duration_hr:.1f}-hr runtime, the cluster's "
                f"predicted load is **{load_reduction:.3f} MW lower** in the deferred window vs starting now. "
                f"For your {gpu_count}-GPU job, this shifts **{energy_shifted_mwh:.3f} MW·h** away from the "
                f"current peak period — exactly the kind of load-shifting that reduces carbon intensity when "
                f"the grid is dirtier at peak hours."
            )

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("Data: Alibaba PAI GPU cluster trace (2020) · Model: Gradient Boosting Regressor · "
           "Synthetic 30-day extension calibrated to real 7-day statistics")
