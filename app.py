import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import io
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DC Workload Scheduler",
    page_icon="⚡",
    layout="wide",
)

# ── Load assets ───────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    ts = pd.read_csv("data/ts_flex_30day_synthetic.csv")
    lmp = pd.read_csv("data/ercot_lmp_30day.csv")
    return ts, lmp

@st.cache_resource
def load_model():
    gbr          = joblib.load("models/gbr_model.joblib")
    scaler       = joblib.load("models/scaler.joblib")
    feature_cols = joblib.load("models/feature_cols.joblib")
    return gbr, scaler, feature_cols

ts, lmp_df = load_data()
gbr, scaler, feature_cols = load_model()

P_IDLE, P_MAX, N_GPUS = 60, 300, 6500
HORIZON   = 16   # bins = 4 hours
BIN_MIN   = 15   # minutes per bin

# ── Feature engineering ───────────────────────────────────────────────────────
def build_features(ts_window: pd.DataFrame) -> pd.DataFrame:
    df = ts_window.copy()
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
    return df.dropna()

def predict_horizon(now_bin: int, n_steps: int = 32) -> np.ndarray:
    RAW_COLS = ["time_bin", "hour", "total_jobs", "flex1_jobs",
                "flex2_jobs", "flex3_jobs", "power_mw", "flexible_mw"]
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

def get_lmp_slice(now_bin: int, n_steps: int = 32):
    """Get LMP for next n_steps bins. In production this would be ERCOT forecast,
    here we use the actual LMP from the bundled synthetic-but-calibrated dataset."""
    end_bin = min(now_bin + n_steps + 1, len(lmp_df))
    slice_ = lmp_df.iloc[now_bin + 1 : end_bin]
    return slice_["lmp_usd_mwh"].values, slice_["is_4cp_bin"].values.astype(bool)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("⚡ DC Workload Scheduler")
st.sidebar.markdown("Carbon-aware + cost-aware scheduling for GPU clusters.\n\nLMP source: ERCOT-calibrated synthetic profile.")
st.sidebar.divider()

mode = st.sidebar.radio(
    "Mode",
    ["🧑‍🔬 Advisor (single job)", "🏢 Planner (job queue)"],
    index=0,
)
st.sidebar.divider()

st.sidebar.subheader("🕐 Simulated Current Time")
max_bin   = len(ts) - HORIZON - 10
now_bin   = st.sidebar.slider(
    "Bin index (15-min steps into 30-day window)",
    min_value=120, max_value=max_bin, value=1440, step=1,
)
now_hour  = now_bin * BIN_MIN / 60
now_day   = int(now_hour // 24) + 1
now_hhmm  = f"{int(now_hour % 24):02d}:{int((now_hour % 1)*60):02d}"
st.sidebar.markdown(f"**Day {now_day}, {now_hhmm}**")
st.sidebar.metric("Current LMP", f"${lmp_df['lmp_usd_mwh'].iloc[now_bin]:.2f}/MWh")
if lmp_df["is_4cp_bin"].iloc[now_bin]:
    st.sidebar.error("🚨 4CP candidate window — avoid loading cluster")

# ── Header ────────────────────────────────────────────────────────────────────
st.title("GPU Cluster Carbon- & Cost-Aware Scheduler")
st.caption(
    "Alibaba PAI workload trace · 30-day synthetic data · GBM 8-hour forecasting · "
    "ERCOT-calibrated LMP · 4CP-aware"
)

# Forecast (shared by both modes)
N_FORECAST = 32
with st.spinner("Running GBM forecast..."):
    forecast = predict_horizon(now_bin, n_steps=N_FORECAST)
fc_lmp, fc_is_4cp = get_lmp_slice(now_bin, N_FORECAST)
# Pad if at end of data
if len(fc_lmp) < N_FORECAST:
    fc_lmp = np.pad(fc_lmp, (0, N_FORECAST - len(fc_lmp)), constant_values=fc_lmp[-1] if len(fc_lmp) else 30)
    fc_is_4cp = np.pad(fc_is_4cp, (0, N_FORECAST - len(fc_is_4cp)), constant_values=False)
forecast_hours = np.arange(1, N_FORECAST + 1) * BIN_MIN / 60

# Rolling std uncertainty
hist_start = max(0, now_bin - 96)
recent_std = ts["flexible_mw"].iloc[hist_start:now_bin].std()
forecast_lo = (forecast - 0.8 * recent_std).clip(0)
forecast_hi = forecast + 0.8 * recent_std

# ── Shared: history + forecast overview ──────────────────────────────────────
hist_bins  = ts.iloc[hist_start : now_bin + 1]
hist_hours = hist_bins["hour"].values - now_hour
hist_lmp   = lmp_df["lmp_usd_mwh"].iloc[hist_start : now_bin + 1].values

fig, axes = plt.subplots(2, 1, figsize=(13, 6.5), sharex=True)

axes[0].plot(hist_hours, hist_bins["flexible_mw"].values,
             color="black", linewidth=1.0, label="Historical flex capacity")
axes[0].plot(forecast_hours, forecast,
             color="steelblue", linewidth=1.2, linestyle="--", label="GBM forecast")
axes[0].fill_between(forecast_hours, forecast_lo, forecast_hi,
                     alpha=0.25, color="steelblue", label="±0.8σ")
axes[0].axvline(0, color="red", linewidth=1.2, linestyle=":", label="Now")
axes[0].set_ylabel("Flex Capacity (MW)", fontsize=10)
axes[0].set_title(f"Cluster State — Day {now_day}, {now_hhmm}", fontsize=10)
axes[0].legend(fontsize=8, ncol=4)
axes[0].grid(alpha=0.3)

# LMP panel
axes[1].plot(hist_hours, hist_lmp, color="darkorange", linewidth=1.0, label="Historical LMP")
axes[1].plot(forecast_hours, fc_lmp, color="darkorange", linewidth=1.2, linestyle="--", label="LMP (next 8h)")
axes[1].axvline(0, color="red", linewidth=1.2, linestyle=":")
# Mark 4CP bins in red
for i, is_4cp in enumerate(fc_is_4cp):
    if is_4cp:
        axes[1].axvspan(forecast_hours[i] - BIN_MIN/120, forecast_hours[i] + BIN_MIN/120,
                        color="red", alpha=0.4, label="4CP candidate" if i == np.argmax(fc_is_4cp) else None)
axes[1].set_ylabel("LMP ($/MWh)", fontsize=10)
axes[1].set_xlabel("Hours relative to now", fontsize=10)
axes[1].legend(fontsize=8)
axes[1].grid(alpha=0.3)
axes[1].set_xlim(-24, 8)

plt.tight_layout()
st.pyplot(fig)
plt.close()

# Common metrics
c1, c2, c3, c4 = st.columns(4)
c1.metric("Flex capacity now",  f"{ts['flexible_mw'].iloc[now_bin]:.3f} MW")
c2.metric("Active jobs",        f"{ts['total_jobs'].iloc[now_bin]:,}")
c3.metric("LMP now",            f"${lmp_df['lmp_usd_mwh'].iloc[now_bin]:.2f}/MWh")
c4.metric("4CP risk window",    f"{int(fc_is_4cp.sum())} bins flagged in next 8h")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# MODE: ADVISOR
# ─────────────────────────────────────────────────────────────────────────────
if mode.startswith("🧑‍🔬"):
    st.header("Advisor — Single Job Submission")
    st.markdown("For researchers / individual users submitting one job at a time.")

    col_a, col_b, col_c = st.columns([1, 1, 2])
    with col_a:
        gpu_count = st.number_input("GPU count", 1, 500, 64, step=1)
    with col_b:
        duration_hr = st.number_input("Duration (hours)", 0.5, 48.0, 4.0, step=0.5)
    with col_c:
        flex_type = st.radio(
            "Flex class",
            ["Flex 1 — Real-time (no defer)",
             "Flex 2 — Checkpointable (≤2h defer)",
             "Flex 3 — Batch (≤8h defer)"],
            index=1, horizontal=True,
        )

    if st.button("Find optimal submission window", type="primary"):
        flex_level = int(flex_type[5])
        duration_bins = max(1, int(np.ceil(duration_hr * 60 / BIN_MIN)))
        job_mw = gpu_count * (P_MAX - P_IDLE) / 1e6
        max_defer = {1: 0, 2: 8, 3: 32}[flex_level]

        if flex_level == 1:
            st.info("**Flex 1 — submit now.** No deferral allowed.")
            # Show cost at current time
            now_cost = job_mw * fc_lmp[0:duration_bins].mean() * duration_hr
            st.metric("Expected energy cost", f"${now_cost:.2f}")
        else:
            N_FC = len(forecast)
            candidate_starts = np.arange(0, min(max_defer, N_FC - 1) + 1)
            # For each candidate, compute cost = sum(LMP × duration's MW)
            costs = []
            loads = []
            for s in candidate_starts:
                end = min(s + duration_bins, N_FC)
                cost = job_mw * fc_lmp[s:end].sum() * (BIN_MIN / 60)
                load = forecast[s:end].mean()
                costs.append(cost)
                loads.append(load)
            costs = np.array(costs); loads = np.array(loads)

            best_step = int(candidate_starts[np.argmin(costs)])
            best_cost = costs[best_step] if best_step < len(costs) else costs.min()
            now_cost  = costs[0]
            cost_save = now_cost - best_cost
            best_delay = best_step * BIN_MIN
            best_load = loads[best_step]
            now_load  = loads[0]

            # 4CP penalty: if "submit now" overlaps with a 4CP bin, flag huge savings
            now_4cp_overlap = fc_is_4cp[0:duration_bins].sum()
            best_4cp_overlap = fc_is_4cp[best_step:best_step+duration_bins].sum()

            if best_delay == 0:
                st.success(
                    f"**Submit now.** Current window has lowest expected cost "
                    f"(${now_cost:.2f}) for a {duration_hr}-hr {flex_type.split(' — ')[0]} job."
                )
            else:
                target_h = (now_hour + best_delay / 60) % 24
                tgt = f"{int(target_h):02d}:{int((target_h % 1)*60):02d}"
                st.success(
                    f"**Defer by {best_delay} min** (start at Day {now_day}, {tgt}). "
                    f"Saves **${cost_save:.2f}** in energy cost for this job."
                )

            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Recommended delay",   f"{best_delay} min")
            r2.metric("Cost if submit now",  f"${now_cost:.2f}")
            r3.metric("Cost if deferred",    f"${best_cost:.2f}", delta=f"-${cost_save:.2f}")
            r4.metric("Avg LMP (deferred)",  f"${fc_lmp[best_step:best_step+duration_bins].mean():.2f}/MWh")

            r5, r6, r7, r8 = st.columns(4)
            r5.metric("Job runtime",         f"{duration_hr} hr ({duration_bins} bins)")
            r6.metric("Job power footprint", f"{job_mw*1000:.1f} kW @ peak")
            r7.metric("Max defer budget",    f"{max_defer * BIN_MIN} min")
            r8.metric(
                "4CP exposure",
                f"{int(best_4cp_overlap)}/{int(now_4cp_overlap)} bins",
                delta=f"{int(best_4cp_overlap - now_4cp_overlap)} bins",
                delta_color="inverse",
            )

            # Plot — forecast LMP with best window highlighted
            fig2, ax = plt.subplots(figsize=(11, 3.5))
            ax2 = ax.twinx()
            ax.plot(forecast_hours, fc_lmp, color="darkorange", linewidth=1.2, label="LMP forecast")
            ax2.plot(forecast_hours, forecast, color="steelblue", linewidth=1.0, alpha=0.6, linestyle="--", label="Flex capacity forecast")
            # 4CP bins
            for i, is_4cp in enumerate(fc_is_4cp):
                if is_4cp:
                    ax.axvspan(forecast_hours[i] - BIN_MIN/120, forecast_hours[i] + BIN_MIN/120,
                               color="red", alpha=0.3)
            # Submit-now span
            now_end = min(duration_bins, N_FC)
            ax.axvspan(0, forecast_hours[now_end - 1], alpha=0.15, color="red",
                       label=f"If submit now (${now_cost:.0f})")
            # Deferred span
            if best_step > 0:
                best_end = min(best_step + duration_bins, N_FC)
                ax.axvspan(forecast_hours[best_step], forecast_hours[best_end - 1],
                           alpha=0.25, color="green",
                           label=f"If deferred (${best_cost:.0f})")
            ax.set_xlabel("Hours from now")
            ax.set_ylabel("LMP ($/MWh)", color="darkorange")
            ax2.set_ylabel("Flex Capacity (MW)", color="steelblue")
            ax.set_title(f"Optimal Window for {duration_hr}-hr Job — {flex_type.split(' — ')[0]}")
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# MODE: PLANNER
# ─────────────────────────────────────────────────────────────────────────────
else:
    st.header("Planner — Cluster-wide Dispatch Optimization")
    st.markdown(
        "For DC operators with a queue of pending jobs. "
        "Upload a CSV of pending jobs; the scheduler finds dispatch times that "
        "minimize total energy cost subject to deadlines and deferral budgets."
    )

    # Sample CSV for demo
    sample_jobs = pd.DataFrame({
        "job_id":        [f"job_{i:03d}" for i in range(1, 13)],
        "gpus":          [64, 128, 32, 256, 16, 64, 32, 8, 128, 64, 32, 16],
        "duration_hr":   [4.0, 8.0, 2.0, 6.0, 1.0, 3.0, 12.0, 0.5, 4.0, 24.0, 2.0, 6.0],
        "flex_class":    [3, 3, 2, 3, 2, 3, 3, 1, 2, 3, 2, 3],
        "deadline_hr":   [12, 16, 4, 12, 2, 6, 24, 1, 8, 48, 4, 12],
    })

    st.markdown("**CSV schema:** `job_id, gpus, duration_hr, flex_class (1/2/3), deadline_hr (relative to now)`")

    use_sample = st.checkbox("Use sample queue (12 jobs)", value=True)
    if use_sample:
        jobs_df = sample_jobs.copy()
    else:
        up = st.file_uploader("Upload job queue CSV", type="csv")
        if up is None:
            st.info("Upload a CSV or check 'Use sample queue' to continue.")
            st.stop()
        jobs_df = pd.read_csv(up)

    st.markdown("### Pending Job Queue")
    st.dataframe(jobs_df, use_container_width=True, hide_index=True)

    if st.button("Run dispatch optimization", type="primary"):
        # ── Greedy scheduler ────────────────────────────────────────────────
        # 1. For each job, compute (max_defer_bins, duration_bins, power_mw)
        # 2. Sort by 'deferral value' — jobs with widest defer × highest power footprint first
        # 3. For each job, find the start bin in [0, max_defer] that minimizes sum(LMP × MW)
        N_FC = len(forecast)

        jobs = jobs_df.copy()
        jobs["duration_bins"] = (jobs["duration_hr"] * 60 / BIN_MIN).apply(np.ceil).astype(int).clip(lower=1)
        jobs["power_mw"]      = jobs["gpus"] * (P_MAX - P_IDLE) / 1e6
        jobs["max_defer_bins"]= jobs.apply(
            lambda r: 0 if r["flex_class"] == 1
                      else min(int(r.get("deadline_hr", 8) * 60 / BIN_MIN) - r["duration_bins"],
                               {2: 8, 3: 32}[r["flex_class"]]),
            axis=1,
        ).clip(lower=0)
        jobs["defer_value"]   = jobs["max_defer_bins"] * jobs["power_mw"]
        jobs = jobs.sort_values("defer_value", ascending=False).reset_index(drop=True)

        # Capacity tracking: existing cluster load + dispatched jobs cumulative
        # We use forecast (predicted cluster load) as baseline; dispatched jobs add on top
        added_mw = np.zeros(N_FC)
        dispatch_rows = []

        for _, job in jobs.iterrows():
            dur = int(job["duration_bins"])
            max_d = int(job["max_defer_bins"])
            pmw = job["power_mw"]

            best_s, best_cost = 0, np.inf
            for s in range(0, max_d + 1):
                end = min(s + dur, N_FC)
                # Cost = sum(LMP × (cluster_baseline + added + this_job) × bin_hours)
                # For simplicity: cost = sum(LMP[s:end]) × pmw × bin_hours
                # (We optimize per-job; cluster load is shared baseline)
                cost = fc_lmp[s:end].sum() * pmw * (BIN_MIN / 60)
                # 4CP penalty — heavily discourage running during 4CP windows
                cost += fc_is_4cp[s:end].sum() * pmw * 5000  # $5k/MW per 4CP bin (arbitrary high)
                if cost < best_cost:
                    best_cost, best_s = cost, s

            # Compute "submit now" baseline cost
            now_end = min(dur, N_FC)
            now_cost = fc_lmp[0:now_end].sum() * pmw * (BIN_MIN / 60)
            now_cost += fc_is_4cp[0:now_end].sum() * pmw * 5000

            # Update added_mw
            end = min(best_s + dur, N_FC)
            added_mw[best_s:end] += pmw

            target_hour_abs = (now_hour + (best_s + 1) * BIN_MIN / 60) % 24
            target_hhmm = f"{int(target_hour_abs):02d}:{int((target_hour_abs % 1)*60):02d}"

            dispatch_rows.append({
                "job_id":         job["job_id"],
                "flex_class":     int(job["flex_class"]),
                "gpus":           int(job["gpus"]),
                "duration_hr":    job["duration_hr"],
                "delay_min":      best_s * BIN_MIN,
                "start_at":       f"Day {now_day}, {target_hhmm}",
                "avg_lmp_$/MWh":  round(fc_lmp[best_s:end].mean(), 2),
                "energy_mwh":     round(pmw * job["duration_hr"], 3),
                "cost_if_now_$":  round(now_cost - fc_is_4cp[0:now_end].sum() * pmw * 5000, 2),
                "cost_optimal_$": round(best_cost - fc_is_4cp[best_s:end].sum() * pmw * 5000, 2),
                "savings_$":      round((now_cost - best_cost), 2),
            })

        dispatch_df = pd.DataFrame(dispatch_rows).sort_values("job_id").reset_index(drop=True)

        # ── Summary ─────────────────────────────────────────────────────────
        total_now      = dispatch_df["cost_if_now_$"].sum()
        total_optimal  = dispatch_df["cost_optimal_$"].sum()
        total_savings  = total_now - total_optimal
        deferred_jobs  = (dispatch_df["delay_min"] > 0).sum()
        total_energy   = dispatch_df["energy_mwh"].sum()

        st.markdown("### 📈 Dispatch Summary")
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Total energy cost (no scheduling)", f"${total_now:.2f}")
        s2.metric("Total energy cost (optimized)",     f"${total_optimal:.2f}",
                  delta=f"-${total_savings:.2f}")
        s3.metric("Jobs deferred", f"{deferred_jobs} / {len(jobs)}")
        s4.metric("Total energy", f"{total_energy:.1f} MW·h")

        st.markdown("### 📋 Dispatch Plan")
        st.dataframe(dispatch_df, use_container_width=True, hide_index=True)

        # ── Gantt chart ─────────────────────────────────────────────────────
        st.markdown("### 🗓 Dispatch Gantt + LMP Forecast")
        fig3, (axg, axl) = plt.subplots(2, 1, figsize=(13, 7), sharex=True,
                                         gridspec_kw={"height_ratios": [2.5, 1]})
        # Gantt
        for i, (_, row) in enumerate(dispatch_df.iterrows()):
            start_h = row["delay_min"] / 60
            end_h   = start_h + row["duration_hr"]
            color   = {1: "tomato", 2: "gold", 3: "seagreen"}[row["flex_class"]]
            axg.barh(i, end_h - start_h, left=start_h, color=color, alpha=0.85, edgecolor="black", linewidth=0.5)
            axg.text(start_h + (end_h - start_h)/2, i, f"{row['job_id']} ({row['gpus']}G)",
                     ha="center", va="center", fontsize=7)
        axg.set_yticks(range(len(dispatch_df)))
        axg.set_yticklabels(dispatch_df["job_id"], fontsize=8)
        axg.set_xlabel("")
        axg.set_title(f"Dispatch Plan — {len(jobs)} jobs, ${total_savings:.2f} saved")
        axg.grid(alpha=0.3, axis="x")
        axg.invert_yaxis()

        # LMP underlay
        axl.plot(forecast_hours, fc_lmp, color="darkorange", linewidth=1.2)
        for i, is_4cp in enumerate(fc_is_4cp):
            if is_4cp:
                axl.axvspan(forecast_hours[i] - BIN_MIN/120, forecast_hours[i] + BIN_MIN/120,
                            color="red", alpha=0.4)
        axl.set_xlabel("Hours from now")
        axl.set_ylabel("LMP ($/MWh)")
        axl.grid(alpha=0.3)
        axl.set_xlim(0, max(forecast_hours.max(),
                            dispatch_df["delay_min"].max()/60 + dispatch_df["duration_hr"].max()))

        from matplotlib.patches import Patch
        axg.legend(handles=[
            Patch(facecolor="tomato",   alpha=0.85, label="Flex 1"),
            Patch(facecolor="gold",     alpha=0.85, label="Flex 2"),
            Patch(facecolor="seagreen", alpha=0.85, label="Flex 3"),
        ], fontsize=8, loc="upper right")
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close()

        # Download dispatch CSV
        csv_buf = io.StringIO()
        dispatch_df.to_csv(csv_buf, index=False)
        st.download_button(
            "📥 Download dispatch plan as CSV",
            csv_buf.getvalue(),
            file_name="dispatch_plan.csv",
            mime="text/csv",
        )

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Data: Alibaba PAI GPU cluster (2020) + ERCOT-calibrated LMP profile · "
    "Forecaster: Gradient Boosting Regressor · "
    "Scheduler: greedy with 4CP penalty (Phase 1; MILP planned for Phase 2)"
)
