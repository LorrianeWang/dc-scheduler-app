import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import io, time
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DC Workload Scheduler",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design tokens ─────────────────────────────────────────────────────────────
COLOR_INK       = "#0f172a"   # slate-900
COLOR_MUTED     = "#64748b"   # slate-500
COLOR_BORDER    = "#e2e8f0"   # slate-200
COLOR_BG        = "#f8fafc"   # slate-50
COLOR_ACCENT    = "#2563eb"   # blue-600
COLOR_GREEN     = "#10b981"   # emerald-500
COLOR_FLEX1     = "#bfdbfe"   # blue-200
COLOR_FLEX2     = "#60a5fa"   # blue-400
COLOR_FLEX3     = "#1d4ed8"   # blue-700

# ── Global CSS — use st.html() so the <style> tag isn't markdown-parsed ──────
_CSS = """
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
  html, body, [class*="css"], .stApp {
      font-family: 'Inter', -apple-system, sans-serif;
      color: #0f172a;
  }
  #MainMenu, footer, header {visibility: hidden;}
  .block-container {padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1500px;}

  h1 {font-size: 28px !important; font-weight: 700; letter-spacing: -0.02em; color: #0f172a;}
  h2 {font-size: 20px !important; font-weight: 600; letter-spacing: -0.01em; color: #0f172a; margin-top: 1.5rem;}
  h3 {font-size: 14px !important; font-weight: 600; text-transform: uppercase;
       letter-spacing: 0.08em; color: #64748b; margin-top: 1rem;}
  p, label, span, div {font-size: 14px;}

  .kpi {
      background: #f8fafc;
      border: 1px solid #e2e8f0;
      border-radius: 8px;
      padding: 16px 20px;
      height: 100%;
  }
  .kpi-label {
      font-size: 11px; font-weight: 500;
      text-transform: uppercase; letter-spacing: 0.08em;
      color: #64748b; margin-bottom: 6px;
  }
  .kpi-value {
      font-size: 32px; font-weight: 700; color: #0f172a;
      line-height: 1.1; letter-spacing: -0.02em;
  }
  .kpi-unit {font-size: 13px; font-weight: 500; color: #64748b; margin-left: 4px;}
  .kpi-delta-pos {font-size: 12px; font-weight: 600; color: #10b981; margin-top: 6px;}
  .kpi-delta-neg {font-size: 12px; font-weight: 600; color: #2563eb; margin-top: 6px;}

  .stTabs [data-baseweb="tab-list"] {gap: 0; border-bottom: 1px solid #e2e8f0;}
  .stTabs [data-baseweb="tab"] {padding: 12px 18px; font-weight: 500; color: #64748b; background: transparent;}
  .stTabs [aria-selected="true"] {color: #0f172a; border-bottom: 2px solid #2563eb;}

  [data-testid="stMetricValue"] {font-size: 28px; font-weight: 700;}
  [data-testid="stMetricLabel"] {font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em; color: #64748b;}

  section[data-testid="stSidebar"] {background: white; border-right: 1px solid #e2e8f0;}

  .stButton button {border-radius: 6px; font-weight: 500; border: 1px solid #e2e8f0;}
  .stButton button[kind="primary"] {background: #0f172a; border-color: #0f172a; color: white;}
  .stButton button[kind="primary"]:hover {background: #1e293b; border-color: #1e293b;}

  [data-testid="stDataFrame"] {border: 1px solid #e2e8f0; border-radius: 6px;}
  .stAlert {background: #f8fafc !important; border: 1px solid #e2e8f0 !important;}
</style>
"""
# Prefer st.html() (1.32+); fall back to markdown unsafe-HTML
if hasattr(st, "html"):
    st.html(_CSS)
else:
    st.markdown(_CSS, unsafe_allow_html=True)

# ── Load assets ───────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    ts  = pd.read_csv("data/ts_flex_30day_synthetic.csv")
    lmp = pd.read_csv("data/ercot_lmp_30day.csv")
    return ts, lmp

@st.cache_resource
def load_model():
    return (
        joblib.load("models/gbr_model.joblib"),
        joblib.load("models/scaler.joblib"),
        joblib.load("models/feature_cols.joblib"),
    )

ts, lmp_df = load_data()
gbr, scaler, feature_cols = load_model()

P_IDLE, P_MAX, N_GPUS = 60, 300, 6500
BIN_MIN = 15
N_FORECAST = 32  # 8 hours of forecast horizon

# ── Forecasting ───────────────────────────────────────────────────────────────
def build_features(window: pd.DataFrame) -> pd.DataFrame:
    df = window.copy()
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

@st.cache_data
def predict_horizon(now_bin: int, n_steps: int = N_FORECAST) -> np.ndarray:
    RAW = ["time_bin","hour","total_jobs","flex1_jobs","flex2_jobs","flex3_jobs","power_mw","flexible_mw"]
    buf = ts.iloc[max(0, now_bin - 110): now_bin + 1][RAW].copy().reset_index(drop=True)
    preds = []
    for step in range(n_steps):
        feats = build_features(buf)
        if len(feats) == 0:
            preds.append(float(ts["flexible_mw"].iloc[now_bin])); continue
        row = feats.iloc[[-1]][feature_cols]
        pred = float(gbr.predict(scaler.transform(row))[0])
        preds.append(pred)
        nb = now_bin + step + 1
        if nb < len(ts):
            nr = ts.iloc[[nb]][RAW].copy()
        else:
            nr = buf.iloc[[-1]][RAW].copy()
            nr["time_bin"] = nb; nr["hour"] = nb * BIN_MIN / 60
        nr = nr.copy()
        nr["flexible_mw"] = pred
        buf = pd.concat([buf, nr], ignore_index=True)
    return np.array(preds)

def get_grid_slice(now_bin: int, n_steps: int = N_FORECAST):
    end = min(now_bin + n_steps + 1, len(lmp_df))
    sl  = lmp_df.iloc[now_bin + 1 : end]
    lmp_v   = sl["lmp_usd_mwh"].values
    carb_v  = sl["carbon_g_kwh"].values
    is_4cp  = sl["is_4cp_bin"].values.astype(bool)
    # Pad if at end of data
    if len(lmp_v) < n_steps:
        lmp_v  = np.pad(lmp_v,  (0, n_steps - len(lmp_v)),  constant_values=lmp_v[-1] if len(lmp_v) else 30)
        carb_v = np.pad(carb_v, (0, n_steps - len(carb_v)), constant_values=carb_v[-1] if len(carb_v) else 380)
        is_4cp = np.pad(is_4cp, (0, n_steps - len(is_4cp)), constant_values=False)
    return lmp_v, carb_v, is_4cp

# ── UI components ─────────────────────────────────────────────────────────────
def kpi_card(label, value, unit="", delta=None, delta_pos=True):
    delta_html = ""
    if delta is not None:
        cls = "kpi-delta-pos" if delta_pos else "kpi-delta-neg"
        delta_html = f'<div class="{cls}">{delta}</div>'
    st.markdown(
        f'<div class="kpi"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}<span class="kpi-unit">{unit}</span></div>'
        f'{delta_html}</div>',
        unsafe_allow_html=True,
    )

PLOTLY_LAYOUT = dict(
    plot_bgcolor="white",
    paper_bgcolor="white",
    font=dict(family="Inter, sans-serif", size=12, color=COLOR_INK),
    margin=dict(l=50, r=20, t=20, b=40),
    xaxis=dict(gridcolor=COLOR_BORDER, zeroline=False, linecolor=COLOR_BORDER),
    yaxis=dict(gridcolor=COLOR_BORDER, zeroline=False, linecolor=COLOR_BORDER),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                font=dict(size=11)),
)

def apply_layout(fig, **overrides):
    """Apply PLOTLY_LAYOUT then merge overrides (deep-merging dict values like xaxis/yaxis)."""
    merged = {k: (dict(v) if isinstance(v, dict) else v) for k, v in PLOTLY_LAYOUT.items()}
    for k, v in overrides.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = {**merged[k], **v}
        else:
            merged[k] = v
    fig.update_layout(**merged)
    return fig

# ── Session state ─────────────────────────────────────────────────────────────
MAX_BIN = len(ts) - N_FORECAST - 10
if "now_bin" not in st.session_state:
    st.session_state.now_bin = 1440
if "playing" not in st.session_state:
    st.session_state.playing = False

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.markdown("### DC Workload Scheduler")
st.sidebar.caption("Carbon- and cost-aware GPU job scheduling")
st.sidebar.markdown("---")

mode = st.sidebar.radio(
    "Mode",
    ["Advisor — single job", "Planner — job queue"],
    label_visibility="visible",
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Simulated time**")
now_bin = st.sidebar.slider(
    "Bin index", min_value=120, max_value=MAX_BIN,
    value=st.session_state.now_bin, step=1, label_visibility="collapsed",
)
# Only update from slider if not playing
if not st.session_state.playing and now_bin != st.session_state.now_bin:
    st.session_state.now_bin = now_bin
nb = st.session_state.now_bin

now_hour = nb * BIN_MIN / 60
day = int(now_hour // 24) + 1
hhmm = f"{int(now_hour % 24):02d}:{int((now_hour % 1) * 60):02d}"
st.sidebar.markdown(f"**Day {day}, {hhmm}**")

# Play controls
ctl1, ctl2, ctl3 = st.sidebar.columns([1, 1, 1])
if ctl1.button("Pause" if st.session_state.playing else "Play", use_container_width=True):
    st.session_state.playing = not st.session_state.playing
if ctl2.button("Reset", use_container_width=True):
    st.session_state.now_bin = 1440
    st.session_state.playing = False
speed = ctl3.selectbox("Speed", [1, 4, 16, 60], index=2, label_visibility="collapsed",
                       format_func=lambda x: f"{x}x")

# ── Header ────────────────────────────────────────────────────────────────────
st.title("GPU Cluster Scheduler")
st.caption("Alibaba PAI workload trace · ERCOT-calibrated LMP & carbon intensity · GBM forecasting")

# ── Compute forecast and current-bin metrics ─────────────────────────────────
forecast = predict_horizon(nb, N_FORECAST)
fc_lmp, fc_carb, fc_4cp = get_grid_slice(nb, N_FORECAST)
hist_start = max(0, nb - 96)
recent_std = ts["flexible_mw"].iloc[hist_start:nb].std() or 0.05
forecast_lo = (forecast - 0.8 * recent_std).clip(0)
forecast_hi = forecast + 0.8 * recent_std
forecast_hours = np.arange(1, N_FORECAST + 1) * BIN_MIN / 60

current_lmp  = lmp_df["lmp_usd_mwh"].iloc[nb]
current_carb = lmp_df["carbon_g_kwh"].iloc[nb]
current_cap  = ts["flexible_mw"].iloc[nb]
current_jobs = ts["total_jobs"].iloc[nb]
current_pwr  = ts["power_mw"].iloc[nb]

# Running totals from bin 0 to now (15-min bins → /4 to get hours)
running_spend = (ts["power_mw"].iloc[:nb+1].values * lmp_df["lmp_usd_mwh"].iloc[:nb+1].values * 0.25).sum()
running_co2   = (ts["power_mw"].iloc[:nb+1].values * 1000 * lmp_df["carbon_g_kwh"].iloc[:nb+1].values * 0.25 / 1e6).sum()  # MW × 1000 × g/kWh × hr / 1e6 = tonnes
running_jobs  = int(ts["total_jobs"].iloc[:nb+1].sum() / 96)  # avg per bin × bins → rough job-hours

# Cost-carbon conflicts in next 8h: count of bins where lmp-rank != carbon-rank by >5 positions
conflicts = 0
if N_FORECAST > 8:
    lmp_rank  = pd.Series(fc_lmp).rank()
    carb_rank = pd.Series(fc_carb).rank()
    conflicts = int((abs(lmp_rank - carb_rank) > 8).sum())

# ── KPI Bar ───────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
with k1: kpi_card("Electricity spend so far", f"${running_spend:,.0f}", "running total")
with k2: kpi_card("CO₂ emitted so far",       f"{running_co2:,.1f}",     "tonnes")
with k3: kpi_card("Cluster snapshot",         f"{int(current_jobs):,}",  "active jobs")
with k4: kpi_card("Cost-carbon conflicts",    f"{conflicts}",
                  "bins in next 8h" if conflicts > 0 else "no conflicts")

st.markdown("")

# Secondary row of grid signals
g1, g2, g3, g4 = st.columns(4)
with g1: kpi_card("Current LMP",        f"${current_lmp:.2f}",    "/MWh")
with g2: kpi_card("Current carbon",     f"{int(current_carb)}",   "gCO₂/kWh")
with g3: kpi_card("Current power draw", f"{current_pwr:.2f}",     "MW")
with g4:
    if lmp_df["is_4cp_bin"].iloc[nb]:
        kpi_card("4CP risk", "ACTIVE", "avoid loading")
    else:
        peak_in_8h = int(fc_4cp.sum())
        kpi_card("4CP risk", f"{peak_in_8h}", "candidate bins ahead")

st.markdown("---")

# ── Auto-advance ──────────────────────────────────────────────────────────────
if st.session_state.playing:
    st.session_state.now_bin = min(nb + speed, MAX_BIN)
    if st.session_state.now_bin >= MAX_BIN:
        st.session_state.playing = False
    time.sleep(0.25)
    st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# ADVISOR MODE
# ═══════════════════════════════════════════════════════════════════════════════
if mode.startswith("Advisor"):
    st.markdown("### Job Parameters")
    j1, j2, j3 = st.columns([1, 1, 2])
    with j1:
        gpu_count = st.number_input("GPU count", 1, 500, 64, step=1)
    with j2:
        duration_hr = st.number_input("Duration (hours)", 0.5, 48.0, 4.0, step=0.5)
    with j3:
        flex_type = st.radio(
            "Flex class",
            ["Flex 1 — Real-time", "Flex 2 — Checkpointable", "Flex 3 — Batch"],
            horizontal=True, index=1,
        )

    flex_level   = int(flex_type[5])
    duration_bins = max(1, int(np.ceil(duration_hr * 60 / BIN_MIN)))
    job_mw       = gpu_count * (P_MAX - P_IDLE) / 1e6
    max_defer    = {1: 0, 2: 8, 3: 32}[flex_level]

    # Compute recommendation
    N_FC = len(forecast)
    candidate_starts = np.arange(0, min(max_defer, N_FC - 1) + 1)
    costs, carbs, loads = [], [], []
    for s in candidate_starts:
        end = min(s + duration_bins, N_FC)
        costs.append(job_mw * fc_lmp[s:end].sum() * (BIN_MIN / 60))
        carbs.append(job_mw * 1000 * fc_carb[s:end].sum() * (BIN_MIN / 60) / 1e6)  # tonnes
        loads.append(forecast[s:end].mean())
    costs, carbs, loads = np.array(costs), np.array(carbs), np.array(loads)

    if len(costs) == 0:
        best_step = 0
    else:
        best_step = int(candidate_starts[np.argmin(costs)])

    best_delay = best_step * BIN_MIN
    now_end    = min(duration_bins, N_FC)
    best_end   = min(best_step + duration_bins, N_FC)
    now_cost   = float(job_mw * fc_lmp[0:now_end].sum() * (BIN_MIN / 60))
    best_cost  = float(costs[best_step]) if len(costs) > best_step else now_cost
    now_carb   = float(job_mw * 1000 * fc_carb[0:now_end].sum() * (BIN_MIN / 60) / 1e6)
    best_carb  = float(carbs[best_step]) if len(carbs) > best_step else now_carb
    cost_save  = now_cost - best_cost
    carb_save  = now_carb - best_carb

    target_h   = (now_hour + best_delay / 60) % 24
    target_str = f"Day {day}, {int(target_h):02d}:{int((target_h % 1) * 60):02d}"

    st.markdown("### Recommendation")
    rec_label = "Submit now" if best_step == 0 else f"Defer {best_delay} min · start {target_str}"

    r1, r2, r3, r4 = st.columns(4)
    with r1: kpi_card("Action", rec_label, "")
    with r2: kpi_card("Energy cost",  f"${best_cost:,.2f}",
                      delta=f"saves ${cost_save:,.2f}" if cost_save > 0.01 else None, delta_pos=True)
    with r3: kpi_card("CO₂ emitted",  f"{best_carb*1000:,.1f}", "kg",
                      delta=f"saves {carb_save*1000:.1f} kg" if carb_save > 0.01 else None, delta_pos=True)
    with r4: kpi_card("4CP exposure", f"{int(fc_4cp[best_step:best_end].sum())}/{int(fc_4cp[:now_end].sum())}",
                      "bins (best / now)")

    st.markdown("")
    col_left, col_right = st.columns(2)

    # ── Left: Flex capacity forecast ────────────────────────────────────────
    with col_left:
        st.markdown("### Cluster flex-capacity forecast")
        hist = ts.iloc[hist_start:nb+1]
        hist_h = hist["hour"].values - now_hour
        hist_lmp_v = lmp_df["lmp_usd_mwh"].iloc[hist_start:nb+1].values

        fig = go.Figure()
        # Confidence interval (drawn first as background)
        fig.add_trace(go.Scatter(
            x=np.concatenate([forecast_hours, forecast_hours[::-1]]),
            y=np.concatenate([forecast_hi, forecast_lo[::-1]]),
            fill="toself", fillcolor="rgba(37, 99, 235, 0.12)",
            line=dict(width=0), hoverinfo="skip", showlegend=False, name="80% interval",
        ))
        fig.add_trace(go.Scatter(
            x=hist_h, y=hist["flexible_mw"].values,
            mode="lines", line=dict(color=COLOR_INK, width=1.5),
            name="Historical", hovertemplate="%{y:.3f} MW<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=forecast_hours, y=forecast,
            mode="lines", line=dict(color=COLOR_ACCENT, width=2, dash="dash"),
            name="Forecast", hovertemplate="%{y:.3f} MW<extra></extra>",
        ))
        # Now line
        fig.add_vline(x=0, line_width=1, line_dash="dot", line_color=COLOR_MUTED)
        apply_layout(fig, height=320,
                     yaxis_title="Flex capacity (MW)",
                     xaxis_title="Hours from now",
                     xaxis=dict(range=[-24, 8]))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # ── Right: LMP forecast with windows ────────────────────────────────────
    with col_right:
        st.markdown("### LMP forecast — submission windows")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=forecast_hours, y=fc_lmp,
            mode="lines", line=dict(color=COLOR_INK, width=1.5),
            name="LMP", hovertemplate="$%{y:.2f}/MWh<extra></extra>",
        ))
        # 4CP markers
        cp_x = forecast_hours[fc_4cp]
        if len(cp_x) > 0:
            fig.add_trace(go.Scatter(
                x=cp_x, y=fc_lmp[fc_4cp],
                mode="markers", marker=dict(color=COLOR_ACCENT, size=10, symbol="diamond-open"),
                name="4CP", hovertemplate="4CP candidate<extra></extra>",
            ))
        # Now-submit window (light shade)
        fig.add_vrect(x0=0, x1=forecast_hours[now_end - 1],
                      fillcolor=COLOR_MUTED, opacity=0.12, line_width=0,
                      annotation_text="if now", annotation_position="top left",
                      annotation_font_size=10, annotation_font_color=COLOR_MUTED)
        # Deferred window (dark blue shade)
        if best_step > 0:
            fig.add_vrect(x0=forecast_hours[best_step], x1=forecast_hours[best_end - 1],
                          fillcolor=COLOR_ACCENT, opacity=0.18, line_width=0,
                          annotation_text="recommended", annotation_position="top right",
                          annotation_font_size=10, annotation_font_color=COLOR_ACCENT)
        apply_layout(fig, height=320,
                     yaxis_title="LMP ($/MWh)", xaxis_title="Hours from now")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# ═══════════════════════════════════════════════════════════════════════════════
# PLANNER MODE
# ═══════════════════════════════════════════════════════════════════════════════
else:
    st.markdown("### Pending Job Queue")
    sample_jobs = pd.DataFrame({
        "job_id":       [f"job_{i:03d}" for i in range(1, 13)],
        "gpus":         [64, 128, 32, 256, 16, 64, 32, 8, 128, 64, 32, 16],
        "duration_hr":  [4.0, 8.0, 2.0, 6.0, 1.0, 3.0, 12.0, 0.5, 4.0, 24.0, 2.0, 6.0],
        "flex_class":   [3, 3, 2, 3, 2, 3, 3, 1, 2, 3, 2, 3],
        "deadline_hr":  [12, 16, 4, 12, 2, 6, 24, 1, 8, 48, 4, 12],
    })

    queue_choice = st.radio(
        "Source", ["Sample queue (12 jobs)", "Upload CSV"],
        horizontal=True, label_visibility="collapsed",
    )
    if queue_choice == "Sample queue (12 jobs)":
        jobs_df = sample_jobs.copy()
    else:
        up = st.file_uploader("CSV with columns: job_id, gpus, duration_hr, flex_class, deadline_hr", type="csv")
        if up is None:
            st.info("Upload a CSV or switch to sample queue.")
            st.stop()
        jobs_df = pd.read_csv(up)

    st.dataframe(jobs_df, use_container_width=True, hide_index=True)

    run = st.button("Run dispatch optimization", type="primary", use_container_width=False)
    if run:
        N_FC = len(forecast)
        jobs = jobs_df.copy()
        jobs["duration_bins"]  = (jobs["duration_hr"] * 60 / BIN_MIN).apply(np.ceil).astype(int).clip(lower=1)
        jobs["power_mw"]       = jobs["gpus"] * (P_MAX - P_IDLE) / 1e6
        jobs["max_defer_bins"] = jobs.apply(
            lambda r: 0 if r["flex_class"] == 1
                      else min(int(r.get("deadline_hr", 8) * 60 / BIN_MIN) - r["duration_bins"],
                               {2: 8, 3: 32}[r["flex_class"]]),
            axis=1,
        ).clip(lower=0)
        jobs["defer_value"] = jobs["max_defer_bins"] * jobs["power_mw"]
        jobs = jobs.sort_values("defer_value", ascending=False).reset_index(drop=True)

        dispatch_rows = []
        for _, job in jobs.iterrows():
            dur = int(job["duration_bins"]); max_d = int(job["max_defer_bins"])
            pmw = job["power_mw"]
            best_s, best_cost = 0, np.inf
            for s in range(0, max_d + 1):
                end = min(s + dur, N_FC)
                cost = fc_lmp[s:end].sum() * pmw * (BIN_MIN / 60)
                cost += fc_4cp[s:end].sum() * pmw * 5000
                if cost < best_cost:
                    best_cost, best_s = cost, s
            end = min(best_s + dur, N_FC)
            now_end = min(dur, N_FC)
            now_cost = fc_lmp[0:now_end].sum() * pmw * (BIN_MIN / 60)
            now_cost += fc_4cp[0:now_end].sum() * pmw * 5000

            target_h = (now_hour + (best_s + 1) * BIN_MIN / 60) % 24
            target_hhmm = f"{int(target_h):02d}:{int((target_h % 1) * 60):02d}"

            dispatch_rows.append({
                "job_id":        job["job_id"],
                "flex_class":    int(job["flex_class"]),
                "gpus":          int(job["gpus"]),
                "duration_hr":   job["duration_hr"],
                "delay_min":     best_s * BIN_MIN,
                "start_at":      f"Day {day}, {target_hhmm}",
                "avg_lmp":       round(fc_lmp[best_s:end].mean(), 2),
                "avg_carbon":    round(fc_carb[best_s:end].mean(), 0),
                "energy_mwh":    round(pmw * job["duration_hr"], 3),
                "cost_if_now":   round(now_cost - fc_4cp[0:now_end].sum() * pmw * 5000, 2),
                "cost_optimal":  round(best_cost - fc_4cp[best_s:end].sum() * pmw * 5000, 2),
                "savings":       round(now_cost - best_cost, 2),
            })

        dispatch_df = pd.DataFrame(dispatch_rows).sort_values("job_id").reset_index(drop=True)

        # ── Summary KPIs ────────────────────────────────────────────────────
        total_now      = dispatch_df["cost_if_now"].sum()
        total_optimal  = dispatch_df["cost_optimal"].sum()
        total_savings  = total_now - total_optimal
        deferred       = int((dispatch_df["delay_min"] > 0).sum())
        total_energy   = dispatch_df["energy_mwh"].sum()
        # Carbon savings
        total_carb_now = (dispatch_df["energy_mwh"] *
                           dispatch_df.apply(lambda r:
                               fc_carb[0:min(int(np.ceil(r['duration_hr']*60/BIN_MIN)), N_FC)].mean(), axis=1)
                           / 1000).sum()
        total_carb_opt = (dispatch_df["energy_mwh"] * dispatch_df["avg_carbon"] / 1000).sum()
        carb_savings_kg = (total_carb_now - total_carb_opt) * 1000

        st.markdown("### Dispatch summary")
        s1, s2, s3, s4 = st.columns(4)
        with s1: kpi_card("Cost without scheduling", f"${total_now:,.2f}", "total")
        with s2: kpi_card("Cost with scheduling",    f"${total_optimal:,.2f}", "",
                          delta=f"saves ${total_savings:,.2f} ({100*total_savings/total_now:.1f}%)",
                          delta_pos=True)
        with s3: kpi_card("Jobs deferred",           f"{deferred}", f"of {len(jobs)}")
        with s4: kpi_card("CO₂ saved",               f"{carb_savings_kg:,.0f}", "kg")

        # ── Gantt timeline (plotly) ─────────────────────────────────────────
        st.markdown("### Dispatch timeline")
        gantt = dispatch_df.copy()
        gantt["start_hr"] = gantt["delay_min"] / 60
        gantt["end_hr"]   = gantt["start_hr"] + gantt["duration_hr"]
        # Use minutes for px.timeline (it expects datetime-like; we trick by using base = today)
        base = pd.Timestamp("2025-01-01 00:00")
        gantt["start_dt"] = base + pd.to_timedelta(gantt["start_hr"], unit="h")
        gantt["end_dt"]   = base + pd.to_timedelta(gantt["end_hr"], unit="h")
        gantt["flex_label"] = gantt["flex_class"].map({1: "Flex 1", 2: "Flex 2", 3: "Flex 3"})

        fig = px.timeline(
            gantt, x_start="start_dt", x_end="end_dt", y="job_id",
            color="flex_label",
            color_discrete_map={"Flex 1": COLOR_FLEX1, "Flex 2": COLOR_FLEX2, "Flex 3": COLOR_FLEX3},
            hover_data={"gpus": True, "duration_hr": True, "delay_min": True,
                        "savings": ":.2f", "start_dt": False, "end_dt": False, "flex_label": False},
        )
        fig.update_yaxes(autorange="reversed")
        apply_layout(fig, height=420,
                     xaxis_title="", yaxis_title="",
                     xaxis=dict(tickformat="%Hh"),
                     legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                 xanchor="left", x=0, title=""))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        # ── LMP underlay ────────────────────────────────────────────────────
        st.markdown("### LMP forecast (overlay for context)")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=forecast_hours, y=fc_lmp,
            mode="lines", line=dict(color=COLOR_INK, width=1.5),
            hovertemplate="$%{y:.2f}/MWh<extra></extra>", showlegend=False,
        ))
        for i, is_cp in enumerate(fc_4cp):
            if is_cp:
                fig.add_vrect(x0=forecast_hours[i] - BIN_MIN/120,
                              x1=forecast_hours[i] + BIN_MIN/120,
                              fillcolor=COLOR_ACCENT, opacity=0.4, line_width=0)
        apply_layout(fig, height=200,
                     yaxis_title="LMP ($/MWh)", xaxis_title="Hours from now")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        # ── Dispatch table ──────────────────────────────────────────────────
        st.markdown("### Dispatch plan")
        st.dataframe(dispatch_df, use_container_width=True, hide_index=True)

        csv_buf = io.StringIO()
        dispatch_df.to_csv(csv_buf, index=False)
        st.download_button(
            "Download as CSV", csv_buf.getvalue(),
            file_name="dispatch_plan.csv", mime="text/csv",
        )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Data: Alibaba PAI GPU cluster (2020) · ERCOT 2023-calibrated LMP & carbon intensity · "
    "Forecaster: Gradient Boosting Regressor · Scheduler: greedy with 4CP penalty"
)
