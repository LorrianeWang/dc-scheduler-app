import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import io
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
  #MainMenu, footer {visibility: hidden;}
  /* Hide the run/stop/share toolbar but keep the sidebar collapse arrow */
  [data-testid="stToolbar"] {display: none !important;}
  [data-testid="stHeader"] {background: transparent;}
  /* Force the collapsed-sidebar toggle to stay visible — without this it can
     disappear once the user collapses the sidebar on Streamlit Cloud */
  [data-testid="collapsedControl"] {
      visibility: visible !important;
      display: block !important;
      opacity: 1 !important;
      z-index: 999 !important;
  }
  [data-testid="stSidebarCollapseButton"] {visibility: visible !important;}
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

  .action-banner {
      background: #f8fafc;
      border: 1px solid #e2e8f0;
      border-left: 3px solid #2563eb;
      border-radius: 8px;
      padding: 14px 20px;
      display: flex;
      align-items: center;
      gap: 24px;
      margin-bottom: 12px;
  }
  .action-banner-label {
      font-size: 11px; font-weight: 600;
      text-transform: uppercase; letter-spacing: 0.08em;
      color: #64748b;
      white-space: nowrap;
  }
  .action-banner-value {
      font-size: 18px; font-weight: 600; color: #0f172a;
      letter-spacing: -0.01em;
  }

  /* Sidebar footer link — visually distinct from radio choices */
  .nav-link-wrap > div > button {
      background: transparent !important;
      border: none !important;
      padding: 0 !important;
      color: #64748b !important;
      font-size: 13px !important;
      font-weight: 500 !important;
      text-align: left !important;
      box-shadow: none !important;
  }
  .nav-link-wrap > div > button:hover {
      color: #2563eb !important;
      background: transparent !important;
  }

  /* Methodology page */
  .mtd-citation {
      font-size: 12px; color: #64748b; line-height: 1.6;
      padding-left: 1em; text-indent: -1em;
  }
  .mtd-stat-table {
      border-collapse: collapse; width: 100%;
      font-size: 13px; margin: 12px 0 18px 0;
  }
  .mtd-stat-table th {
      text-align: left; font-weight: 600; font-size: 11px;
      text-transform: uppercase; letter-spacing: 0.06em;
      color: #64748b; border-bottom: 1px solid #e2e8f0;
      padding: 8px 12px;
  }
  .mtd-stat-table td {
      padding: 8px 12px; border-bottom: 1px solid #f1f5f9;
  }
  .mtd-stat-table tr.best td {font-weight: 600; color: #0f172a;}
  .mtd-stat-table tr.best td:first-child::before {
      content: "● "; color: #2563eb; margin-right: 4px;
  }
  .mtd-stat-table tr td:first-child {color: #475569;}

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

def action_banner(label, value):
    st.markdown(
        f'<div class="action-banner">'
        f'<div class="action-banner-label">{label}</div>'
        f'<div class="action-banner-value">{value}</div>'
        f'</div>',
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
if "advisor_result" not in st.session_state:
    st.session_state.advisor_result = None
if "view" not in st.session_state:
    st.session_state.view = "app"

def _on_mode_change():
    # Any radio interaction also returns the user from methodology view
    st.session_state.view = "app"

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.markdown("### DC Workload Scheduler")
st.sidebar.caption("Carbon- and cost-aware GPU job scheduling")
st.sidebar.markdown("---")

mode = st.sidebar.radio(
    "Mode",
    ["Advisor — single job", "Planner — job queue"],
    label_visibility="visible",
    on_change=_on_mode_change,
    key="mode_radio",
)

# Slider + clock only shown in app view (irrelevant on methodology page)
if st.session_state.view == "app":
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Simulated time**")
    now_bin_input = st.sidebar.slider(
        "Bin index", min_value=120, max_value=MAX_BIN,
        value=st.session_state.now_bin, step=1, label_visibility="collapsed",
    )
    if now_bin_input != st.session_state.now_bin:
        st.session_state.now_bin = now_bin_input
    nb = st.session_state.now_bin
    now_hour = nb * BIN_MIN / 60
    day = int(now_hour // 24) + 1
    hhmm = f"{int(now_hour % 24):02d}:{int((now_hour % 1) * 60):02d}"
    st.sidebar.markdown(f"**Day {day}, {hhmm}**")
    if st.sidebar.button("Reset to default", use_container_width=True):
        st.session_state.now_bin = 1440
        st.rerun()

# Footer link — visually distinct from radio buttons
st.sidebar.markdown("---")
with st.sidebar.container():
    st.markdown('<div class="nav-link-wrap">', unsafe_allow_html=True)
    if st.session_state.view == "app":
        if st.sidebar.button("Methodology & validation →", key="link_to_mtd",
                             use_container_width=True):
            st.session_state.view = "methodology"
            st.rerun()
    else:
        if st.sidebar.button("← Back to application", key="link_to_app",
                             use_container_width=True):
            st.session_state.view = "app"
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("GPU Cluster Scheduler")
st.caption("Alibaba PAI workload trace · ERCOT-calibrated LMP & carbon intensity · GBM forecasting")

# ═════════════════════════════════════════════════════════════════════════════
# Route to methodology view if requested; otherwise render the application.
# ═════════════════════════════════════════════════════════════════════════════
if st.session_state.view == "methodology":
    # ── Methodology & validation page ────────────────────────────────────────
    from sklearn.linear_model import LinearRegression as _LinReg

    @st.cache_data
    def _compute_test_predictions():
        """Replay training pipeline on synthetic data, generate test-set predictions for live models."""
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
        df["target"] = df["flexible_mw"].shift(-16)
        df = df.dropna().reset_index(drop=True)
        X = df[feature_cols].values
        y = df["target"].values
        split = int(len(df) * 0.8)
        X_train_sc = scaler.transform(X[:split])
        X_test_sc  = scaler.transform(X[split:])
        y_train, y_test = y[:split], y[split:]

        gbm_pred = gbr.predict(X_test_sc)
        lr = _LinReg().fit(X_train_sc, y_train)
        lr_pred = lr.predict(X_test_sc)
        persistence_pred = X[split:, feature_cols.index("flexible_mw_lag1")]

        hours = (df["hour"].values[split:] - df["hour"].values[split])
        return dict(hours=hours, actual=y_test,
                    gbm=gbm_pred, persistence=persistence_pred, linear=lr_pred)

    # ─── 1. The problem ──────────────────────────────────────────────────────
    st.markdown("## 1. The Problem")
    st.markdown(
        "GPU clusters now consume hundreds of megawatts at hyperscale. In Alibaba's "
        "PAI production trace [1], approximately **50% of submitted jobs are deferrable** — "
        "training, batch processing, and background workloads with no tight latency "
        "requirements. The remaining ~50% (inference, real-time monitoring) cannot be moved.\n\n"
        "In grid markets like ERCOT, deferrable load creates three monetizable opportunities:"
    )
    st.markdown(
        "- **Energy arbitrage** via Locational Marginal Prices (LMPs) that swing $30 → $500+/MWh within hours\n"
        "- **Avoidance of the Four Coincident Peaks (4CP)** [3], which set transmission charges (often 20–30% of an industrial customer's annual bill)\n"
        "- **Demand response capacity revenue** for committing to load curtailment on signal"
    )
    st.markdown(
        "This project asks: can a learned forecaster of cluster deferrable capacity, "
        "combined with a cost- and carbon-aware scheduler, capture this value?"
    )

    # ─── 2. Data foundation ──────────────────────────────────────────────────
    st.markdown("## 2. Data Foundation")
    st.markdown(
        "We use the **Alibaba PAI v2020 GPU cluster trace** [1] — 7 days, 6,500 GPUs, "
        "91k jobs across multiple GPU SKUs (T4 / P100 / V100 / V100M32). We classify "
        "each job into three flexibility tiers based on `task_name` and runtime:"
    )
    st.markdown(
        "- **Flex 1 (0% deferrable):** `evaluator`, `TensorboardTask` — real-time monitoring\n"
        "- **Flex 2 (50% deferrable):** `worker`, `ps`, `PyTorchWorker`, etc., with runtime < 24h — checkpointable training\n"
        "- **Flex 3 (100% deferrable):** training jobs > 24h + batch workers — no hard deadline"
    )

    # Stacked composition chart on the synthetic 30d data
    plot_df = ts[["hour", "flex1_jobs", "flex2_jobs", "flex3_jobs"]].iloc[:7*96]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=plot_df["hour"], y=plot_df["flex1_jobs"], stackgroup="one",
                             name="Flex 1", fillcolor=COLOR_FLEX1,
                             line=dict(width=0), hovertemplate="%{y:.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=plot_df["hour"], y=plot_df["flex2_jobs"], stackgroup="one",
                             name="Flex 2", fillcolor=COLOR_FLEX2,
                             line=dict(width=0), hovertemplate="%{y:.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=plot_df["hour"], y=plot_df["flex3_jobs"], stackgroup="one",
                             name="Flex 3", fillcolor=COLOR_FLEX3,
                             line=dict(width=0), hovertemplate="%{y:.0f}<extra></extra>"))
    apply_layout(fig, height=260, yaxis_title="Active jobs",
                 xaxis_title="Hours (first 7 days)")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.markdown(
        "Because 7 days is insufficient to train deep sequence models, we **augment with "
        "30 days of synthetic data** calibrated to the real distribution: daily/weekly "
        "seasonality, flex-class ratios, job-arrival spike statistics. Synthetic data is "
        "used for sequence-model training only; real data drives workload characterization "
        "and the scheduler's structural assumptions."
    )

    # ─── 3. Forecasting models ───────────────────────────────────────────────
    st.markdown("## 3. Forecasting Models")
    st.markdown(
        "We trained **seven forecasters** to predict cluster deferrable capacity (in MW) "
        "4 hours ahead at 15-minute resolution, using lag features, rolling statistics, "
        "and calendar variables. Train/test split is the last 20% of the 30-day window, "
        "preserving temporal order."
    )

    # Static results table
    st.markdown(
        '<table class="mtd-stat-table">'
        '<tr><th>Model</th><th>MAE (MW)</th><th>RMSE (MW)</th><th>MAPE</th><th>Notes</th></tr>'
        '<tr><td>Persistence</td><td>0.0885</td><td>0.1104</td><td>20.5%</td><td>Predict last observed value</td></tr>'
        '<tr><td>Linear Regression</td><td>0.0539</td><td>0.0715</td><td>12.2%</td><td>Lag + calendar features</td></tr>'
        '<tr class="best"><td>Gradient Boosting</td><td>0.0448</td><td>0.0607</td><td>10.0%</td><td>Deployed in production</td></tr>'
        '<tr><td>LSTM v1 (50 ep)</td><td>0.0468</td><td>0.0636</td><td>10.5%</td><td>2 layers, 64 hidden, lookback 96</td></tr>'
        '<tr><td>LSTM v2 (100 ep + LR sched)</td><td>0.0515</td><td>0.0706</td><td>11.7%</td><td>ReduceLROnPlateau</td></tr>'
        '<tr><td>TCN</td><td>0.0500</td><td>0.0676</td><td>11.5%</td><td>Train loss 0.0006 — overfit</td></tr>'
        '<tr><td>Quantile LSTM (q50)</td><td>0.0512</td><td>—</td><td>—</td><td>Pinball loss, 80% interval target</td></tr>'
        '</table>',
        unsafe_allow_html=True,
    )

    st.markdown(
        "**Interactive comparison.** Select models to overlay their predictions on the "
        "synthetic test window. Persistence, Linear, and GBM are computed live from the "
        "deployed scaler + model. Deep models (LSTM / TCN / Quantile LSTM) are summarized "
        "in the table above; their test-set predictions can be added by running the "
        "training pipeline on Colab and bundling `model_predictions.csv` into `/data`."
    )
    selected = st.multiselect(
        "Models",
        ["Gradient Boosting (deployed)", "Persistence", "Linear Regression"],
        default=["Gradient Boosting (deployed)", "Persistence"],
        label_visibility="collapsed",
    )
    preds = _compute_test_predictions()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=preds["hours"], y=preds["actual"],
                             mode="lines", line=dict(color=COLOR_INK, width=1.5),
                             name="Actual", hovertemplate="%{y:.3f} MW<extra></extra>"))
    style_map = {
        "Gradient Boosting (deployed)": (COLOR_ACCENT, "solid", "gbm"),
        "Persistence":                  (COLOR_MUTED, "dot",   "persistence"),
        "Linear Regression":            ("#94a3b8",   "dash",  "linear"),
    }
    for label in selected:
        color, dash, key = style_map[label]
        fig.add_trace(go.Scatter(x=preds["hours"], y=preds[key],
                                 mode="lines", line=dict(color=color, width=1.4, dash=dash),
                                 name=label, hovertemplate="%{y:.3f} MW<extra></extra>"))
    apply_layout(fig, height=320, yaxis_title="Flex capacity (MW)",
                 xaxis_title="Hours into test window")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.markdown(
        "We hypothesized that deep sequence models would outperform classical methods on "
        "this temporal task. Empirically, **Gradient Boosting with hand-crafted lag features "
        "achieves the lowest MAE (0.0448 MW)** — narrowly beating both LSTM variants and TCN. "
        "We attribute this to (a) the limited training data (2,880 samples after synthetic "
        "augmentation), and (b) the dominant calendar periodicity that gradient boosting "
        "captures efficiently. TCN's training loss is 4× lower than LSTM's, but its test MAE "
        "is worse — a classic overfitting signature on a small dataset."
    )

    # ─── 4. Feature importance ───────────────────────────────────────────────
    st.markdown("## 4. What Drives the Forecast")

    importances = gbr.feature_importances_
    idx = np.argsort(importances)[::-1]
    sorted_feat = [feature_cols[i] for i in idx]
    sorted_imp  = importances[idx]
    name_map = {
        "flexible_mw_lag1": "flex_mw lag-1 (15 min)",
        "flexible_mw_lag4": "flex_mw lag-4 (1 hr)",
        "flexible_mw_lag8": "flex_mw lag-8 (2 hr)",
        "flexible_mw_lag32": "flex_mw lag-32 (8 hr)",
        "flexible_mw_lag96": "flex_mw lag-96 (24 hr)",
        "flexible_mw_roll4_mean": "flex_mw roll mean 1hr",
        "flexible_mw_roll4_std": "flex_mw roll std 1hr",
        "flexible_mw_roll96_mean": "flex_mw roll mean 24hr",
        "total_jobs_lag1": "total_jobs lag-1",
        "total_jobs_lag4": "total_jobs lag-4",
        "total_jobs_lag8": "total_jobs lag-8",
        "total_jobs_lag32": "total_jobs lag-32",
        "total_jobs_lag96": "total_jobs lag-96",
        "hour_of_day": "hour of day",
        "day_of_week": "day of week",
        "is_business_hours": "business hours flag",
        "job_arrival": "job arrival rate",
        "job_completion": "job completion rate",
        "flex3_ratio": "flex3 ratio",
        "flex2_ratio": "flex2 ratio",
        "total_jobs": "total active jobs",
    }
    display_names = [name_map.get(f, f) for f in sorted_feat]
    fig = go.Figure(go.Bar(
        x=sorted_imp, y=display_names, orientation="h",
        marker_color=COLOR_ACCENT, hovertemplate="%{x:.3f}<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed")
    apply_layout(fig, height=480, xaxis_title="Gini importance", yaxis_title="")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.markdown(
        "Calendar features alone account for **77% of the GBM's predictive power** "
        "(`hour_of_day`: 60%, `day_of_week`: 17%). Lag and rolling statistics add an "
        "incremental ~8%; flex composition ratios contribute the rest. This finding "
        "explains why LSTM and TCN close the gap with GBM despite the data scarcity — "
        "both learn the same daily / weekly periodicity, one through explicit features, "
        "the other from the raw sequence. **Deep models become advantageous when** "
        "(a) longer traces are available, (b) raw sequences replace feature engineering, "
        "or (c) uncertainty estimates are required for risk-aware dispatch."
    )

    # ─── 5. Uncertainty quantification ───────────────────────────────────────
    st.markdown("## 5. Uncertainty Quantification")
    st.markdown(
        "Cost-aware dispatch is asymmetric: under-prediction of capacity is more costly "
        "(deferred jobs miss their cheap window) than over-prediction. We trained a "
        "**Quantile LSTM** with the pinball loss at the 10th, 50th, and 90th percentiles "
        "to produce calibrated prediction intervals."
    )
    qa, qb, qc = st.columns(3)
    with qa: kpi_card("Empirical coverage", "58.5%", "vs 80% target",
                      delta="overconfident", delta_pos=False)
    with qb: kpi_card("MAE (q50)", "0.0512", "MW")
    with qc: kpi_card("Avg interval width", "0.107", "MW")
    st.markdown(
        "Coverage on the test window is **58.5% against the 80% target** — the model is "
        "overconfident. Inspection of the prediction plot (not shown here; see "
        "`analysis.ipynb`) reveals that failures concentrate at sudden load spikes — "
        "burst job submissions that violate the smooth daily pattern. The current "
        "Quantile LSTM is unfit for risk-critical dispatch; future work is to condition "
        "interval width on recent volatility (e.g., rolling std as an additional feature) "
        "or to apply post-hoc conformal calibration."
    )

    # ─── 6. From prediction to dispatch ──────────────────────────────────────
    st.markdown("## 6. From Prediction to Dispatch")
    st.markdown(
        "We simulate a **greedy carbon- and cost-aware scheduler** over the synthetic "
        "test window. At each 15-minute bin, the scheduler pulls the GBM 8-hour forecast, "
        "identifies bins where current power exceeds the 75th percentile threshold (peak), "
        "and defers a fraction of Flex 2 / 3 jobs from peak to predicted off-peak windows, "
        "with a $5,000/MW penalty for any window overlapping a 4CP candidate bin."
    )
    da, db, dc = st.columns(3)
    with da: kpi_card("Peak power reduction", "18.7%", "MW (-0.27)", delta="0.27 MW avg drop", delta_pos=True)
    with db: kpi_card("Energy shifted", "9.4", "MW·h from peak")
    with dc: kpi_card("Deferral events", "160", "over 138 hours")
    st.markdown(
        "Maximum *instantaneous* power increased (1.72 MW → 2.02 MW) because deferred jobs "
        "cluster into the same off-peak window — a known peak-shifting side effect, "
        "addressable in Phase 2 with capacity-aware MILP scheduling that explicitly limits "
        "simultaneous starts per bin."
    )

    # ─── 7. Limitations & roadmap ────────────────────────────────────────────
    st.markdown("## 7. Limitations & Roadmap")
    st.markdown(
        "We deploy **GBM** (not the deep models) in the application because of "
        "(a) interpretability, (b) sub-millisecond inference, and (c) robustness under "
        "data scarcity. The Quantile LSTM is retained as a research artifact for future "
        "uncertainty-aware extensions.\n\n"
        "**Known limitations:**"
    )
    st.markdown(
        "- **Synthetic data extension.** 7 days of real Alibaba data were augmented to 30 days. "
        "Deep-model results carry the calibration assumptions; cross-validation on additional "
        "real traces is needed before stronger empirical claims.\n"
        "- **LMP is synthetic-but-calibrated** to ERCOT 2023 distributional statistics, not live. "
        "Production deployment requires integration with ERCOT's MIS / EMP6 endpoints "
        "(or a gridstatus.io wrapper).\n"
        "- **No demand response market access.** DR revenue requires Qualified Scheduling Entity "
        "(QSE) relationships and multi-quarter procurement; the savings shown here represent only "
        "energy arbitrage + 4CP avoidance.\n"
        "- **Greedy scheduling, not optimal.** Phase 2 replaces the greedy heuristic with a "
        "MILP formulation (PuLP / CVXPY / Gurobi) for cluster-wide global optimization across "
        "multiple campuses, capacity constraints, and inter-job dependencies."
    )

    # ─── References ──────────────────────────────────────────────────────────
    st.markdown("## References")
    st.markdown(
        '<div class="mtd-citation">[1] Q. Weng et al., "MLaaS in the Wild: Workload Analysis '
        'and Scheduling in Large-Scale Heterogeneous GPU Clusters," <em>NSDI</em>, 2022.</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="mtd-citation">[2] A. Radovanović et al., "Carbon-Aware Computing for '
        'Datacenters," <em>IEEE Transactions on Power Systems</em>, 2023.</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="mtd-citation">[3] ERCOT, "Nodal Protocols Section 4: Scheduling, '
        'Operations Planning and Reliability Unit Commitment," 2024.</div>',
        unsafe_allow_html=True,
    )

    # ── Footer ──────────────────────────────────────────────────────────────
    st.markdown("---")
    st.caption(
        "Data: Alibaba PAI GPU cluster (2020) · ERCOT 2023-calibrated LMP & carbon intensity · "
        "Forecaster: Gradient Boosting Regressor · Scheduler: greedy with 4CP penalty"
    )
    st.stop()

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

    run_opt = st.button("Optimize submission window", type="primary")

    if run_opt:
        flex_level    = int(flex_type[5])
        duration_bins = max(1, int(np.ceil(duration_hr * 60 / BIN_MIN)))
        job_mw        = gpu_count * (P_MAX - P_IDLE) / 1e6
        max_defer     = {1: 0, 2: 8, 3: 32}[flex_level]

        N_FC = len(forecast)
        candidate_starts = np.arange(0, min(max_defer, N_FC - 1) + 1)
        costs, carbs = [], []
        for s in candidate_starts:
            end = min(s + duration_bins, N_FC)
            costs.append(job_mw * fc_lmp[s:end].sum() * (BIN_MIN / 60))
            carbs.append(job_mw * 1000 * fc_carb[s:end].sum() * (BIN_MIN / 60) / 1e6)
        costs, carbs = np.array(costs), np.array(carbs)
        best_step = 0 if len(costs) == 0 else int(candidate_starts[np.argmin(costs)])
        best_delay = best_step * BIN_MIN
        now_end    = min(duration_bins, N_FC)
        best_end   = min(best_step + duration_bins, N_FC)
        now_cost   = float(job_mw * fc_lmp[0:now_end].sum() * (BIN_MIN / 60))
        best_cost  = float(costs[best_step]) if len(costs) > best_step else now_cost
        now_carb   = float(job_mw * 1000 * fc_carb[0:now_end].sum() * (BIN_MIN / 60) / 1e6)
        best_carb  = float(carbs[best_step]) if len(carbs) > best_step else now_carb

        target_h   = (now_hour + best_delay / 60) % 24
        target_str = f"Day {day}, {int(target_h):02d}:{int((target_h % 1) * 60):02d}"

        st.session_state.advisor_result = dict(
            snapshot_day=day, snapshot_hhmm=hhmm,
            gpus=gpu_count, duration_hr=duration_hr, flex_label=flex_type.split(" — ")[0],
            best_step=best_step, best_delay=best_delay, target_str=target_str,
            now_end=now_end, best_end=best_end,
            now_cost=now_cost, best_cost=best_cost,
            now_carb=now_carb, best_carb=best_carb,
            now_bin=nb, fc_lmp=fc_lmp, fc_4cp=fc_4cp, fc_carb=fc_carb,
            forecast=forecast, forecast_lo=forecast_lo, forecast_hi=forecast_hi,
            hist_start=hist_start, now_hour=now_hour,
        )

    result = st.session_state.advisor_result
    if result is None:
        st.info("Set job parameters and click **Optimize submission window** to compute a recommendation.")
    else:
        r = result
        cost_save = r["now_cost"] - r["best_cost"]
        carb_save = r["now_carb"] - r["best_carb"]
        rec_label = "Submit now" if r["best_step"] == 0 else f"Defer {r['best_delay']} min · start {r['target_str']}"

        st.markdown("### Recommendation")
        st.caption(
            f"Computed at Day {r['snapshot_day']}, {r['snapshot_hhmm']} · "
            f"{r['gpus']} GPUs · {r['duration_hr']:.1f} hr · {r['flex_label']}"
        )

        action_banner("Action", rec_label)

        c1, c2, c3 = st.columns(3)
        with c1: kpi_card("Energy cost", f"${r['best_cost']:,.2f}",
                          delta=f"saves ${cost_save:,.2f}" if cost_save > 0.01 else None, delta_pos=True)
        with c2: kpi_card("CO₂ emitted", f"{r['best_carb']*1000:,.1f}", "kg",
                          delta=f"saves {carb_save*1000:.1f} kg" if carb_save > 0.01 else None, delta_pos=True)
        with c3: kpi_card("4CP exposure",
                          f"{int(r['fc_4cp'][r['best_step']:r['best_end']].sum())}/"
                          f"{int(r['fc_4cp'][:r['now_end']].sum())}",
                          "bins (best / now)")

        st.markdown("")
        col_left, col_right = st.columns(2)

        # ── Left: Flex capacity forecast ────────────────────────────────────
        with col_left:
            st.markdown("### Cluster flex-capacity forecast")
            hist = ts.iloc[r["hist_start"]:r["now_bin"]+1]
            hist_h = hist["hour"].values - r["now_hour"]

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=np.concatenate([forecast_hours, forecast_hours[::-1]]),
                y=np.concatenate([r["forecast_hi"], r["forecast_lo"][::-1]]),
                fill="toself", fillcolor="rgba(37, 99, 235, 0.12)",
                line=dict(width=0), hoverinfo="skip", showlegend=False,
            ))
            fig.add_trace(go.Scatter(
                x=hist_h, y=hist["flexible_mw"].values,
                mode="lines", line=dict(color=COLOR_INK, width=1.5),
                name="Historical", hovertemplate="%{y:.3f} MW<extra></extra>",
            ))
            fig.add_trace(go.Scatter(
                x=forecast_hours, y=r["forecast"],
                mode="lines", line=dict(color=COLOR_ACCENT, width=2, dash="dash"),
                name="Forecast", hovertemplate="%{y:.3f} MW<extra></extra>",
            ))
            fig.add_vline(x=0, line_width=1, line_dash="dot", line_color=COLOR_MUTED)
            apply_layout(fig, height=320,
                         yaxis_title="Flex capacity (MW)",
                         xaxis_title="Hours from now",
                         xaxis=dict(range=[-24, 8]))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        # ── Right: LMP forecast with windows ────────────────────────────────
        with col_right:
            st.markdown("### LMP forecast — submission windows")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=forecast_hours, y=r["fc_lmp"],
                mode="lines", line=dict(color=COLOR_INK, width=1.5),
                name="LMP", hovertemplate="$%{y:.2f}/MWh<extra></extra>",
            ))
            cp_mask = r["fc_4cp"]
            if cp_mask.any():
                fig.add_trace(go.Scatter(
                    x=forecast_hours[cp_mask], y=r["fc_lmp"][cp_mask],
                    mode="markers", marker=dict(color=COLOR_ACCENT, size=10, symbol="diamond-open"),
                    name="4CP", hovertemplate="4CP candidate<extra></extra>",
                ))
            fig.add_vrect(x0=0, x1=forecast_hours[r["now_end"] - 1],
                          fillcolor=COLOR_MUTED, opacity=0.12, line_width=0,
                          annotation_text="if now", annotation_position="top left",
                          annotation_font_size=10, annotation_font_color=COLOR_MUTED)
            if r["best_step"] > 0:
                fig.add_vrect(x0=forecast_hours[r["best_step"]],
                              x1=forecast_hours[r["best_end"] - 1],
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
