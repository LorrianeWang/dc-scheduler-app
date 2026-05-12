# Carbon- and Cost-Aware GPU Workload Scheduler

**Live application:** <https://dc-scheduler-app-cev64kudmw5qdne6ihibdt.streamlit.app>

---

## Abstract

We present a carbon- and cost-aware scheduling system for GPU data centers
that integrates (i) a workload-classification of approximately 91,000 jobs
from the Alibaba PAI v2020 production trace into three flexibility tiers,
(ii) a Gradient Boosting forecaster of cluster deferrable capacity at 15-minute
resolution four hours ahead, and (iii) a greedy dispatch heuristic with a
Four Coincident Peak (4CP) penalty term aligned to ERCOT's transmission cost
allocation methodology. On a 138-hour test window with ERCOT 2023-calibrated
Locational Marginal Prices (LMPs), the system reduces average peak power by
**18.7 %** and shifts **9.4 MW·h** of energy from peak to off-peak windows
across 160 deferral events.

We benchmark seven forecasters — Persistence, Linear Regression, Gradient
Boosting, two LSTM variants, a Temporal Convolutional Network, and a
Quantile LSTM trained with pinball loss — and find that gradient boosting
with hand-crafted lag features (MAE 0.0448 MW) marginally outperforms deep
sequence models at this data scale, while the Quantile LSTM provides
asymmetric uncertainty estimates suitable for risk-aware dispatch. An
interactive Streamlit application demonstrates both single-job (Advisor) and
queue-level (Planner) scheduling modes with live ERCOT-calibrated grid
signals.

---

## 1. Motivation

Hyperscale GPU clusters now consume hundreds of megawatts of electricity.
Within the Alibaba PAI production trace, approximately 50 % of submitted
jobs are deferrable — training, batch processing, and background workloads
with no tight latency requirements. In wholesale electricity markets such as
ERCOT, deferrable load creates three monetisable opportunities:

1. **Energy arbitrage.** Real-time LMPs swing from **$30/MWh** to **$500+/MWh**
   within hours during summer scarcity events.
2. **4CP avoidance.** Industrial customers are billed for transmission service
   based on the average of their consumption during the four monthly system
   peaks (one per June, July, August, and September). Avoiding one megawatt
   during 4CP windows yields roughly **$60–100/kW·year** in savings.
3. **Demand response revenue.** Ancillary-services programmes (ERS, RRS, ECRS)
   pay capacity for committed curtailment.

This work asks whether a learned forecaster of cluster deferrable capacity,
combined with a cost- and carbon-aware scheduler, can capture this value.

---

## 2. Data

### 2.1 Workload Trace
We use the Alibaba PAI v2020 GPU-cluster trace [1]: 7 days, 6,500 GPUs across
T4, P100, V100 and V100M32 SKUs, ~91k jobs. We join seven released tables
(`job`, `task`, `instance`, `sensor`, `group_tag`, `machine_spec`,
`machine_metric`) to obtain per-job runtime, resource request, and per-worker
GPU utilisation.

### 2.2 Flexibility Tiers
Each job is classified into one of three tiers by `task_name` and runtime:

| Tier | Deferrability | Members |
| :--: | :--: | :-- |
| Flex 1 | 0 % | `evaluator`, `TensorboardTask` (real-time monitoring) |
| Flex 2 | 50 % | `worker`, `ps`, `PyTorchWorker`, `DecoderWorker`, etc. with runtime < 24 h |
| Flex 3 | 100 % | identical task types with runtime ≥ 24 h, plus batch transfer workers |

### 2.3 Synthetic Extension
The 7-day real trace is insufficient for sequence-model training. We
generate 30 days of synthetic time-series data calibrated to the real
distribution (daily/weekly seasonality, flex-class ratios, spike statistics).
Real data drives workload characterisation and the scheduler's structural
assumptions; synthetic data is used exclusively for sequence-model training
and validation.

### 2.4 Grid Signals
Two ERCOT 2023-calibrated time series accompany the workload:

- **LMP** (mean $33/MWh, median $23, p95 $40, p99 $460, occasional negative
  values during high-wind nights).
- **Marginal carbon intensity** (mean 381 gCO₂/kWh, range 212–749 g/kWh).
- Four bins labelled as 4CP candidates (top-LMP afternoons across distinct
  days).

These signals are bundled as static CSVs; the production deployment path
substitutes them with live ERCOT API or `gridstatus.io` feeds.

---

## 3. Forecasting Models

The forecasting task is: given history `t-96 ... t` (24 h at 15-min
resolution), predict cluster deferrable capacity (`flexible_mw`) at horizon
`t+16` (4 hours ahead). Train/test split is 80/20, temporally ordered.

| Model | MAE (MW) | RMSE (MW) | MAPE |
| :-- | :--: | :--: | :--: |
| Persistence (last-value) | 0.0885 | 0.1104 | 20.5 % |
| Linear Regression | 0.0539 | 0.0715 | 12.2 % |
| **Gradient Boosting (deployed)** | **0.0448** | **0.0607** | **10.0 %** |
| LSTM, 2-layer, 50 epoch | 0.0530 | 0.0690 | 12.2 % |
| LSTM, 2-layer, 100 epoch + LR sched | 0.0539 | 0.0732 | 11.8 % |
| TCN, dilations {1,2,4} | 0.0547 | 0.0734 | 12.3 % |
| Quantile LSTM (q50), pinball loss | 0.0505 | 0.0697 | 10.9 % |

We hypothesised that deep sequence models would outperform classical
methods. Empirically, gradient boosting with hand-crafted lag features
attains the lowest test MAE. Feature-importance analysis reveals that
calendar features alone (`hour_of_day`, `day_of_week`) account for **77 %**
of GBM's predictive power — explaining why both LSTM and TCN narrow but do
not close the gap, as both ultimately learn the same diurnal periodicity
that the hand-crafted features encode explicitly.

The Quantile LSTM achieves **61.4 % empirical coverage against an 80 %
target**, indicating overconfident interval estimates. Failure modes
concentrate at sudden load spikes; future work is to condition interval
width on recent volatility or to apply conformal calibration post hoc.

---

## 4. Dispatch Simulation

A greedy scheduler is evaluated on the 138-hour test window. At each
15-minute bin the scheduler

1. retrieves the GBM 8-hour forecast,
2. identifies bins above the 75-th percentile of recent power as peak,
3. defers a fraction of Flex 2 / 3 jobs from peak to the lowest-predicted-load
   bin within their flexibility budget,
4. applies a $5,000/MW penalty to any window overlapping a 4CP candidate bin.

Results on the test window:

| Metric | Value |
| :-- | --: |
| Average peak power reduction | 18.7 % (−0.27 MW) |
| Energy shifted from peak | 9.4 MW·h |
| Deferral events | 160 |
| Cumulative job-deferrals | 685,439 |

The maximum *instantaneous* power increased from 1.72 MW to 2.02 MW because
deferred jobs cluster into the same off-peak window — a known peak-shifting
side effect, addressable in Phase 2 with capacity-aware mixed-integer linear
programming (MILP).

---

## 5. Application

The Streamlit application offers two modes plus an embedded methodology
page.

### 5.1 Advisor — Single Job
Inputs: GPU count, estimated duration (h), flexibility class (1/2/3).
On clicking *Optimize submission window* the application returns the
recommended start time, expected energy cost ($), CO₂ emission delta (kg),
and 4CP exposure (bins).

### 5.2 Planner — Job Queue
Inputs: a CSV (`job_id, gpus, duration_hr, flex_class, deadline_hr`) or a
12-job sample queue. The greedy scheduler returns a Gantt-style dispatch
plan, summary KPIs (cost without scheduling, cost with scheduling, CO₂
avoided, jobs deferred), and a downloadable CSV.

### 5.3 Methodology & Validation
A seven-section narrative reproducing this README in-app, with an
interactive multiselect over the seven forecasters described in §3.
Persistence / Linear / GBM are computed live from the deployed model; LSTM
/ TCN / Quantile LSTM predictions are bundled from
`data/model_predictions.csv`.

---

## 6. Repository Layout

```
dc-scheduler-app/
├── app.py                                # Streamlit application
├── requirements.txt
├── data/
│   ├── ts_flex_30day_synthetic.csv       # 30-day synthetic workload trace
│   ├── ercot_lmp_30day.csv               # LMP + carbon intensity + 4CP flags
│   └── model_predictions.csv             # Deep-model test-set predictions
├── models/
│   ├── gbr_model.joblib                  # Deployed Gradient Boosting model
│   ├── scaler.joblib
│   └── feature_cols.joblib
└── scripts/
    └── generate_model_predictions.py     # Re-train deep models, regen CSV
```

---

## 7. Reproducibility

```bash
git clone https://github.com/LorrianeWang/dc-scheduler-app.git
cd dc-scheduler-app
pip install -r requirements.txt
streamlit run app.py
```

To regenerate the bundled deep-model predictions (≈20 minutes on a 2020
MacBook CPU):

```bash
python scripts/generate_model_predictions.py
```

Random seeds are fixed (`torch.manual_seed(42)`, `numpy.random.seed(42)`),
but deep-learning results remain sensitive to platform-level non-determinism.
The MAE values in §3 correspond to the bundled CSV.

---

## 8. Limitations and Roadmap

We deploy gradient boosting (not the deep models) in the application because
of (i) interpretability, (ii) sub-millisecond inference, and (iii)
robustness under data scarcity. The Quantile LSTM is retained as a research
artifact for future uncertainty-aware extensions.

**Known limitations.**
* **Synthetic data extension.** Sequence-model results carry calibration
  assumptions of the 7-day-to-30-day augmentation.
* **Synthetic LMP and carbon series.** Distributionally calibrated to ERCOT
  2023 statistics but not live. Production deployment requires integration
  with ERCOT MIS / EMP6 endpoints.
* **No demand-response market access.** DR revenue requires Qualified
  Scheduling Entity (QSE) relationships and multi-quarter procurement;
  reported savings represent only energy arbitrage and 4CP avoidance.
* **Greedy heuristic.** Phase 2 replaces the greedy scheduler with a MILP
  formulation (PuLP / CVXPY / Gurobi) for cluster-wide global optimisation.

---

## References

[1] Q. Weng, W. Xiao, Y. Yu, W. Wang, C. Wang, J. He, Y. Li, L. Zhang,
W. Lin, and Y. Ding. *MLaaS in the Wild: Workload Analysis and Scheduling
in Large-Scale Heterogeneous GPU Clusters.* Proceedings of the 19th USENIX
Symposium on Networked Systems Design and Implementation (NSDI), 2022.

[2] A. Radovanović, R. Koningstein, I. Schneider, B. Chen, A. Duarte, B. Roy,
D. Xiao, M. Haridasan, P. Hung, N. Care, S. Talukdar, E. Mullen, K. Smith,
M. Cottman, and W. Cirne. *Carbon-Aware Computing for Datacenters.* IEEE
Transactions on Power Systems, 38(2), 2023.

[3] Electric Reliability Council of Texas (ERCOT). *Nodal Protocols Section
4: Scheduling, Operations Planning and Reliability Unit Commitment.* 2024.
