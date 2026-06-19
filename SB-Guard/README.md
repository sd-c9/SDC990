# SB-Guard 🛢️
### AI-Powered Stuffing Box Leak Prediction System for Rod Pump Wells

> Predictive-maintenance decision-support for **Petroleum Development Oman (PDO)**
> beam-lift (rod pump) fleet. SB-Guard detects polished-rod **misalignment** —
> the root cause of Stuffing Box (SB) leaks — up to **72 hours** before failure,
> preventing environmental releases and unplanned production downtime.

---

## 1. What it does

Stuffing Box leaks caused by polished-rod misalignment are a critical, largely
unsolved problem across PDO's 1,700+ beam-lift wells. SB-Guard fuses two
complementary models into a single risk score:

| Model | Role |
|-------|------|
| **LSTM sequence classifier** | Learns temporal degradation patterns; outputs multi-horizon failure probability (24 h / 48 h / 72 h) |
| **Isolation Forest** | Unsupervised anomaly detector trained on *normal* operation only |

**Ensemble:** `combined = 0.65 · LSTM_24h + 0.35 · IF_anomaly`
An alert is raised when `combined > 0.70`.

The system engineers **14 features** from Dynacard + sensor telemetry, including
the custom **Misalignment Severity Index (MSI)** and a derived **Stuffing Box
Friction Coefficient**.

---

## 2. Quick start

```bash
cd SB-Guard
python -m venv .venv && source .venv/bin/activate     # optional
pip install -r requirements.txt

# One-shot: generate data, train models, launch UI, open browser
python run.py
```

`run.py` trains automatically on first launch (a few minutes on CPU), then
serves the console at **http://127.0.0.1:5000**.

### Run the steps manually

```bash
python -m data.dataset_generator   # 1) generate synthetic fleet  -> data/generated/
python train.py                    # 2) train LSTM + Isolation Forest, score fleet
python app.py                      # 3) launch Flask app
```

---

## 3. The five console pages

1. **Dashboard** — fleet status grid (green/yellow/red), KPI counters (active
   alerts, wells monitored, failures prevented, cost saved), live 24 h fleet-health chart.
2. **Well Inspector** — interactive Dynacard (current vs baseline overlay),
   SPM / vibration / friction / risk time-series, MSI gauge, 24/48/72 h
   probability bars, auto-generated maintenance recommendation, **PDF export**.
3. **Alerts & History** — active alert table (lead time, recommended action),
   historical alerts with outcome tracking, **Excel export**.
4. **Analytics** — live model metrics (Precision/Recall/F1, ROC-AUC, MTTD, FAR,
   lead time), ROC curve, confusion matrix, feature importance, monthly failure
   trend, most-at-risk ranking, cost-savings estimate.
5. **Model Management** — retrain button with progress bar, threshold slider,
   model version history, data-quality report.

The UI runs a **real-time simulation**: a background thread advances the fleet
playhead every 30 seconds (`REALTIME_REFRESH_SEC`), and the front-end polls for
fresh scores.

---

## 4. Project structure

```
SB-Guard/
├── app.py                  Flask application (pages + JSON API + exports)
├── run.py                  Launcher (auto-train + open browser)
├── train.py                End-to-end training & fleet-scoring pipeline
├── config.py               All tunable constants (single source of truth)
├── requirements.txt
├── data/
│   ├── simulator.py        Synthetic rod-pump telemetry generator
│   ├── preprocessor.py     Feature engineering (14-feature vector, MSI, …)
│   └── dataset_generator.py  Build + persist the fleet dataset
├── models/
│   ├── lstm_model.py       Multi-horizon LSTM (Keras) + graceful fallback
│   ├── isolation_forest.py Anomaly detector
│   ├── ensemble.py         Ensemble combination + severity banding
│   └── evaluator.py        Metrics: P/R/F1, ROC-AUC, MTTD, FAR, lead time, CM
├── core/
│   ├── predictor.py        Real-time serving engine (playhead, dynacard, KPIs)
│   ├── alert_manager.py    Alert lifecycle + bilingual recommendations
│   └── cost_calculator.py  Savings estimation
├── reports/
│   └── report_generator.py PDF (per well) + Excel (alerts) export
├── static/css/style.css    PDO dark-navy + orange industrial theme
├── static/js/charts.js     Plotly chart factory + page controllers
├── templates/              base + 5 pages
├── saved_models/           trained weights, scaler, metrics, registry
├── data/generated/         synthetic CSV/Parquet datasets
├── logs/                   alert log (alerts.jsonl)
├── reports/                generated PDF reports
└── docs/                   screenshots
```

---

## 5. Engineered features (14)

| # | Feature | Description |
|---|---------|-------------|
| 1 | SCA | Surface Card Area — energy transferred per stroke |
| 2 | PRLR | Polished Rod Load Range |
| 3 | PPRL | Peak Polished Rod Load |
| 4 | MPRL | Minimum Polished Rod Load |
| 5 | pump_fillage | Pump Fillage % |
| 6 | SPM | Strokes Per Minute |
| 7 | LVI | Lateral Vibration Index (key misalignment indicator) |
| 8 | TDS | Torque Deviation Score |
| 9 | rod_stretch | Rod String Stretch Factor |
| 10 | FLP | Fluid Load on Pump |
| 11 | gearbox_temp | Gearbox Temperature |
| 12 | sb_friction | Stuffing Box Friction Coefficient (derived) |
| 13 | rolling_anomaly_24h | Rolling 24 h anomaly score |
| 14 | MSI | Misalignment Severity Index (custom composite) |

---

## 6. Data simulation

`data/simulator.py` fabricates physically-plausible telemetry for **50 wells**,
**6 months** each at **10-minute** resolution (~1.3 M rows). **30%** of wells
progress through a four-stage degradation toward an SB failure:

```
NORMAL → STAGE 1 (72 h: slight vibration) → STAGE 2 (48 h: LVI spike, SCA change)
        → STAGE 3 (24 h: friction rise, card asymmetry) → FAILURE (load anomaly)
```

The generator injects **noise, sensor drift, missing data and outliers** so the
pipeline is exercised on realistic, messy data.

---

## 7. Configuration

Everything tunable lives in `config.py`: fleet size, simulation horizons, the
14-feature list, LSTM architecture/epochs, Isolation Forest contamination,
ensemble weights, alert thresholds, the cost model and the realtime refresh
interval.

---

## 8. Notes & disclaimer

- Real PDO field data is confidential; **all data here is synthetic**.
- SB-Guard is a **decision-support tool** — predictions are probabilistic;
  confirm with field inspection before intervention.
- If TensorFlow is unavailable, `lstm_model.py` automatically falls back to a
  gradient-boosting sequence classifier so the application still runs end-to-end.

🤖 Built for the PDO predictive-maintenance initiative.
