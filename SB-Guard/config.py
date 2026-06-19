"""
config.py — Central configuration for SB-Guard.

SB-Guard: AI-Powered Stuffing Box Leak Prediction System for Rod Pump Wells.
Designed for Petroleum Development Oman (PDO) beam-lift well fleet.

All tunable constants (paths, simulation parameters, model hyper-parameters,
ensemble weights and alert thresholds) live here so the rest of the codebase
imports a single source of truth.
"""

import os

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DATASET_DIR = os.path.join(BASE_DIR, "data", "generated")
SAVED_MODELS_DIR = os.path.join(BASE_DIR, "saved_models")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

for _d in (DATASET_DIR, SAVED_MODELS_DIR, LOGS_DIR, REPORTS_DIR):
    os.makedirs(_d, exist_ok=True)

# --------------------------------------------------------------------------- #
# Branding
# --------------------------------------------------------------------------- #
APP_NAME = "SB-Guard"
APP_TAGLINE = "AI-Powered Stuffing Box Leak Prediction"
OPERATOR = "Petroleum Development Oman (PDO)"
THEME = {
    "bg_navy": "#0a1628",
    "panel": "#10233d",
    "accent": "#ff6b35",
    "green": "#2ecc71",
    "yellow": "#f1c40f",
    "red": "#e74c3c",
    "text": "#e6edf5",
}

# --------------------------------------------------------------------------- #
# Data simulation
# --------------------------------------------------------------------------- #
N_WELLS = 50
MONTHS_OF_DATA = 6
SAMPLE_INTERVAL_MIN = 10              # 10-minute sampling
SAMPLES_PER_HOUR = 60 // SAMPLE_INTERVAL_MIN
SAMPLES_PER_DAY = SAMPLES_PER_HOUR * 24
DAYS_OF_DATA = MONTHS_OF_DATA * 30
FAILURE_WELL_FRACTION = 0.30          # 30% of wells experience an SB failure
RANDOM_SEED = 42

# Pre-failure degradation horizons (hours before the failure event)
STAGE_HORIZONS_H = {"stage1": 72, "stage2": 48, "stage3": 24}

# Realism injection
MISSING_DATA_RATE = 0.012             # ~1.2% of samples dropped
OUTLIER_RATE = 0.004                  # ~0.4% spurious spikes
SENSOR_DRIFT_MAX = 0.03               # up to 3% slow drift per sensor

# --------------------------------------------------------------------------- #
# Feature engineering — canonical 14-feature vector
# --------------------------------------------------------------------------- #
FEATURE_COLUMNS = [
    "SCA",                 # Surface Card Area
    "PRLR",                # Polished Rod Load Range
    "PPRL",                # Peak Polished Rod Load
    "MPRL",                # Minimum Polished Rod Load
    "pump_fillage",        # Pump Fillage Percentage
    "SPM",                 # Strokes Per Minute
    "LVI",                 # Lateral Vibration Index
    "TDS",                 # Torque Deviation Score
    "rod_stretch",         # Rod String Stretch Factor
    "FLP",                 # Fluid Load on Pump
    "gearbox_temp",        # Gearbox Temperature
    "sb_friction",         # Stuffing Box Friction Coefficient (derived)
    "rolling_anomaly_24h", # Rolling 24h anomaly score
    "MSI",                 # Misalignment Severity Index (custom)
]
# Features fed to the Isolation Forest ("All 13 engineered features" — every
# engineered signal except the raw SPM control set-point).
IF_FEATURES = [c for c in FEATURE_COLUMNS if c != "SPM"]

# --------------------------------------------------------------------------- #
# LSTM model
# --------------------------------------------------------------------------- #
LSTM_WINDOW = 72                      # 72 time steps == 12 h lookback
LSTM_UNITS = [128, 64, 32]
LSTM_DROPOUT = 0.2
LSTM_HORIZONS_H = [24, 48, 72]        # multi-horizon failure prediction
LSTM_EPOCHS = 12
LSTM_BATCH = 256

# --------------------------------------------------------------------------- #
# Isolation Forest
# --------------------------------------------------------------------------- #
IF_CONTAMINATION = 0.05
IF_N_ESTIMATORS = 200

# --------------------------------------------------------------------------- #
# Ensemble + alerting
# --------------------------------------------------------------------------- #
ENSEMBLE_W_LSTM = 0.65
ENSEMBLE_W_IF = 0.35
ALERT_THRESHOLD = 0.70                # combined score above this -> alert
WARNING_THRESHOLD = 0.45              # yellow band

# --------------------------------------------------------------------------- #
# Cost model (USD) — used by core.cost_calculator
# --------------------------------------------------------------------------- #
COST_PER_UNPLANNED_FAILURE = 85000    # workover + cleanup + lost production
COST_PER_ENV_INCIDENT = 40000         # environmental remediation
DEFERRED_OIL_BBL_PER_FAILURE = 220    # barrels deferred per event
OIL_PRICE_USD_PER_BBL = 78.0
PLANNED_INTERVENTION_COST = 12000     # cost of acting on a prediction

# --------------------------------------------------------------------------- #
# Realtime simulation
# --------------------------------------------------------------------------- #
REALTIME_REFRESH_SEC = 30


class Config:
    """Flask application configuration object."""
    SECRET_KEY = os.environ.get("SB_GUARD_SECRET", "sb-guard-pdo-dev-key")
    JSON_SORT_KEYS = False
    TEMPLATES_AUTO_RELOAD = True
