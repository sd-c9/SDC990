"""
preprocessor.py — Feature engineering pipeline for SB-Guard.

Turns raw simulated/field telemetry into the canonical 14-feature vector used
by both models. Handles missing-data imputation, derives the engineered
indicators (rolling anomaly score, Misalignment Severity Index, Stuffing Box
friction coefficient) and exposes a scaler that is persisted with the models.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import config as cfg


RAW_SENSOR_COLS = ["SPM", "PPRL", "MPRL", "PRLR", "SCA", "pump_fillage",
                   "LVI", "sb_friction", "TDS", "gearbox_temp",
                   "rod_stretch", "FLP"]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing samples per well using time-aware interpolation."""
    df = df.sort_values(["well_id", "timestamp"]).copy()
    df[RAW_SENSOR_COLS] = (
        df.groupby("well_id")[RAW_SENSOR_COLS]
          .transform(lambda g: g.interpolate(limit_direction="both")
                                 .ffill().bfill())
    )
    # Any column still fully empty -> fill with column median.
    df[RAW_SENSOR_COLS] = df[RAW_SENSOR_COLS].fillna(df[RAW_SENSOR_COLS].median())
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive the engineered indicators and the final FEATURE_COLUMNS vector.

    Operates per well so rolling windows do not bleed across wells.
    """
    df = clean(df)
    out = []
    win_24h = cfg.SAMPLES_PER_DAY  # rolling 24h window length

    for wid, g in df.groupby("well_id", sort=False):
        g = g.copy()

        # PRLR may have been corrupted by realism injection — recompute clean.
        g["PRLR"] = g["PPRL"] - g["MPRL"]

        # Stuffing Box friction coefficient (derived, normalized):
        # higher load range relative to fluid load + raw friction proxy.
        g["sb_friction"] = (g["sb_friction"]
                            + 0.15 * (g["PRLR"] / (g["FLP"].abs() + 1.0)))

        # Rolling 24h anomaly score: z-score magnitude of vibration + friction
        # against each well's own recent baseline.
        for col in ("LVI", "sb_friction"):
            roll = g[col].rolling(win_24h, min_periods=cfg.SAMPLES_PER_HOUR)
            g[f"_{col}_z"] = (g[col] - roll.mean()) / (roll.std() + 1e-6)
        g["rolling_anomaly_24h"] = (
            g[["_LVI_z", "_sb_friction_z"]].abs().mean(axis=1).fillna(0.0)
        )

        # Misalignment Severity Index (MSI) — custom composite of the three
        # mechanical misalignment signatures, scaled to roughly 0..1.
        lvi_n = _minmax(g["LVI"])
        tds_n = _minmax(g["TDS"])
        fric_n = _minmax(g["sb_friction"])
        asym = _minmax((g["PPRL"] - g["PPRL"].rolling(win_24h, min_periods=6).median()).abs())
        g["MSI"] = np.clip(0.40 * lvi_n + 0.25 * tds_n
                           + 0.20 * fric_n + 0.15 * asym, 0, 1)

        g.drop(columns=["_LVI_z", "_sb_friction_z"], inplace=True)
        out.append(g)

    res = pd.concat(out, ignore_index=True)
    # Guarantee every canonical feature exists.
    for c in cfg.FEATURE_COLUMNS:
        if c not in res.columns:
            res[c] = 0.0
    return res


def _minmax(s: pd.Series) -> pd.Series:
    lo, hi = s.min(), s.max()
    if hi - lo < 1e-9:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - lo) / (hi - lo)


def fit_scaler(df: pd.DataFrame) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(df[cfg.FEATURE_COLUMNS].values)
    return scaler


def transform(df: pd.DataFrame, scaler: StandardScaler) -> np.ndarray:
    return scaler.transform(df[cfg.FEATURE_COLUMNS].values)


def build_sequences(values: np.ndarray,
                    labels: np.ndarray,
                    window: int = cfg.LSTM_WINDOW,
                    stride: int = 1):
    """
    Slice a single well's scaled matrix into overlapping LSTM windows.

    `labels` is an (n, 3) array of the 24/48/72h multi-horizon targets aligned
    to the *last* time step of each window.
    """
    X, y = [], []
    n = len(values)
    for end in range(window, n + 1, stride):
        X.append(values[end - window:end])
        y.append(labels[end - 1])
    if not X:
        return np.empty((0, window, values.shape[1])), np.empty((0, labels.shape[1]))
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)
