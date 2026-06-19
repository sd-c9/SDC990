"""
ensemble.py — Combine LSTM + Isolation Forest into a single risk score.

    combined = 0.65 * LSTM_prob + 0.35 * (1 - IF_normalized_score)

where LSTM_prob is the 24h-horizon failure probability (the operationally
actionable window) and IF_normalized_score is the detector's normality. An
alert fires when combined > ALERT_THRESHOLD (0.70).
"""

from __future__ import annotations

import numpy as np

import config as cfg


def combine(lstm_prob_24h: np.ndarray,
            if_anomaly_score: np.ndarray,
            w_lstm: float = cfg.ENSEMBLE_W_LSTM,
            w_if: float = cfg.ENSEMBLE_W_IF) -> np.ndarray:
    """
    Vectorized ensemble combination.

    `if_anomaly_score` is already the anomaly score in [0,1] (1 == anomalous),
    i.e. (1 - normality). The spec's `(1 - IF_normalized_score)` equals this
    anomaly score directly.
    """
    lstm_prob_24h = np.asarray(lstm_prob_24h, dtype=float)
    if_anomaly_score = np.asarray(if_anomaly_score, dtype=float)
    return np.clip(w_lstm * lstm_prob_24h + w_if * if_anomaly_score, 0.0, 1.0)


def severity_band(score: float,
                  alert: float = cfg.ALERT_THRESHOLD,
                  warn: float = cfg.WARNING_THRESHOLD) -> str:
    if score >= alert:
        return "CRITICAL"
    if score >= warn:
        return "WARNING"
    return "NORMAL"


def status_color(band: str) -> str:
    return {"CRITICAL": "red", "WARNING": "yellow", "NORMAL": "green"}.get(band, "green")


def is_alert(score: float, threshold: float = cfg.ALERT_THRESHOLD) -> bool:
    return score >= threshold
