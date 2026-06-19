"""
isolation_forest.py — Unsupervised anomaly detector for SB-Guard.

An Isolation Forest trained exclusively on NORMAL-operation samples. It scores
each time step's deviation from healthy fleet behaviour and complements the
supervised LSTM in the ensemble.

    Contamination : 0.05
    Features      : the 13 engineered features (all except raw SPM set-point)
    Output        : normalized anomaly score in [0, 1] (1 == most anomalous)
"""

from __future__ import annotations

import os
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest

import config as cfg


class AnomalyDetector:
    def __init__(self):
        self.model = IsolationForest(
            n_estimators=cfg.IF_N_ESTIMATORS,
            contamination=cfg.IF_CONTAMINATION,
            random_state=cfg.RANDOM_SEED,
            n_jobs=-1,
        )
        self._score_min = None
        self._score_max = None

    def fit(self, X_normal: np.ndarray):
        """Fit on normal-operation feature rows only."""
        self.model.fit(X_normal)
        raw = self.model.score_samples(X_normal)   # higher == more normal
        self._score_min = float(raw.min())
        self._score_max = float(raw.max())
        return self

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """
        Return a normalized anomaly score in [0, 1] where 1 is most anomalous.
        Inverts sklearn's score_samples (which is high for normal points).
        """
        raw = self.model.score_samples(X)
        lo, hi = self._score_min, self._score_max
        if hi is None or hi - lo < 1e-9:
            norm = 1.0 / (1.0 + np.exp(raw))
        else:
            norm = (hi - raw) / (hi - lo)           # invert + scale
        return np.clip(norm, 0.0, 1.0)

    def normal_score(self, X: np.ndarray) -> np.ndarray:
        """Normalized "normality" in [0,1] (1 == perfectly normal)."""
        return 1.0 - self.anomaly_score(X)

    def save(self, directory: str = cfg.SAVED_MODELS_DIR):
        os.makedirs(directory, exist_ok=True)
        joblib.dump(
            {"model": self.model, "min": self._score_min, "max": self._score_max},
            os.path.join(directory, "isolation_forest.joblib"),
        )

    def load(self, directory: str = cfg.SAVED_MODELS_DIR):
        blob = joblib.load(os.path.join(directory, "isolation_forest.joblib"))
        self.model = blob["model"]
        self._score_min = blob["min"]
        self._score_max = blob["max"]
        return self

    @staticmethod
    def exists(directory: str = cfg.SAVED_MODELS_DIR) -> bool:
        return os.path.exists(os.path.join(directory, "isolation_forest.joblib"))
