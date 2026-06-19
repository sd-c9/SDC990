"""
lstm_model.py — Multi-horizon LSTM sequence classifier for SB-Guard.

Architecture (per spec):
    Input  : (LSTM_WINDOW=72 time steps, 14 features)  -> 12 h lookback
    Layers : LSTM(128) -> LSTM(64) -> LSTM(32), Dropout(0.2) between
    Head   : Dense(32, relu) -> Dense(3, sigmoid)
    Output : P(failure within 24h / 48h / 72h)
    Loss   : Binary cross-entropy with class weights (severe imbalance)

If TensorFlow is unavailable the class degrades gracefully to a logistic
gradient-boosting fallback so the wider application still runs.
"""

from __future__ import annotations

import os
import numpy as np

import config as cfg

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models, callbacks
    _TF = True
except Exception:                                    # pragma: no cover
    _TF = False


class LSTMLeakClassifier:
    """Wraps the Keras model + persistence + multi-horizon inference."""

    def __init__(self, n_features: int = len(cfg.FEATURE_COLUMNS),
                 window: int = cfg.LSTM_WINDOW):
        self.n_features = n_features
        self.window = window
        self.model = None
        self.history = None
        self.backend = "tensorflow" if _TF else "fallback"

    # ------------------------------------------------------------------ #
    # Build
    # ------------------------------------------------------------------ #
    def build(self):
        if not _TF:
            from sklearn.ensemble import HistGradientBoostingClassifier
            # One classifier per horizon, fed the flattened window.
            self.model = [HistGradientBoostingClassifier(max_iter=120)
                          for _ in cfg.LSTM_HORIZONS_H]
            return self

        m = models.Sequential(name="SB_Guard_LSTM")
        m.add(layers.Input(shape=(self.window, self.n_features)))
        m.add(layers.LSTM(cfg.LSTM_UNITS[0], return_sequences=True))
        m.add(layers.Dropout(cfg.LSTM_DROPOUT))
        m.add(layers.LSTM(cfg.LSTM_UNITS[1], return_sequences=True))
        m.add(layers.Dropout(cfg.LSTM_DROPOUT))
        m.add(layers.LSTM(cfg.LSTM_UNITS[2]))
        m.add(layers.Dropout(cfg.LSTM_DROPOUT))
        m.add(layers.Dense(32, activation="relu"))
        m.add(layers.Dense(len(cfg.LSTM_HORIZONS_H), activation="sigmoid"))
        m.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="binary_crossentropy",
                  metrics=[tf.keras.metrics.AUC(name="auc"),
                           tf.keras.metrics.Precision(name="prec"),
                           tf.keras.metrics.Recall(name="rec")])
        self.model = m
        return self

    # ------------------------------------------------------------------ #
    # Train
    # ------------------------------------------------------------------ #
    def fit(self, X, y, X_val=None, y_val=None,
            epochs: int = cfg.LSTM_EPOCHS, batch: int = cfg.LSTM_BATCH):
        if self.model is None:
            self.build()

        if not _TF:
            Xf = X.reshape(len(X), -1)
            for i, clf in enumerate(self.model):
                clf.fit(Xf, y[:, i])
            return self

        # Class weights — positives are rare. Weight by inverse prevalence on
        # the 24h horizon (the rarest, most important target).
        pos = float(y[:, 0].mean()) + 1e-6
        class_weight = {0: 1.0, 1: float(min(50.0, (1 - pos) / pos))}

        val = (X_val, y_val) if X_val is not None else None
        cbs = [callbacks.EarlyStopping(monitor="loss", patience=3,
                                       restore_best_weights=True)]
        self.history = self.model.fit(
            X, y, validation_data=val, epochs=epochs, batch_size=batch,
            class_weight=class_weight, verbose=2, callbacks=cbs,
        )
        return self

    # ------------------------------------------------------------------ #
    # Predict
    # ------------------------------------------------------------------ #
    def predict_proba(self, X) -> np.ndarray:
        """Return (n, 3) probabilities for the 24/48/72h horizons."""
        if self.model is None:
            raise RuntimeError("Model not trained/loaded.")
        if not _TF:
            Xf = X.reshape(len(X), -1)
            cols = [clf.predict_proba(Xf)[:, 1] for clf in self.model]
            return np.vstack(cols).T
        return self.model.predict(X, verbose=0)

    def predict_latest(self, window_seq: np.ndarray) -> dict:
        """Score a single (window, n_features) sequence -> horizon dict."""
        p = self.predict_proba(window_seq[None, ...])[0]
        return {f"{h}h": float(p[i]) for i, h in enumerate(cfg.LSTM_HORIZONS_H)}

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, directory: str = cfg.SAVED_MODELS_DIR):
        os.makedirs(directory, exist_ok=True)
        if not _TF:
            import joblib
            joblib.dump(self.model, os.path.join(directory, "lstm_fallback.joblib"))
        else:
            self.model.save(os.path.join(directory, "lstm_model.keras"))

    def load(self, directory: str = cfg.SAVED_MODELS_DIR):
        if not _TF:
            import joblib
            self.model = joblib.load(os.path.join(directory, "lstm_fallback.joblib"))
        else:
            self.model = models.load_model(
                os.path.join(directory, "lstm_model.keras"))
        return self

    @staticmethod
    def exists(directory: str = cfg.SAVED_MODELS_DIR) -> bool:
        return (os.path.exists(os.path.join(directory, "lstm_model.keras"))
                or os.path.exists(os.path.join(directory, "lstm_fallback.joblib")))
