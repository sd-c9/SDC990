"""
run.py — Convenience launcher for SB-Guard.

Ensures models exist (training automatically on first run if needed), then
starts the Flask server and opens the operator console in the default browser.

    python run.py
"""

from __future__ import annotations

import os
import threading
import webbrowser

import config as cfg
from models.lstm_model import LSTMLeakClassifier
from models.isolation_forest import AnomalyDetector


def ensure_trained():
    scored = os.path.join(cfg.DATASET_DIR, "fleet_scored.parquet")
    if (os.path.exists(scored)
            and LSTMLeakClassifier.exists()
            and AnomalyDetector.exists()):
        return
    print("First run — generating data and training models (one-time)…")
    import train
    train.main()


def main():
    ensure_trained()
    # Import app after training so the predictor loads fresh artifacts.
    from app import app, start_realtime
    start_realtime()
    url = "http://127.0.0.1:5000"
    threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    print(f"\n  SB-Guard running at {url}  (Ctrl+C to stop)\n")
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True,
            use_reloader=False)


if __name__ == "__main__":
    main()
