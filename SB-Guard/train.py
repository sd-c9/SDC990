"""
train.py — End-to-end training & scoring pipeline for SB-Guard.

Pipeline
--------
1. Load (or generate) the engineered feature dataset.
2. Fit a StandardScaler on the 14-feature vector.
3. Train the Isolation Forest on NORMAL-operation rows only.
4. Train the multi-horizon LSTM (well-level train/test split, class-weighted).
5. Score the entire fleet timeline with the ensemble (hourly resolution).
6. Evaluate on held-out wells: P/R/F1, ROC-AUC, MTTD, FAR, lead time, CM.
7. Persist models, scaler, metrics, feature importance and the scored frame
   that the live application serves from.

Run:  python train.py
"""

from __future__ import annotations

import os
import json
import time
import numpy as np
import pandas as pd
import joblib

import config as cfg
from data import dataset_generator, preprocessor
from models.lstm_model import LSTMLeakClassifier
from models.isolation_forest import AnomalyDetector
from models import ensemble, evaluator


SERVE_STRIDE = cfg.SAMPLES_PER_HOUR          # 1 scored point / hour for serving
TRAIN_STRIDE = cfg.SAMPLES_PER_HOUR * 2      # 1 training window / 2h (speed)


def _hours_to_failure(g: pd.DataFrame) -> np.ndarray:
    """Reconstruct hours-to-failure from the failure_event flag per well."""
    n = len(g)
    h2f = np.full(n, np.inf)
    ev = np.where(g["failure_event"].values == 1)[0]
    if len(ev):
        fail_idx = ev[0]
        idx = np.arange(n)
        h = (fail_idx - idx) * cfg.SAMPLE_INTERVAL_MIN / 60.0
        h[h < 0] = np.inf
        h2f = h
    return h2f


def main():
    t0 = time.time()
    print("=" * 64)
    print("SB-Guard :: Training & scoring pipeline")
    print("=" * 64)

    # --- 1. data --------------------------------------------------------- #
    feat_path = os.path.join(cfg.DATASET_DIR, "fleet_features.parquet")
    if not os.path.exists(feat_path):
        print("Feature dataset missing — generating synthetic fleet ...")
        dataset_generator.generate()
    df = dataset_generator.load_features()
    print(f"Loaded features: {df.shape[0]:,} rows, {df['well_id'].nunique()} wells")

    # --- 2. scaler ------------------------------------------------------- #
    scaler = preprocessor.fit_scaler(df)
    joblib.dump(scaler, os.path.join(cfg.SAVED_MODELS_DIR, "scaler.joblib"))
    print("Fitted + saved StandardScaler")

    # --- 3. Isolation Forest on normal rows ------------------------------ #
    normal_mask = (df["stage"].values == 0) & (df["well_fails"].values == 0)
    Xn = scaler.transform(df.loc[normal_mask, cfg.FEATURE_COLUMNS].values)
    # subsample for speed
    if len(Xn) > 120_000:
        rng = np.random.default_rng(cfg.RANDOM_SEED)
        Xn = Xn[rng.choice(len(Xn), 120_000, replace=False)]
    detector = AnomalyDetector().fit(Xn)
    detector.save()
    print(f"Trained Isolation Forest on {len(Xn):,} normal rows")

    # --- 4. LSTM --------------------------------------------------------- #
    wells = df["well_id"].unique().tolist()
    rng = np.random.default_rng(cfg.RANDOM_SEED)
    rng.shuffle(wells)
    n_test = max(1, int(0.30 * len(wells)))
    test_wells = set(wells[:n_test])
    train_wells = set(wells[n_test:])
    print(f"Well split: {len(train_wells)} train / {len(test_wells)} test")

    label_cols = ["label_24h", "label_48h", "label_72h"]
    Xtr, ytr, Xte, yte = [], [], [], []
    for wid, g in df.groupby("well_id", sort=False):
        g = g.sort_values("timestamp")
        vals = scaler.transform(g[cfg.FEATURE_COLUMNS].values)
        labs = g[label_cols].values.astype(np.float32)
        stride = TRAIN_STRIDE
        Xs, ys = preprocessor.build_sequences(vals, labs, cfg.LSTM_WINDOW, stride)
        if wid in test_wells:
            Xte.append(Xs); yte.append(ys)
        else:
            Xtr.append(Xs); ytr.append(ys)

    Xtr = np.concatenate(Xtr); ytr = np.concatenate(ytr)
    Xte = np.concatenate(Xte); yte = np.concatenate(yte)
    print(f"LSTM sequences: train {Xtr.shape}  test {Xte.shape}  "
          f"(pos24h train={int(ytr[:,0].sum())})")

    lstm = LSTMLeakClassifier().build()
    print(f"LSTM backend: {lstm.backend}")
    lstm.fit(Xtr, ytr, Xte, yte)
    lstm.save()
    print("Trained + saved LSTM")

    # --- 5. score full fleet (hourly) ----------------------------------- #
    print("Scoring full fleet timeline (hourly resolution) ...")
    scored_parts = []
    serve_cols = ["SPM", "LVI", "sb_friction", "PPRL", "MPRL", "PRLR",
                  "SCA", "pump_fillage", "gearbox_temp", "TDS", "MSI", "FLP"]
    for wid, g in df.groupby("well_id", sort=False):
        g = g.sort_values("timestamp").reset_index(drop=True)
        vals = scaler.transform(g[cfg.FEATURE_COLUMNS].values)
        if_anom = detector.anomaly_score(vals)
        h2f = _hours_to_failure(g)

        # window-end indices at serving stride
        ends = np.arange(cfg.LSTM_WINDOW, len(g) + 1, SERVE_STRIDE)
        seqs = np.stack([vals[e - cfg.LSTM_WINDOW:e] for e in ends]).astype(np.float32)
        probs = lstm.predict_proba(seqs)              # (m, 3)
        end_rows = ends - 1
        combined = ensemble.combine(probs[:, 0], if_anom[end_rows])

        part = pd.DataFrame({
            "well_id": wid,
            "timestamp": g["timestamp"].values[end_rows],
            "lstm_24h": probs[:, 0],
            "lstm_48h": probs[:, 1],
            "lstm_72h": probs[:, 2],
            "if_anomaly": if_anom[end_rows],
            "combined_score": combined,
            "hours_to_failure": h2f[end_rows],
            "well_fails": int(g["well_fails"].iloc[0]),
            "stage": g["stage"].values[end_rows],
            "label_24h": g["label_24h"].values[end_rows],
            "in_test": wid in test_wells,
        })
        for c in serve_cols:
            part[c] = g[c].values[end_rows]
        scored_parts.append(part)

    scored = pd.concat(scored_parts, ignore_index=True)
    scored_path = os.path.join(cfg.DATASET_DIR, "fleet_scored.parquet")
    scored.to_parquet(scored_path, index=False)
    print(f"Saved scored fleet -> {scored_path}  ({len(scored):,} rows)")

    # --- 6. evaluation (held-out wells) --------------------------------- #
    test_df = scored[scored["in_test"]]
    metrics = evaluator.classification_metrics(
        test_df["label_24h"].values, test_df["combined_score"].values)
    roc = evaluator.roc_curve_points(
        test_df["label_24h"].values, test_df["combined_score"].values)
    timing = evaluator.detection_timing(test_df)
    fi = evaluator.feature_importance(detector.model, cfg.IF_FEATURES)

    metrics_blob = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "backend": lstm.backend,
        "n_wells": len(wells),
        "n_train_wells": len(train_wells),
        "n_test_wells": len(test_wells),
        "classification": metrics,
        "roc": roc,
        "timing": timing,
        "feature_importance": fi,
        "ensemble_weights": {"lstm": cfg.ENSEMBLE_W_LSTM, "if": cfg.ENSEMBLE_W_IF},
        "alert_threshold": cfg.ALERT_THRESHOLD,
        "version": "1.0.0",
    }
    evaluator.save_metrics(metrics_blob)
    print(json.dumps({k: metrics[k] for k in
                      ("precision", "recall", "f1", "roc_auc", "false_alarm_rate")},
                     indent=2))
    print(json.dumps(timing, indent=2))

    # --- 7. model version registry -------------------------------------- #
    registry_path = os.path.join(cfg.SAVED_MODELS_DIR, "registry.json")
    registry = []
    if os.path.exists(registry_path):
        with open(registry_path) as fh:
            registry = json.load(fh)
    registry.append({
        "version": f"1.0.{len(registry)}",
        "trained_at": pd.Timestamp.utcnow().isoformat(),
        "backend": lstm.backend,
        "roc_auc": metrics.get("roc_auc"),
        "f1": metrics.get("f1"),
        "precision": metrics.get("precision"),
        "recall": metrics.get("recall"),
        "n_train_wells": len(train_wells),
    })
    with open(registry_path, "w") as fh:
        json.dump(registry, fh, indent=2, default=evaluator._json_default)

    print("-" * 64)
    print(f"Pipeline complete in {time.time()-t0:.1f}s")
    print(f"ROC-AUC={metrics['roc_auc']:.3f}  F1={metrics['f1']:.3f}  "
          f"Detection rate={timing['detection_rate']:.0%}  "
          f"Mean lead={timing['mean_lead_time_h']:.1f}h")


if __name__ == "__main__":
    main()
