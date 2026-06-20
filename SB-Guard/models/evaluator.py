"""
evaluator.py — Model evaluation metrics for SB-Guard.

Computes the full evaluation suite required by the spec:
    Precision / Recall / F1, ROC-AUC, confusion matrix,
    Mean Time To Detection (MTTD), False Alarm Rate (FAR),
    and lead time (hours) before actual failure.

Results are JSON-serializable so the Analytics page can render them live.
"""

from __future__ import annotations

import json
import os
import numpy as np
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, confusion_matrix)

import config as cfg


def classification_metrics(y_true: np.ndarray,
                           y_score: np.ndarray,
                           threshold: float = cfg.ALERT_THRESHOLD) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = (y_score >= threshold).astype(int)

    out = {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "threshold": float(threshold),
        "n_samples": int(len(y_true)),
        "n_positive": int(y_true.sum()),
    }
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, y_score))
    except ValueError:
        out["roc_auc"] = float("nan")

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    out["confusion_matrix"] = {"tn": int(tn), "fp": int(fp),
                               "fn": int(fn), "tp": int(tp)}
    # False Alarm Rate = FP / (FP + TN)
    out["false_alarm_rate"] = float(fp / (fp + tn)) if (fp + tn) else 0.0
    return out


def roc_curve_points(y_true, y_score, max_points: int = 200) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    try:
        fpr, tpr, _ = roc_curve(y_true, y_score)
    except ValueError:
        return {"fpr": [0, 1], "tpr": [0, 1], "auc": float("nan")}
    if len(fpr) > max_points:
        idx = np.linspace(0, len(fpr) - 1, max_points).astype(int)
        fpr, tpr = fpr[idx], tpr[idx]
    try:
        auc = float(roc_auc_score(y_true, y_score))
    except ValueError:
        auc = float("nan")
    return {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": auc}


def detection_timing(df_scores) -> dict:
    """
    Compute MTTD and lead-time statistics from a per-well scored frame.

    Expects columns: well_id, hours_to_failure (>=0 pre-failure; inf otherwise),
    combined_score, well_fails. For each failing well the lead time is the gap
    (hours) between the first alert crossing and the failure event.
    """
    lead_times = []
    detections = 0
    failing_wells = 0
    for wid, g in df_scores.groupby("well_id"):
        if not g["well_fails"].iloc[0]:
            continue
        failing_wells += 1
        pre = g[(g["hours_to_failure"] >= 0) & np.isfinite(g["hours_to_failure"])]
        alerts = pre[pre["combined_score"] >= cfg.ALERT_THRESHOLD]
        if len(alerts):
            detections += 1
            lead_times.append(float(alerts["hours_to_failure"].max()))

    lead = np.asarray(lead_times) if lead_times else np.array([0.0])
    return {
        "failing_wells": failing_wells,
        "detected_wells": detections,
        "detection_rate": float(detections / failing_wells) if failing_wells else 0.0,
        "mean_lead_time_h": float(lead.mean()),
        "median_lead_time_h": float(np.median(lead)),
        "max_lead_time_h": float(lead.max()),
        # MTTD: average hours from the earliest detectable stage (72h) to alert.
        "mttd_h": float(max(0.0, 72.0 - lead.mean())),
    }


def feature_importance(detector_model, feature_names=None) -> dict:
    """
    Approximate feature importance from the Isolation Forest by averaging the
    normalized count of times each feature is used as a split across trees.
    """
    feature_names = feature_names or cfg.IF_FEATURES
    counts = np.zeros(len(feature_names))
    try:
        for est in detector_model.estimators_:
            feats = est.tree_.feature
            for f in feats:
                if 0 <= f < len(feature_names):
                    counts[f] += 1
    except Exception:
        counts = np.ones(len(feature_names))
    if counts.sum() == 0:
        counts = np.ones(len(feature_names))
    imp = counts / counts.sum()
    order = np.argsort(imp)[::-1]
    return {"features": [feature_names[i] for i in order],
            "importance": [float(imp[i]) for i in order]}


def sanitize(obj):
    """Recursively replace non-finite floats with None so output is valid JSON
    (json.dump otherwise emits the non-standard `NaN`/`Infinity` tokens that
    break browsers' JSON.parse)."""
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


def save_metrics(metrics: dict, path: str | None = None):
    path = path or os.path.join(cfg.SAVED_MODELS_DIR, "metrics.json")
    with open(path, "w") as fh:
        json.dump(sanitize(metrics), fh, indent=2, allow_nan=False,
                  default=_json_default)
    return path


def load_metrics(path: str | None = None) -> dict:
    path = path or os.path.join(cfg.SAVED_MODELS_DIR, "metrics.json")
    if not os.path.exists(path):
        return {}
    with open(path) as fh:
        return json.load(fh)


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
