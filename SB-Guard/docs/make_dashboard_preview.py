"""
make_dashboard_preview.py — Render real artifact-driven preview images.

Produces dark-themed PNGs from the *actual* trained model output (no browser
needed), mirroring what the live Plotly pages show:

    docs/preview_analytics.png   ROC curve, confusion matrix, feature importance
    docs/preview_inspector.png   dynacard overlay, horizon bars, risk timeline

    python docs/make_dashboard_preview.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg
from core.predictor import Predictor

HERE = os.path.dirname(os.path.abspath(__file__))
NAVY, PANEL, ACCENT = cfg.THEME["bg_navy"], cfg.THEME["panel"], cfg.THEME["accent"]
TEXT, MUTED = cfg.THEME["text"], "#8aa0bd"
GREEN, YELLOW, RED = cfg.THEME["green"], cfg.THEME["yellow"], cfg.THEME["red"]


def _style(ax, title):
    ax.set_facecolor(PANEL)
    ax.set_title(title, color=ACCENT, fontsize=11, fontweight="bold")
    ax.tick_params(colors=MUTED, labelsize=8)
    for s in ax.spines.values():
        s.set_color("#1d3454")
    ax.grid(color="#1d3454", lw=0.5, alpha=0.6)


def analytics_preview(metrics):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), facecolor=NAVY)
    cls = metrics["classification"]
    roc = metrics["roc"]

    ax = axes[0]; _style(ax, f"ROC Curve (AUC = {roc['auc']:.3f})")
    ax.plot(roc["fpr"], roc["tpr"], color=ACCENT, lw=2.5)
    ax.fill_between(roc["fpr"], roc["tpr"], color=ACCENT, alpha=0.12)
    ax.plot([0, 1], [0, 1], "--", color=MUTED, lw=1)
    ax.set_xlabel("False Positive Rate", color=TEXT, fontsize=9)
    ax.set_ylabel("True Positive Rate", color=TEXT, fontsize=9)

    ax = axes[1]; _style(ax, "Confusion Matrix (held-out wells)")
    cm = cls["confusion_matrix"]
    Z = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    ax.imshow(Z, cmap="Oranges", aspect="auto")
    for (i, j), v in np.ndenumerate(Z):
        ax.text(j, i, f"{['TN','FP','FN','TP'][i*2+j]}\n{v:,}",
                ha="center", va="center", color="#1a1205", fontweight="bold")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred Normal", "Pred Leak"], color=TEXT)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Act Normal", "Act Leak"], color=TEXT)
    ax.grid(False)

    ax = axes[2]; _style(ax, "Feature Importance (Isolation Forest)")
    fi = metrics["feature_importance"]
    f = fi["features"][:10][::-1]; imp = fi["importance"][:10][::-1]
    ax.barh(f, imp, color=ACCENT)
    ax.set_xlabel("Relative Importance", color=TEXT, fontsize=9)

    fig.suptitle("SB-Guard — Analytics (from trained model output)",
                 color=ACCENT, fontsize=14, fontweight="bold")
    fig.text(0.5, 0.005,
             f"Precision {cls['precision']:.3f}  ·  Recall {cls['recall']:.3f}  ·  "
             f"F1 {cls['f1']:.3f}  ·  FAR {cls['false_alarm_rate']:.4f}  ·  "
             f"Mean lead {metrics['timing']['mean_lead_time_h']:.1f} h",
             color=TEXT, ha="center", fontsize=10)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    out = os.path.join(HERE, "preview_analytics.png")
    fig.savefig(out, dpi=110, facecolor=NAVY); print("Saved", out)


def inspector_preview(pred):
    # pick the highest-risk well
    fs = pred.fleet_status()
    wid = fs[0]["well_id"]
    d = pred.well_detail(wid)
    dc = d["dynacard"]; ts = d["timeseries"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), facecolor=NAVY)

    ax = axes[0]; _style(ax, f"Surface Dynacard — {wid}")
    ax.plot(dc["position"], dc["baseline_load"], "--", color=MUTED, lw=2, label="Baseline")
    ax.plot(dc["position"], dc["current_load"], color=ACCENT, lw=2.5, label="Current")
    ax.set_xlabel("Polished Rod Position (%)", color=TEXT, fontsize=9)
    ax.set_ylabel("Load (lb)", color=TEXT, fontsize=9)
    ax.legend(facecolor=PANEL, edgecolor="#1d3454", labelcolor=TEXT, fontsize=8)

    ax = axes[1]; _style(ax, "Failure Probability by Horizon")
    hz = d["horizons"]; labels = ["24 h", "48 h", "72 h"]
    vals = [hz["24h"], hz["48h"], hz["72h"]]
    bars = ax.bar(labels, vals, color=[RED if v >= 70 else YELLOW if v >= 45 else GREEN for v in vals])
    ax.set_ylim(0, 100); ax.set_ylabel("P(failure) %", color=TEXT, fontsize=9)
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, v+2, f"{v:.0f}%", ha="center", color=TEXT, fontsize=9)

    ax = axes[2]; _style(ax, "Combined Risk Score (last 72 h)")
    x = np.arange(len(ts["risk"]))
    ax.plot(x, ts["risk"], color=ACCENT, lw=2)
    ax.fill_between(x, ts["risk"], color=ACCENT, alpha=0.12)
    ax.axhline(cfg.ALERT_THRESHOLD*100, color=RED, ls="--", lw=1, label="Alert threshold")
    ax.set_ylim(0, 100); ax.set_xlabel("Hours", color=TEXT, fontsize=9)
    ax.set_ylabel("Risk %", color=TEXT, fontsize=9)
    ax.legend(facecolor=PANEL, edgecolor="#1d3454", labelcolor=TEXT, fontsize=8)

    fig.suptitle(f"SB-Guard — Well Inspector  ·  {wid}  ·  {d['band']} "
                 f"({d['score']:.0f}% risk)", color=ACCENT, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(HERE, "preview_inspector.png")
    fig.savefig(out, dpi=110, facecolor=NAVY); print("Saved", out)


def main():
    metrics = json.load(open(os.path.join(cfg.SAVED_MODELS_DIR, "metrics.json")))
    analytics_preview(metrics)
    pred = Predictor()
    inspector_preview(pred)


if __name__ == "__main__":
    main()
