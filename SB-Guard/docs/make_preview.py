"""
make_preview.py — Render a static preview of SB-Guard's degradation signal.

Produces docs/signal_preview.png: a multi-panel plot of a failing well's
key signals (Lateral Vibration, SB Friction, Combined Risk, Surface Card Area)
annotated with the four degradation stages. Useful as a quick visual reference
without launching the full UI.

    python docs/make_preview.py
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfg
from data import simulator, preprocessor

NAVY = cfg.THEME["bg_navy"]
ACCENT = cfg.THEME["accent"]


def main():
    df = simulator.simulate_well("PDO-PREVIEW-001", fails=True, seed=7)
    feat = preprocessor.engineer_features(df)
    # zoom into the final 5 days around the failure
    ev = np.where(feat["failure_event"].values == 1)[0]
    end = ev[0] + cfg.SAMPLES_PER_HOUR * 6 if len(ev) else len(feat)
    start = max(0, end - cfg.SAMPLES_PER_DAY * 5)
    w = feat.iloc[start:end]
    t = (w["timestamp"] - w["timestamp"].iloc[0]).dt.total_seconds() / 3600.0

    fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True,
                             facecolor=NAVY)
    panels = [
        ("LVI", "Lateral Vibration Index", cfg.THEME["yellow"]),
        ("sb_friction", "Stuffing Box Friction", "#b07bff"),
        ("MSI", "Misalignment Severity Index", ACCENT),
        ("SCA", "Surface Card Area", "#4fa3ff"),
    ]
    for ax, (col, label, c) in zip(axes, panels):
        ax.plot(t, w[col].values, color=c, lw=1.6)
        ax.set_facecolor("#10233d")
        ax.set_ylabel(label, color=cfg.THEME["text"], fontsize=9)
        ax.tick_params(colors="#8aa0bd", labelsize=8)
        for s in ax.spines.values():
            s.set_color("#1d3454")
        ax.grid(color="#1d3454", lw=0.5, alpha=0.6)

    axes[-1].set_xlabel("Hours", color=cfg.THEME["text"])
    fig.suptitle("SB-Guard — Synthetic Degradation Signal (failing well)",
                 color=ACCENT, fontsize=14, fontweight="bold")
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "signal_preview.png")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out, dpi=110, facecolor=NAVY)
    print("Saved", out)


if __name__ == "__main__":
    main()
