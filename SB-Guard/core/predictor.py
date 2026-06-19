"""
predictor.py — Real-time prediction engine for SB-Guard.

Serves the Flask application from the pre-scored fleet timeline produced by
train.py. A monotonically advancing "playhead" simulates live operation: each
refresh exposes the next hour of model output for every well, driving the
dashboard, alerts and well-inspector views.

Heavy model inference happens once (in train.py). At serving time this class
performs only cheap lookups + light derived computations (dynacard synthesis,
fleet aggregation), so the UI stays responsive and the 30-second realtime loop
is essentially free.
"""

from __future__ import annotations

import os
import threading
import numpy as np
import pandas as pd

import config as cfg
from models import ensemble
from models.evaluator import load_metrics
from core.alert_manager import AlertManager
from core import cost_calculator


class Predictor:
    def __init__(self):
        self._lock = threading.Lock()
        self.alerts = AlertManager()
        self.metrics = load_metrics()
        self.threshold = cfg.ALERT_THRESHOLD
        self.scored = None
        self.wells: list[str] = []
        self._well_frames: dict[str, pd.DataFrame] = {}
        self._times: np.ndarray = np.array([])
        self.playhead = 0
        self._failures_prevented = 0
        self._counted_wells: set[str] = set()
        self.ready = False
        self.load()

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #
    def load(self) -> bool:
        path = os.path.join(cfg.DATASET_DIR, "fleet_scored.parquet")
        if not os.path.exists(path):
            self.ready = False
            return False
        scored = pd.read_parquet(path).sort_values(["well_id", "timestamp"])
        self.scored = scored
        self.wells = sorted(scored["well_id"].unique().tolist())
        self._well_frames = {w: g.reset_index(drop=True)
                             for w, g in scored.groupby("well_id")}
        # Shared hourly time axis (use the longest well's timestamps).
        self._times = np.sort(scored["timestamp"].unique())
        # Start the playhead near the end so alerts are visible immediately,
        # but leave a 48h window of "future" for the realtime loop to reveal.
        n = len(self._times)
        self.playhead = max(0, n - 1 - 6 * 48)
        self.metrics = load_metrics()
        self.threshold = self.metrics.get("alert_threshold", cfg.ALERT_THRESHOLD)
        self.ready = True
        self._recompute_alerts()
        return True

    def reload(self):
        with self._lock:
            self.load()

    # ------------------------------------------------------------------ #
    # Time / realtime
    # ------------------------------------------------------------------ #
    def current_time(self) -> pd.Timestamp:
        if len(self._times) == 0:
            return pd.Timestamp.utcnow()
        return pd.Timestamp(self._times[min(self.playhead, len(self._times) - 1)])

    def advance(self, steps: int = 1):
        """Advance the realtime playhead and re-evaluate alerts."""
        with self._lock:
            self.playhead = min(self.playhead + steps, len(self._times) - 1)
            self._recompute_alerts()

    def _row_at_playhead(self, g: pd.DataFrame):
        """Latest scored row for a well at/just before the playhead time."""
        t = self.current_time()
        sub = g[g["timestamp"] <= t]
        if len(sub) == 0:
            return g.iloc[0]
        return sub.iloc[-1]

    # ------------------------------------------------------------------ #
    # Alert recomputation
    # ------------------------------------------------------------------ #
    def _recompute_alerts(self):
        for w, g in self._well_frames.items():
            row = self._row_at_playhead(g)
            horizons = {"24h": float(row["lstm_24h"]),
                        "48h": float(row["lstm_48h"]),
                        "72h": float(row["lstm_72h"])}
            h2f = row["hours_to_failure"]
            lead = float(h2f) if np.isfinite(h2f) else None
            alert = self.alerts.evaluate(
                w, float(row["combined_score"]), horizons,
                lead_time_h=lead, msi=float(row["MSI"]))
            # Count a "failure prevented" when a genuinely failing well is in
            # a critical alert with useful lead time before its failure event.
            if (alert and alert["severity"] == "CRITICAL"
                    and row["well_fails"] == 1 and lead and lead >= 12
                    and w not in self._counted_wells):
                self._counted_wells.add(w)
                self._failures_prevented += 1

    # ------------------------------------------------------------------ #
    # Fleet / dashboard
    # ------------------------------------------------------------------ #
    def fleet_status(self) -> list[dict]:
        out = []
        with self._lock:
            for w in self.wells:
                row = self._row_at_playhead(self._well_frames[w])
                score = float(row["combined_score"])
                band = ensemble.severity_band(score, self.threshold)
                out.append({
                    "well_id": w,
                    "score": round(score * 100, 1),
                    "band": band,
                    "color": ensemble.status_color(band),
                    "msi": round(float(row["MSI"]) * 100, 1),
                    "spm": round(float(row["SPM"]), 2),
                    "lvi": round(float(row["LVI"]), 3),
                    "fillage": round(float(row["pump_fillage"]), 1),
                })
        return sorted(out, key=lambda d: d["score"], reverse=True)

    def kpis(self) -> dict:
        counts = self.alerts.counts()
        savings = cost_calculator.estimate_total_savings(self._failures_prevented)
        with self._lock:
            statuses = [self._row_at_playhead(self._well_frames[w])["combined_score"]
                        for w in self.wells]
        fleet_health = float(100.0 * (1.0 - np.mean(statuses))) if statuses else 100.0
        return {
            "active_alerts": counts["active_total"],
            "critical": counts["critical"],
            "warning": counts["warning"],
            "wells_monitored": len(self.wells),
            "failures_prevented": self._failures_prevented,
            "estimated_cost_saved_usd": savings["total_savings_usd"],
            "estimated_cost_saved_fmt": cost_calculator.fmt_usd(
                savings["total_savings_usd"]),
            "fleet_health": round(fleet_health, 1),
            "current_time": self.current_time().isoformat(),
        }

    def fleet_health_history(self, hours: int = 24) -> dict:
        """Mean fleet health score over the trailing `hours` window."""
        with self._lock:
            t_end = self.current_time()
            t_start = t_end - pd.Timedelta(hours=hours)
            sub = self.scored[(self.scored["timestamp"] > t_start)
                              & (self.scored["timestamp"] <= t_end)]
        if len(sub) == 0:
            return {"t": [], "health": []}
        grp = (sub.groupby("timestamp")["combined_score"].mean()
                  .sort_index())
        health = (100.0 * (1.0 - grp)).round(2)
        return {"t": [pd.Timestamp(x).isoformat() for x in grp.index],
                "health": health.tolist()}

    # ------------------------------------------------------------------ #
    # Well inspector
    # ------------------------------------------------------------------ #
    def well_detail(self, well_id: str, hours: int = 72) -> dict:
        if well_id not in self._well_frames:
            return {}
        with self._lock:
            g = self._well_frames[well_id]
            t_end = self.current_time()
            t_start = t_end - pd.Timedelta(hours=hours)
            window = g[(g["timestamp"] > t_start) & (g["timestamp"] <= t_end)]
            row = self._row_at_playhead(g)

        score = float(row["combined_score"])
        band = ensemble.severity_band(score, self.threshold)
        ts = [pd.Timestamp(x).isoformat() for x in window["timestamp"]]
        detail = {
            "well_id": well_id,
            "band": band,
            "color": ensemble.status_color(band),
            "score": round(score * 100, 1),
            "msi": round(float(row["MSI"]) * 100, 1),
            "horizons": {
                "24h": round(float(row["lstm_24h"]) * 100, 1),
                "48h": round(float(row["lstm_48h"]) * 100, 1),
                "72h": round(float(row["lstm_72h"]) * 100, 1),
            },
            "recommendation": self.alerts.recommendation(band),
            "timeseries": {
                "t": ts,
                "spm": window["SPM"].round(3).tolist(),
                "lvi": window["LVI"].round(4).tolist(),
                "friction": window["sb_friction"].round(4).tolist(),
                "risk": (window["combined_score"] * 100).round(2).tolist(),
            },
            "dynacard": self._dynacard(row),
            "stats": {
                "pprl": round(float(row["PPRL"]), 0),
                "mprl": round(float(row["MPRL"]), 0),
                "prlr": round(float(row["PRLR"]), 0),
                "sca": round(float(row["SCA"]), 2),
                "fillage": round(float(row["pump_fillage"]), 1),
                "gearbox_temp": round(float(row["gearbox_temp"]), 1),
                "if_anomaly": round(float(row["if_anomaly"]) * 100, 1),
            },
        }
        return detail

    def _dynacard(self, row) -> dict:
        """
        Synthesize a surface dynacard (load vs polished-rod position) for
        visualization: a healthy baseline ellipse-like loop plus the current
        card distorted by MSI / load range. Position normalized 0..100%.
        """
        n = 80
        theta = np.linspace(0, 2 * np.pi, n)
        pos = 50 * (1 - np.cos(theta))                       # 0..100 stroke
        pprl, mprl = float(row["PPRL"]), float(row["MPRL"])
        mid = (pprl + mprl) / 2.0
        amp = (pprl - mprl) / 2.0

        # Baseline: smooth, near-symmetric card.
        base = mid + amp * np.sin(theta) - 0.12 * amp * np.sin(2 * theta)

        # Current: misalignment adds asymmetry, friction adds an upstroke bump.
        msi = float(row["MSI"])
        fric = float(row["sb_friction"])
        current = (mid + amp * np.sin(theta)
                   - (0.12 + 0.5 * msi) * amp * np.sin(2 * theta)
                   + 0.35 * msi * amp * np.sin(3 * theta)
                   + fric * amp * 0.4 * np.maximum(0, np.sin(theta)))
        return {
            "position": pos.round(2).tolist(),
            "baseline_load": base.round(1).tolist(),
            "current_load": current.round(1).tolist(),
        }

    # ------------------------------------------------------------------ #
    # Analytics helpers
    # ------------------------------------------------------------------ #
    def at_risk_ranking(self, top: int = 10) -> list[dict]:
        return self.fleet_status()[:top]

    def monthly_failure_trend(self) -> dict:
        fail_events = self.scored[self.scored["stage"] == 4]
        if len(fail_events) == 0:
            # fall back to failing wells grouped by failure month
            df = self.scored[self.scored["well_fails"] == 1]
        else:
            df = fail_events
        by_month = (df.assign(month=df["timestamp"].dt.to_period("M").astype(str))
                      .groupby("month")["well_id"].nunique())
        return {"months": by_month.index.tolist(),
                "failures": by_month.values.tolist()}

    def model_info(self) -> dict:
        return {
            "metrics": self.metrics,
            "threshold": self.threshold,
            "weights": {"lstm": cfg.ENSEMBLE_W_LSTM, "if": cfg.ENSEMBLE_W_IF},
        }

    def set_threshold(self, value: float):
        with self._lock:
            self.threshold = float(np.clip(value, 0.05, 0.95))
            self._recompute_alerts()

    def data_quality(self) -> dict:
        """Quick data-quality report for the Model Management page."""
        raw_path = os.path.join(cfg.DATASET_DIR, "fleet_raw.parquet")
        report = {"scored_rows": int(len(self.scored)) if self.scored is not None else 0,
                  "wells": len(self.wells)}
        if os.path.exists(raw_path):
            raw = pd.read_parquet(raw_path,
                                  columns=["SPM", "LVI", "sb_friction", "PPRL"])
            total = raw.size
            missing = int(raw.isna().sum().sum())
            report.update({
                "raw_rows": int(len(raw)),
                "missing_cells": missing,
                "missing_pct": round(100.0 * missing / total, 3),
                "completeness_pct": round(100.0 * (1 - missing / total), 3),
            })
        return report


# Singleton used by the Flask app.
predictor = Predictor()
