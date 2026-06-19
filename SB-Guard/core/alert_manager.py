"""
alert_manager.py — Alert lifecycle + maintenance recommendations.

Owns the in-memory active-alert set, the historical alert log (persisted to
logs/alerts.jsonl) and the rule-based maintenance recommendation text shown on
the Well Inspector page. Thread-safe for the realtime simulation loop.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime

import config as cfg
from models import ensemble


_REC = {
    "CRITICAL": (
        "IMMEDIATE ACTION — Dispatch crew within 24h. Polished-rod misalignment "
        "confirmed. Inspect/align stuffing box, replace packing, verify carrier "
        "bar and bridle alignment. Consider shutdown to prevent environmental "
        "release. | إجراء فوري خلال ٢٤ ساعة"
    ),
    "WARNING": (
        "SCHEDULE INSPECTION — Plan intervention within 48–72h. Monitor lateral "
        "vibration and friction trend. Check polished-rod centering and stuffing "
        "box temperature on next route. | جدولة الفحص خلال ٧٢ ساعة"
    ),
    "NORMAL": (
        "NO ACTION — Well operating within normal envelope. Continue routine "
        "surveillance. | لا يلزم اتخاذ إجراء"
    ),
}


class AlertManager:
    def __init__(self, log_path: str | None = None):
        self.log_path = log_path or os.path.join(cfg.LOGS_DIR, "alerts.jsonl")
        self._lock = threading.Lock()
        self.active: dict[str, dict] = {}      # well_id -> alert
        self.history: list[dict] = []
        self._load_history()

    # ------------------------------------------------------------------ #
    def recommendation(self, band: str) -> str:
        return _REC.get(band, _REC["NORMAL"])

    def evaluate(self, well_id: str, score: float, horizons: dict,
                 lead_time_h: float | None = None,
                 msi: float | None = None) -> dict | None:
        """
        Update alert state for a well given its latest combined score.
        Returns the alert dict if the well is in an alert state, else None.
        """
        band = ensemble.severity_band(score)
        now = datetime.utcnow().isoformat(timespec="seconds")

        with self._lock:
            if band in ("CRITICAL", "WARNING"):
                existing = self.active.get(well_id)
                alert = {
                    "well_id": well_id,
                    "alert_type": "SB Leak Risk — Polished Rod Misalignment",
                    "severity": band,
                    "score": round(float(score), 4),
                    "horizons": {k: round(float(v), 4) for k, v in horizons.items()},
                    "msi": round(float(msi), 4) if msi is not None else None,
                    "lead_time_h": round(float(lead_time_h), 1)
                                   if lead_time_h is not None else None,
                    "detected_at": existing["detected_at"] if existing else now,
                    "updated_at": now,
                    "recommended_action": self.recommendation(band),
                    "status": "ACTIVE",
                }
                self.active[well_id] = alert
                return alert
            else:
                # Clearing an active alert -> archive as resolved.
                if well_id in self.active:
                    closed = self.active.pop(well_id)
                    closed["status"] = "CLEARED"
                    closed["cleared_at"] = now
                    self._append_history(closed)
                return None

    def confirm_outcome(self, well_id: str, outcome: str):
        """Mark a historical alert outcome: confirmed / false_alarm / averted."""
        with self._lock:
            for a in reversed(self.history):
                if a["well_id"] == well_id and a.get("outcome") is None:
                    a["outcome"] = outcome
                    self._rewrite_history()
                    return True
        return False

    def seed_history(self, records: list[dict]):
        """Bulk-load historical alerts (used after model evaluation)."""
        with self._lock:
            self.history.extend(records)
            self._rewrite_history()

    def active_list(self) -> list[dict]:
        with self._lock:
            return sorted(self.active.values(),
                          key=lambda a: a["score"], reverse=True)

    def history_list(self, limit: int = 500) -> list[dict]:
        with self._lock:
            return list(reversed(self.history[-limit:]))

    def counts(self) -> dict:
        with self._lock:
            crit = sum(1 for a in self.active.values() if a["severity"] == "CRITICAL")
            warn = sum(1 for a in self.active.values() if a["severity"] == "WARNING")
            return {"critical": crit, "warning": warn,
                    "active_total": len(self.active),
                    "history_total": len(self.history)}

    # ------------------------------------------------------------------ #
    def _append_history(self, alert: dict):
        alert.setdefault("outcome", None)
        self.history.append(alert)
        with open(self.log_path, "a") as fh:
            fh.write(json.dumps(alert) + "\n")

    def _rewrite_history(self):
        with open(self.log_path, "w") as fh:
            for a in self.history:
                fh.write(json.dumps(a) + "\n")

    def _load_history(self):
        if os.path.exists(self.log_path):
            with open(self.log_path) as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        try:
                            self.history.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
