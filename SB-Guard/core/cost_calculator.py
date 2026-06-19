"""
cost_calculator.py — Cost-savings estimation for SB-Guard.

Translates averted Stuffing Box failures into a USD savings figure used across
the Dashboard and Analytics pages. A prevented unplanned failure avoids
workover + cleanup + deferred production + environmental remediation, minus the
cost of acting early on the prediction (planned intervention).
"""

from __future__ import annotations

import config as cfg


def savings_per_averted_failure() -> float:
    """Net USD saved each time a predicted failure is acted on in time."""
    gross = (cfg.COST_PER_UNPLANNED_FAILURE
             + cfg.COST_PER_ENV_INCIDENT
             + cfg.DEFERRED_OIL_BBL_PER_FAILURE * cfg.OIL_PRICE_USD_PER_BBL)
    return float(gross - cfg.PLANNED_INTERVENTION_COST)


def estimate_total_savings(failures_prevented: int) -> dict:
    per = savings_per_averted_failure()
    total = per * max(0, failures_prevented)
    deferred_bbl = cfg.DEFERRED_OIL_BBL_PER_FAILURE * max(0, failures_prevented)
    return {
        "failures_prevented": int(failures_prevented),
        "savings_per_event_usd": round(per, 2),
        "total_savings_usd": round(total, 2),
        "deferred_oil_avoided_bbl": int(deferred_bbl),
        "intervention_cost_usd": round(
            cfg.PLANNED_INTERVENTION_COST * max(0, failures_prevented), 2),
    }


def fmt_usd(amount: float) -> str:
    if abs(amount) >= 1_000_000:
        return f"${amount/1_000_000:.2f}M"
    if abs(amount) >= 1_000:
        return f"${amount/1_000:.1f}K"
    return f"${amount:,.0f}"
