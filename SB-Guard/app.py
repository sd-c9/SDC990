"""
app.py — SB-Guard Flask application.

Serves the five-page operator console (Dashboard, Well Inspector, Alerts &
History, Analytics, Model Management) plus a JSON API consumed by the Plotly
front-end. A background thread advances the realtime playhead every
REALTIME_REFRESH_SEC seconds so the fleet view updates live.

Run:  python app.py   (or python run.py to auto-open a browser)
"""

from __future__ import annotations

import os
import threading
import time
import io

from flask import (Flask, render_template, jsonify, request,
                   send_file, abort)

import config as cfg
from config import Config
from core.predictor import predictor
from core import cost_calculator
from reports import report_generator

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config.from_object(Config)


# --------------------------------------------------------------------------- #
# Realtime simulation loop
# --------------------------------------------------------------------------- #
_sim_thread = None
_sim_running = threading.Event()


def _realtime_loop():
    while _sim_running.is_set():
        time.sleep(cfg.REALTIME_REFRESH_SEC)
        try:
            predictor.advance(steps=1)        # reveal next hour of telemetry
        except Exception as exc:              # pragma: no cover
            app.logger.error("realtime advance failed: %s", exc)


def start_realtime():
    global _sim_thread
    if _sim_thread and _sim_thread.is_alive():
        return
    _sim_running.set()
    _sim_thread = threading.Thread(target=_realtime_loop, daemon=True)
    _sim_thread.start()


# --------------------------------------------------------------------------- #
# Template context
# --------------------------------------------------------------------------- #
@app.context_processor
def inject_globals():
    return {
        "APP_NAME": cfg.APP_NAME,
        "APP_TAGLINE": cfg.APP_TAGLINE,
        "OPERATOR": cfg.OPERATOR,
        "THEME": cfg.THEME,
        "model_ready": predictor.ready,
    }


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #
@app.route("/")
def dashboard():
    return render_template("dashboard.html", active="dashboard")


@app.route("/inspector")
def inspector():
    return render_template("well_inspector.html", active="inspector",
                           wells=predictor.wells)


@app.route("/alerts")
def alerts_page():
    return render_template("alerts.html", active="alerts")


@app.route("/analytics")
def analytics_page():
    return render_template("analytics.html", active="analytics")


@app.route("/model")
def model_page():
    return render_template("model_management.html", active="model")


# --------------------------------------------------------------------------- #
# API — dashboard
# --------------------------------------------------------------------------- #
@app.route("/api/kpis")
def api_kpis():
    return jsonify(predictor.kpis())


@app.route("/api/fleet")
def api_fleet():
    return jsonify({"wells": predictor.fleet_status(),
                    "current_time": predictor.current_time().isoformat()})


@app.route("/api/fleet_health")
def api_fleet_health():
    hours = int(request.args.get("hours", 24))
    return jsonify(predictor.fleet_health_history(hours))


# --------------------------------------------------------------------------- #
# API — well inspector
# --------------------------------------------------------------------------- #
@app.route("/api/well/<well_id>")
def api_well(well_id):
    detail = predictor.well_detail(well_id)
    if not detail:
        abort(404, description="Unknown well")
    return jsonify(detail)


# --------------------------------------------------------------------------- #
# API — alerts
# --------------------------------------------------------------------------- #
@app.route("/api/alerts")
def api_alerts():
    return jsonify({
        "active": predictor.alerts.active_list(),
        "history": predictor.alerts.history_list(),
        "counts": predictor.alerts.counts(),
    })


@app.route("/api/alerts/outcome", methods=["POST"])
def api_alert_outcome():
    body = request.get_json(force=True)
    ok = predictor.alerts.confirm_outcome(body.get("well_id"),
                                          body.get("outcome", "confirmed"))
    return jsonify({"ok": ok})


# --------------------------------------------------------------------------- #
# API — analytics & model
# --------------------------------------------------------------------------- #
@app.route("/api/metrics")
def api_metrics():
    return jsonify(predictor.model_info())


@app.route("/api/at_risk")
def api_at_risk():
    return jsonify({"ranking": predictor.at_risk_ranking(int(request.args.get("top", 10)))})


@app.route("/api/trends")
def api_trends():
    savings = cost_calculator.estimate_total_savings(
        predictor._failures_prevented)
    return jsonify({
        "monthly_failures": predictor.monthly_failure_trend(),
        "savings": savings,
        "timing": predictor.metrics.get("timing", {}),
    })


@app.route("/api/threshold", methods=["POST"])
def api_threshold():
    body = request.get_json(force=True)
    predictor.set_threshold(float(body.get("threshold", cfg.ALERT_THRESHOLD)))
    return jsonify({"threshold": predictor.threshold})


@app.route("/api/data_quality")
def api_data_quality():
    return jsonify(predictor.data_quality())


@app.route("/api/registry")
def api_registry():
    import json
    path = os.path.join(cfg.SAVED_MODELS_DIR, "registry.json")
    if not os.path.exists(path):
        return jsonify({"versions": []})
    with open(path) as fh:
        return jsonify({"versions": json.load(fh)})


@app.route("/api/retrain", methods=["POST"])
def api_retrain():
    """
    Kick off retraining in a background thread. The UI polls /api/retrain/status.
    Retraining reruns train.py then hot-reloads the predictor.
    """
    if _retrain_state["running"]:
        return jsonify({"status": "already_running"})

    def _job():
        _retrain_state.update(running=True, progress=5, message="Generating data")
        try:
            import train
            _retrain_state.update(progress=30, message="Training models")
            train.main()
            _retrain_state.update(progress=85, message="Reloading predictor")
            predictor.reload()
            _retrain_state.update(progress=100, message="Complete", running=False)
        except Exception as exc:               # pragma: no cover
            _retrain_state.update(running=False, progress=0,
                                  message=f"Failed: {exc}")

    threading.Thread(target=_job, daemon=True).start()
    return jsonify({"status": "started"})


_retrain_state = {"running": False, "progress": 0, "message": "idle"}


@app.route("/api/retrain/status")
def api_retrain_status():
    return jsonify(_retrain_state)


# --------------------------------------------------------------------------- #
# Reports — PDF / Excel export
# --------------------------------------------------------------------------- #
@app.route("/api/report/<well_id>")
def api_report(well_id):
    detail = predictor.well_detail(well_id)
    if not detail:
        abort(404)
    path = report_generator.build_well_pdf(well_id, detail, predictor.metrics)
    return send_file(path, as_attachment=True,
                     download_name=f"SB-Guard_{well_id}_report.pdf")


@app.route("/api/export/alerts.xlsx")
def api_export_alerts():
    buf = report_generator.alerts_to_excel(
        predictor.alerts.active_list(), predictor.alerts.history_list())
    return send_file(buf, as_attachment=True,
                     download_name="SB-Guard_alerts.xlsx",
                     mimetype="application/vnd.openxmlformats-officedocument."
                              "spreadsheetml.sheet")


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model_ready": predictor.ready,
                    "wells": len(predictor.wells)})


# --------------------------------------------------------------------------- #
if not predictor.ready:
    app.logger.warning("Models/scored data not found. Run `python train.py` first.")
start_realtime()


if __name__ == "__main__":
    print(f"\n  {cfg.APP_NAME} — {cfg.APP_TAGLINE}")
    print(f"  {cfg.OPERATOR}")
    print(f"  Model ready: {predictor.ready} | Wells: {len(predictor.wells)}")
    print("  Open http://127.0.0.1:5000\n")
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
