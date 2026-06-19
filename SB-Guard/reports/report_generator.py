"""
report_generator.py — PDF (per-well) and Excel (alerts) export for SB-Guard.

Uses reportlab for a branded one-page well report and openpyxl (via pandas)
for the alert workbook export.
"""

from __future__ import annotations

import io
import os
from datetime import datetime

import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle)

import config as cfg

NAVY = colors.HexColor(cfg.THEME["bg_navy"])
ACCENT = colors.HexColor(cfg.THEME["accent"])
PANEL = colors.HexColor(cfg.THEME["panel"])


def _band_color(band: str):
    return {"CRITICAL": colors.HexColor(cfg.THEME["red"]),
            "WARNING": colors.HexColor(cfg.THEME["yellow"]),
            "NORMAL": colors.HexColor(cfg.THEME["green"])}.get(band, colors.grey)


def build_well_pdf(well_id: str, detail: dict, metrics: dict) -> str:
    """Render a one-page PDF report for a well and return its path."""
    os.makedirs(cfg.REPORTS_DIR, exist_ok=True)
    path = os.path.join(cfg.REPORTS_DIR, f"{well_id}_report.pdf")
    doc = SimpleDocTemplate(path, pagesize=A4,
                            topMargin=18 * mm, bottomMargin=18 * mm,
                            leftMargin=16 * mm, rightMargin=16 * mm)
    styles = getSampleStyleSheet()
    title = ParagraphStyle("title", parent=styles["Title"], textColor=NAVY,
                           fontSize=20, spaceAfter=2)
    sub = ParagraphStyle("sub", parent=styles["Normal"], textColor=ACCENT,
                         fontSize=11, spaceAfter=10)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], textColor=NAVY,
                        fontSize=13, spaceBefore=10, spaceAfter=4)
    body = ParagraphStyle("body", parent=styles["Normal"], fontSize=9.5,
                          leading=13)

    story = []
    story.append(Paragraph(f"{cfg.APP_NAME} — Stuffing Box Leak Risk Report", title))
    story.append(Paragraph(f"{cfg.OPERATOR} &nbsp;|&nbsp; "
                           f"Generated {datetime.utcnow():%Y-%m-%d %H:%M UTC}", sub))

    band = detail.get("band", "NORMAL")
    summary = [
        ["Well ID", well_id, "Risk Band", band],
        ["Combined Risk Score", f"{detail.get('score', 0)} %",
         "Misalignment Severity (MSI)", f"{detail.get('msi', 0)} %"],
    ]
    t = Table(summary, colWidths=[40 * mm, 45 * mm, 45 * mm, 38 * mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.whitesmoke),
        ("TEXTCOLOR", (0, 0), (-1, -1), NAVY),
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ("BACKGROUND", (3, 0), (3, 0), _band_color(band)),
        ("TEXTCOLOR", (3, 0), (3, 0), colors.white),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(t)

    story.append(Paragraph("Multi-Horizon Failure Probability", h2))
    hz = detail.get("horizons", {})
    hz_tbl = Table([["Horizon", "24 hours", "48 hours", "72 hours"],
                    ["P(failure)", f"{hz.get('24h',0)} %",
                     f"{hz.get('48h',0)} %", f"{hz.get('72h',0)} %"]],
                   colWidths=[42 * mm] + [42 * mm] * 3)
    hz_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(hz_tbl)

    story.append(Paragraph("Key Operating Parameters", h2))
    s = detail.get("stats", {})
    stat_rows = [
        ["Peak Polished Rod Load (PPRL)", f"{s.get('pprl',0):,.0f} lb",
         "Min Polished Rod Load (MPRL)", f"{s.get('mprl',0):,.0f} lb"],
        ["Polished Rod Load Range (PRLR)", f"{s.get('prlr',0):,.0f} lb",
         "Surface Card Area (SCA)", f"{s.get('sca',0)}"],
        ["Pump Fillage", f"{s.get('fillage',0)} %",
         "Gearbox Temperature", f"{s.get('gearbox_temp',0)} °C"],
        ["Isolation-Forest Anomaly", f"{s.get('if_anomaly',0)} %",
         "Ensemble Threshold", f"{int(cfg.ALERT_THRESHOLD*100)} %"],
    ]
    st = Table(stat_rows, colWidths=[52 * mm, 32 * mm, 52 * mm, 32 * mm])
    st.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
        ("BACKGROUND", (0, 0), (0, -1), colors.whitesmoke),
        ("BACKGROUND", (2, 0), (2, -1), colors.whitesmoke),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(st)

    story.append(Paragraph("Maintenance Recommendation", h2))
    story.append(Paragraph(detail.get("recommendation", ""), body))

    story.append(Spacer(1, 8))
    story.append(Paragraph("Model Performance (held-out wells)", h2))
    cl = (metrics or {}).get("classification", {})
    tm = (metrics or {}).get("timing", {})
    perf = Table([
        ["Precision", f"{cl.get('precision',0):.3f}",
         "Recall", f"{cl.get('recall',0):.3f}"],
        ["F1-Score", f"{cl.get('f1',0):.3f}",
         "ROC-AUC", f"{cl.get('roc_auc',0):.3f}"],
        ["False Alarm Rate", f"{cl.get('false_alarm_rate',0):.3f}",
         "Mean Lead Time", f"{tm.get('mean_lead_time_h',0):.1f} h"],
    ], colWidths=[42 * mm, 42 * mm, 42 * mm, 42 * mm])
    perf.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(perf)

    story.append(Spacer(1, 14))
    story.append(Paragraph(
        "<i>SB-Guard is a decision-support tool. Predictions are probabilistic; "
        "confirm with field inspection before intervention.</i>", body))

    doc.build(story)
    return path


def alerts_to_excel(active: list[dict], history: list[dict]) -> io.BytesIO:
    """Export active + historical alerts to an Excel workbook in memory."""
    buf = io.BytesIO()

    def _flat(rows):
        out = []
        for a in rows:
            out.append({
                "Well ID": a.get("well_id"),
                "Alert Type": a.get("alert_type"),
                "Severity": a.get("severity"),
                "Score": a.get("score"),
                "MSI": a.get("msi"),
                "Lead Time (h)": a.get("lead_time_h"),
                "Detected At": a.get("detected_at"),
                "Status": a.get("status"),
                "Outcome": a.get("outcome"),
            })
        return pd.DataFrame(out)

    with pd.ExcelWriter(buf, engine="openpyxl") as xl:
        _flat(active).to_excel(xl, sheet_name="Active Alerts", index=False)
        _flat(history).to_excel(xl, sheet_name="Alert History", index=False)
    buf.seek(0)
    return buf
