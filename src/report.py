"""PDF report generation (in memory, no shared temp file)."""
from io import BytesIO
from xml.sax.saxutils import escape

from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from reportlab.lib import colors

from src.config import DISCLAIMER


def generate_pdf(result: dict) -> bytes:
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, title="AI Legal Analysis Report")
    styles = getSampleStyleSheet()
    els = [Paragraph("AI LEGAL ANALYSIS REPORT", styles["Title"]), Spacer(1, 10)]
    els.append(Paragraph(
        f"Overall risk: <b>{escape(result['overall'])}</b> &nbsp;|&nbsp; Score: {result['score']:.1f}% "
        f"&nbsp;|&nbsp; High: {result['high']} &nbsp;|&nbsp; Medium: {result['medium']} "
        f"&nbsp;|&nbsp; Low: {result['low']}", styles["Normal"]))
    els.append(Spacer(1, 12))
    a = result.get("analysis") or {}
    if a:
        els.append(Paragraph("Risk Score Analysis", styles["Heading2"]))
        for line in a["summary"]:
            els.append(Paragraph("&bull; " + escape(line), styles["Normal"]))
        els.append(Spacer(1, 6))
        rows = [["Category", "Clauses", "H/M/L", "Category risk", "Share of risk"]]
        for r in a["categories"]:
            rows.append([r["category"].replace("_", " "), str(r["clauses"]),
                         f"{r['high']}/{r['medium']}/{r['low']}", f"{r['category_score']}%", f"{r['share_of_total_risk']}%"])
        t = Table(rows, hAlign="LEFT")
        t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                               ("GRID", (0, 0), (-1, -1), 0.4, colors.grey), ("FONTSIZE", (0, 0), (-1, -1), 8)]))
        els += [t, Spacer(1, 6)]
        for w in a["what_if"]:
            els.append(Paragraph(f"What if: {escape(w['scenario'])} &rarr; {w['score']}% ({escape(w['band'])})", styles["Normal"]))
        els.append(Spacer(1, 12))

    plan = result.get("action_plan") or []
    if plan:
        els.append(Paragraph("Suggestions (priority order)", styles["Heading2"]))
        for item in plan:
            els.append(Paragraph(f"<b>[{escape(item['risk'])}]</b> {escape(item['clause'])}", styles["Normal"]))
            for sg in item["suggestions"]:
                els.append(Paragraph("&bull; " + escape(sg["action"]), styles["Normal"]))
                els.append(Paragraph("<i>Example: " + escape(sg["safer_wording"]) + "</i>", styles["Normal"]))
            els.append(Spacer(1, 6))
    if result.get("missing"):
        els.append(Paragraph("Possibly missing protections", styles["Heading2"]))
        for m in result["missing"]:
            els.append(Paragraph(f"&bull; <b>{escape(m['topic'])}</b>: {escape(m['action'])}", styles["Normal"]))
        els.append(Spacer(1, 12))

    els.append(Paragraph("Clause-by-clause results", styles["Heading2"]))
    # clause text is user-supplied: it MUST be escaped or characters like & < > break reportlab
    for i, p in enumerate(result["predictions"], 1):
        els.append(Paragraph(f"<b>Clause {i}</b> ({escape(p.category)}): {escape(p.clause)}", styles["Normal"]))
        els.append(Paragraph(f"<b>Risk:</b> {escape(p.risk)} &nbsp; {escape(p.explanation)}", styles["Normal"]))
        els.append(Spacer(1, 8))
    els.append(Spacer(1, 12))
    els.append(Paragraph(f"<i>{escape(DISCLAIMER)}</i>", styles["Normal"]))
    doc.build(els)
    return buf.getvalue()
