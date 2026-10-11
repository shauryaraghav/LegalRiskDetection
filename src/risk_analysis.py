"""Detailed risk-score analysis.

Score formula (unchanged): (2 * High + Medium) / (2 * clauses) * 100
Each High clause is worth 2 points, each Medium 1 point, Low 0 points.
"""
from collections import defaultdict
from typing import Dict, List

BANDS = {"LOW": (0, 35), "MEDIUM": (35, 60), "HIGH": (60, 100.01)}
POINTS = {"High": 2, "Medium": 1, "Low": 0}
LOW_CONFIDENCE = 50.0
LONG_DOC_CLAUSES = 60


def band_for(score: float) -> str:
    if score >= 60:
        return "HIGH"
    if score >= 35:
        return "MEDIUM"
    return "LOW"


def _score(points: float, n: int) -> float:
    return round(points / (2 * n) * 100, 1) if n else 0.0


def _what_if(preds: list, total: int) -> List[dict]:
    pts = sum(POINTS[p.risk] for p in preds)
    high = [p for p in preds if p.risk == "High"]
    scenarios = []

    def add(label, new_pts):
        s = _score(new_pts, total)
        scenarios.append({"scenario": label, "score": s, "band": band_for(s)})

    # reduce each of the (up to) 3 most critical clauses by one level
    top3 = sorted(preds, key=lambda p: (POINTS[p.risk], p.rule_score, p.risk_confidence), reverse=True)[:3]
    add("Renegotiate the 3 most critical clauses (one level lower)", pts - sum(1 for p in top3 if p.risk != "Low"))
    if high:
        add("Soften every High clause to Medium", pts - len(high))
        add("Fix every High clause to Low", pts - 2 * len(high))
    return scenarios


def analyze_risk(preds: list) -> Dict:
    total = len(preds)
    if not total:
        return {}
    points = sum(POINTS[p.risk] for p in preds)
    score = _score(points, total)
    band = band_for(score)

    # ---- per-category breakdown -------------------------------------------------
    groups = defaultdict(list)
    for p in preds:
        groups[p.category].append(p)
    rows = []
    for cat, items in groups.items():
        pts = sum(POINTS[p.risk] for p in items)
        rows.append({
            "category": cat,
            "clauses": len(items),
            "high": sum(p.risk == "High" for p in items),
            "medium": sum(p.risk == "Medium" for p in items),
            "low": sum(p.risk == "Low" for p in items),
            "category_score": _score(pts, len(items)),            # how risky this group is
            "points": pts,
            "share_of_total_risk": round(pts / points * 100, 1) if points else 0.0,  # how much of the score it explains
        })
    rows.sort(key=lambda r: (r["points"], r["category_score"]), reverse=True)

    # ---- reliability ------------------------------------------------------------
    low_conf = [p for p in preds if p.risk_confidence < LOW_CONFIDENCE]
    rule_adjusted = [p for p in preds if p.source.startswith("Rule")]
    reliability = {
        "low_confidence_clauses": len(low_conf),
        "rule_adjusted_clauses": len(rule_adjusted),
        "avg_confidence": round(sum(p.risk_confidence for p in preds) / total, 1),
    }
    ratio = len(low_conf) / total
    reliability["level"] = "High" if ratio < 0.2 else "Moderate" if ratio < 0.4 else "Low"

    # ---- distance to bands --------------------------------------------------------
    if band == "HIGH":
        gap = f"{score - 60:.1f} points above the High threshold (60)."
    elif band == "MEDIUM":
        gap = f"{60 - score:.1f} points below High and {score - 35:.1f} above Low."
    else:
        gap = f"{35 - score:.1f} points below the Medium threshold (35)."

    # ---- plain-English summary ----------------------------------------------------
    drivers = [r for r in rows if r["points"] > 0][:3]
    lines = [f"Overall score {score}% ({band}); {gap}"]
    if drivers:
        names = ", ".join(f"{r['category'].replace('_', ' ')} ({r['share_of_total_risk']:.0f}%)" for r in drivers)
        lines.append(f"Biggest drivers of risk: {names}.")
    high_share = round(sum(p.risk == "High" for p in preds) / total * 100)
    lines.append(f"{high_share}% of clauses are High risk.")
    if reliability["level"] != "High":
        lines.append(f"Treat the score as indicative: {len(low_conf)} of {total} clauses have model confidence below {LOW_CONFIDENCE:.0f}%.")

    if total >= LONG_DOC_CLAUSES:
        lines.append(f"Long document ({total} clauses): the risk model was trained on short clauses, so risk levels on long real-world "
                     "contracts are less reliable and tend to over-flag, and boilerplate clauses still receive a category, which skews the "
                     "overall score. Treat this as a starting point and review the flagged clauses yourself.")

    return {
        "score": score, "band": band, "points": points, "max_points": 2 * total,
        "gap_note": gap, "categories": rows, "what_if": _what_if(preds, total),
        "reliability": reliability, "summary": lines,
    }
