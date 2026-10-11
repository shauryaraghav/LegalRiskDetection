"""Agreement-level analysis (shared by the CLI and the Streamlit app)."""
import sys
from collections import Counter
from typing import Dict

from src.clause_splitter import extract_clauses
from src.config import DISCLAIMER, RISK_RANK
from src.predict import predict_clause
from src.risk_analysis import analyze_risk
from src.suggestions import missing_protections


def overall_label(score: float) -> str:
    if score >= 60:
        return "HIGH"
    if score >= 35:
        return "MEDIUM"
    return "LOW"


def analyze_agreement(text: str) -> Dict:
    """Analyse an agreement and return a result dictionary (no printing)."""
    clauses = extract_clauses(text)
    preds = [predict_clause(c) for c in clauses]

    risk_counts = Counter(p.risk for p in preds)
    high, medium, low = (risk_counts.get(k, 0) for k in ("High", "Medium", "Low"))
    total = len(preds)
    score = round(((2 * high) + medium) / (2 * total) * 100, 2) if total else 0.0

    # most critical first: risk level, then rule evidence, then model confidence
    ranked = sorted(preds, key=lambda p: (RISK_RANK[p.risk], p.rule_score, p.risk_confidence), reverse=True)

    # prioritised action plan: High first, then Medium; most critical first inside each level
    action_plan = [
        {"clause": p.clause, "category": p.category, "risk": p.risk, "suggestions": p.suggestions}
        for p in ranked if p.suggestions
    ]

    return {
        "analysis": analyze_risk(preds),
        "action_plan": action_plan,
        "missing": missing_protections(text, total),
        "predictions": preds,
        "ranked": ranked,
        "total": total,
        "high": high, "medium": medium, "low": low,
        "score": score,
        "overall": overall_label(score) if total else "N/A",
        "categories": Counter(p.category for p in preds),
    }


def serialize_result(result: Dict) -> Dict:
    """JSON-friendly version of analyze_agreement() output (used by the API)."""
    return {
        "total": result["total"], "high": result["high"], "medium": result["medium"], "low": result["low"],
        "score": result["score"], "overall": result["overall"],
        "categories": dict(result["categories"]),
        "analysis": result["analysis"], "action_plan": result["action_plan"], "missing": result["missing"],
        "predictions": [p.as_dict() for p in result["predictions"]],
        "disclaimer": DISCLAIMER,
    }


def print_report(result: Dict) -> None:
    print("\n========== AGREEMENT ANALYSIS ==========\n")
    for i, p in enumerate(result["predictions"], 1):
        print(f"Clause {i}: {p.clause[:150]}")
        print(f"  Category : {p.category} ({p.category_confidence}%)")
        print(f"  Risk     : {p.risk} ({p.risk_confidence}%)  [{p.source}]")
        print(f"  Why      : {p.explanation}")
        print("-" * 60)
    if not result["total"]:
        print("No clauses detected")
        return
    print("========== SUMMARY ==========")
    print(f"Clauses: {result['total']}  High: {result['high']}  Medium: {result['medium']}  Low: {result['low']}")
    print(f"Risk score: {result['score']}%  ->  Overall: {result['overall']}")
    print(f"Categories: {dict(result['categories'])}")

    a = result["analysis"]
    print("\n========== RISK SCORE ANALYSIS ==========")
    for line in a["summary"]:
        print(" -", line)
    print("\nBy category (score = risk within that category, share = part of total risk):")
    for r in a["categories"]:
        print(f"  {r['category']:<17} clauses={r['clauses']:<2} H/M/L={r['high']}/{r['medium']}/{r['low']}  "
              f"score={r['category_score']:>5}%  share={r['share_of_total_risk']:>5}%")
    print("\nWhat-if scenarios:")
    for w in a["what_if"]:
        print(f"  {w['scenario']}: {w['score']}% ({w['band']})")

    print("\n========== SUGGESTIONS (priority order) ==========")
    for item in result["action_plan"]:
        print(f"[{item['risk']}] {item['clause'][:110]}")
        for s in item["suggestions"]:
            print(f"   - {s['action']}")
            print(f"     e.g. \"{s['safer_wording']}\"")
    if result["missing"]:
        print("\nPossibly missing protections:")
        for m in result["missing"]:
            print(f"  - {m['topic']}: {m['action']}")
    print("\n" + DISCLAIMER)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        with open(sys.argv[1], encoding="utf-8", errors="ignore") as fh:
            sample = fh.read()
    else:
        sample = """
        The tenant shall pay rent on time.
        Failure to pay will result in penalty.
        The landlord may terminate without notice.
        Security deposit will be refunded.
        """
    print_report(analyze_agreement(sample))
