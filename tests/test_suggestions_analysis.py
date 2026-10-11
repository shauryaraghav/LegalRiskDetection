from src.agreement_analyzer import analyze_agreement
from src.predict import predict_clause
from src.report import generate_pdf
from src.risk_analysis import analyze_risk, band_for
from src.rules import evaluate_rules
from src.suggestions import missing_protections, suggest_for_clause

CONTRACT = """
1. The Client shall pay all invoices within 30 days of receipt.
2. Late payments shall incur a late fee of 2% per month.
3. The Company may terminate this Agreement at any time without notice.
4. The Client shall indemnify and hold harmless the Provider from all claims.
5. Each party shall keep the other's confidential information secret for three years.
6. The Provider's total liability shall not exceed the fees paid in the last 12 months.
"""


# ---------------------------------------------------------------- suggestions
def test_low_risk_gets_no_suggestions():
    assert suggest_for_clause("Payment", "Low", evaluate_rules("Rent is paid monthly")) == []


def test_without_notice_suggestion_mentions_notice():
    s = suggest_for_clause("Termination", "High", evaluate_rules("Company may terminate without notice"))
    assert s and "notice" in s[0].action.lower() and s[0].safer_wording


def test_category_fallback_when_no_rule_fired():
    s = suggest_for_clause("Restriction", "Medium", evaluate_rules("Tenant shall keep pets out of the garden area"))
    assert len(s) == 1


def test_cap_clause_not_told_to_add_cap_via_liability_rule():
    r = evaluate_rules("The Provider's total liability shall not exceed the fees paid")
    reasons = [x.issue for x in suggest_for_clause("Liability", "Medium", r)]
    assert "Liability is placed on one party." not in reasons


def test_at_most_three_suggestions():
    r = evaluate_rules("Company may, at its sole discretion and without notice, forfeit the deposit and indemnify itself, waive claims")
    assert len(suggest_for_clause("Termination", "High", r)) <= 3


def test_predictions_carry_suggestions():
    p = predict_clause("The Company may terminate this Agreement at any time without notice")
    assert p.risk == "High" and p.suggestions


def test_missing_protections_skipped_for_short_text_and_found_for_long():
    assert missing_protections("Rent is due monthly", 2) == []
    topics = {m["topic"] for m in missing_protections("Rent is due monthly. " * 6, 6)}
    assert {"Termination", "Liability cap", "Confidentiality"} <= topics


# -------------------------------------------------------------- risk analysis
def test_bands():
    assert (band_for(10), band_for(40), band_for(75)) == ("LOW", "MEDIUM", "HIGH")


def test_analysis_numbers_are_consistent():
    r = analyze_agreement(CONTRACT)
    a = r["analysis"]
    assert a["score"] == round(r["score"], 1)
    assert a["points"] == 2 * r["high"] + r["medium"]
    assert sum(c["clauses"] for c in a["categories"]) == r["total"]
    assert abs(sum(c["share_of_total_risk"] for c in a["categories"]) - 100) < 0.5
    assert a["categories"][0]["points"] == max(c["points"] for c in a["categories"])


def test_what_if_scores_decrease():
    a = analyze_agreement(CONTRACT)["analysis"]
    base = a["score"]
    assert all(w["score"] <= base for w in a["what_if"])
    fix_all = [w for w in a["what_if"] if "Low" in w["scenario"]]
    assert fix_all and fix_all[0]["score"] < base


def test_analysis_empty_input():
    assert analyze_risk([]) == {}


def test_action_plan_is_high_first():
    plan = analyze_agreement(CONTRACT)["action_plan"]
    ranks = [{"High": 2, "Medium": 1}[i["risk"]] for i in plan]
    assert ranks == sorted(ranks, reverse=True)


def test_pdf_contains_new_sections():
    pdf = generate_pdf(analyze_agreement(CONTRACT))
    assert pdf[:4] == b"%PDF" and len(pdf) > 3000
