from src.compare import compare_agreements

A = """1. The Client shall pay invoices within 30 days of receipt.
2. Either party may terminate this Agreement on 60 days written notice.
3. The Provider's total liability shall not exceed the fees paid in the last 12 months.
4. This Agreement is governed by the laws of India."""
B = """1. The Client shall pay invoices within 7 days of receipt.
2. The Company may terminate this Agreement at any time without notice.
3. The Provider's total liability shall not exceed the fees paid in the last 12 months.
4. The Client shall indemnify and hold harmless the Provider from all claims."""


def test_identical_documents():
    r = compare_agreements(A, A)
    assert r["delta"] == 0 and r["counts"]["unchanged"] == 4
    assert r["counts"]["added"] == r["counts"]["removed"] == r["counts"]["modified"] == 0


def test_riskier_revision_detected():
    r = compare_agreements(A, B)
    assert r["delta"] > 0 and "RISKIER" in r["summary"][0]
    assert r["counts"]["riskier"] >= 2
    # most important rows first, unchanged last
    assert r["rows"][0]["trend"] == "riskier" and r["rows"][-1]["status"] == "unchanged"


def test_rewritten_clause_is_modified_not_removed_plus_added():
    r = compare_agreements(A, B)
    term = [x for x in r["rows"] if "terminate" in (x["clause_b"] or "")]
    assert term and term[0]["status"] == "modified" and term[0]["risk_a"] == "Low" and term[0]["risk_b"] == "High"


def test_number_change_is_a_modification():
    r = compare_agreements(A, B)
    pay = [x for x in r["rows"] if "invoices" in (x["clause_b"] or "")][0]
    assert pay["status"] == "modified" and "30" in pay["removed_words"] and "7" in pay["added_words"]


def test_safer_revision():
    r = compare_agreements(B, A)
    assert r["delta"] < 0 and "SAFER" in r["summary"][0]


def test_empty_side_is_handled():
    r = compare_agreements(A, "")
    assert "no detectable clauses" in r["summary"][0] and r["band_b"] == "N/A"
