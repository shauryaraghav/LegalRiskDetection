from src.agreement_analyzer import analyze_agreement
from src.report import generate_pdf

SAMPLE = """
1. The tenant shall pay rent on or before the fifth of every month.
2. Failure to pay shall result in a penalty of five percent & interest.
3. The landlord may terminate this agreement without notice.
4. The security deposit shall be refunded after inspection of the premises.
"""


def test_analysis_counts_and_score():
    r = analyze_agreement(SAMPLE)
    assert r["total"] == 4
    assert r["high"] + r["medium"] + r["low"] == 4
    assert 0 <= r["score"] <= 100
    assert r["ranked"][0].risk == "High"


def test_empty_text():
    r = analyze_agreement("")
    assert r["total"] == 0 and r["overall"] == "N/A"


def test_pdf_handles_special_characters():
    # "&" and "<" in clause text used to crash reportlab in the old version
    r = analyze_agreement(SAMPLE + "\n5. Fees <net> & taxes are payable by the buyer within thirty days.")
    pdf = generate_pdf(r)
    assert pdf[:4] == b"%PDF"
