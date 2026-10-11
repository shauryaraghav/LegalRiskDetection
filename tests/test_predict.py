import pytest

from src.predict import combine_risk, predict_clause
from src.preprocess import normalize_text
from src.rules import RuleResult


def test_normalize_is_stable():
    assert normalize_text("Pay within 30 days; non-refundable!") == "pay within num days nonrefundable"


def test_strong_rule_forces_high():
    r = RuleResult(score=3)
    assert combine_risk("Low", {"Low": 0.9}, r)[0] == "High"


def test_confident_low_not_raised_by_moderate_keyword():
    r = RuleResult(score=2)
    assert combine_risk("Low", {"Low": 0.85}, r)[0] == "Low"


def test_weak_high_softened_by_mitigating_language():
    r = RuleResult(score=0, mitigating=["mutual / balanced"])
    assert combine_risk("High", {"High": 0.45}, r)[0] == "Medium"


def test_confident_high_not_downgraded():
    r = RuleResult(score=0, mitigating=["mutual / balanced"])
    assert combine_risk("High", {"High": 0.9}, r)[0] == "High"


@pytest.fixture(scope="module")
def pred():
    return predict_clause("The company may terminate employment without prior notice.")


def test_prediction_fields(pred):
    assert pred.risk == "High"
    assert pred.category == "Termination"
    assert 0 <= pred.risk_confidence <= 100 and 0 <= pred.category_confidence <= 100
    assert pred.explanation


def test_standard_clause_is_low():
    assert predict_clause("Rent shall be paid on or before the 5th of every month.").risk == "Low"


def test_confidence_is_not_clamped():
    # the old code clamped every confidence into 65-95%; real probabilities can fall outside
    vals = [predict_clause(t).risk_confidence for t in [
        "The parties agree to the terms herein.",
        "Rent shall be paid on or before the 5th of every month.",
        "Delay in delivery by the vendor of goods.",
    ]]
    assert min(vals) < 65 or max(vals) > 95
