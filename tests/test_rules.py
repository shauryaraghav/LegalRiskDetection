from src.rules import evaluate_rules, get_rule_score


def test_whole_word_matching():
    # "fine" inside "refine"/"define" and "charge" inside "discharge" must not match
    assert get_rule_score("We will refine and define the discharge procedure") == 0


def test_without_notice_is_strong():
    assert evaluate_rules("The company may terminate this contract without prior notice").score >= 3


def test_negated_penalty_is_ignored():
    assert evaluate_rules("There shall be no penalty for early repayment").score == 0


def test_mitigating_language_reduces_score():
    plain = evaluate_rules("The tenant shall be liable for damage").score
    softened = evaluate_rules("The tenant shall be liable for damage subject to reasonable wear and mutual agreement").score
    assert softened < plain


def test_indemnity_is_flagged_but_not_definitive():
    s = evaluate_rules("The vendor shall indemnify the client").score
    assert 2 <= s < 3
