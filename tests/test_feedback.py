import pandas as pd
import pytest

from src import config, feedback, merge_dataset


@pytest.fixture
def tmp_feedback(tmp_path, monkeypatch):
    path = tmp_path / "user_feedback.csv"
    monkeypatch.setattr(config, "FEEDBACK_PATH", path)
    monkeypatch.setattr(merge_dataset, "FEEDBACK_PATH", path)
    monkeypatch.setattr(merge_dataset, "DATASET_PATH", tmp_path / "final.csv")
    return path


def test_save_and_load(tmp_feedback):
    feedback.save_feedback("The tenant shall pay the rent monthly in advance", "Payment", "Low", "Payment", "Medium")
    df = feedback.load_feedback()
    assert len(df) == 1 and df.loc[0, "Risk_Level"] == "Low" and df.loc[0, "Predicted_Risk"] == "Medium"
    assert feedback.feedback_count() == 1


@pytest.mark.parametrize("clause,cat,risk", [
    ("too short", "Payment", "Low"),
    ("The tenant shall pay the rent monthly in advance", "NotACategory", "Low"),
    ("The tenant shall pay the rent monthly in advance", "Payment", "Extreme"),
])
def test_invalid_feedback_rejected(tmp_feedback, clause, cat, risk):
    with pytest.raises(ValueError):
        feedback.save_feedback(clause, cat, risk)
    assert not tmp_feedback.exists()


def test_formula_injection_is_neutralised(tmp_feedback):
    feedback.save_feedback('=HYPERLINK("http://evil") the tenant shall pay rent', "Payment", "Low")
    assert not feedback.load_feedback().loc[0, "Clause"].startswith("=")


def test_feedback_overrides_existing_label(tmp_feedback):
    base = merge_dataset.build_dataset(write=False)
    clause = base.loc[base.Risk_Level == "High", "Clause"].iloc[0]
    feedback.save_feedback(clause, "Termination", "Low")
    merged = merge_dataset.build_dataset(write=False)
    assert len(merged) == len(base)                         # replaced, not duplicated
    assert merged.loc[merged.Clause == clause, "Risk_Level"].iloc[0] == "Low"


def test_new_feedback_clause_is_added(tmp_feedback):
    n = len(merge_dataset.build_dataset(write=False))
    feedback.save_feedback("The lessee shall keep the garden tidy at all times", "Restriction", "Low")
    assert len(merge_dataset.build_dataset(write=False)) == n + 1
