"""User feedback loop: store corrections and retrain with them."""
import csv
import re
from datetime import datetime

import pandas as pd

from src import config
from src.config import CATEGORIES, RISK_LEVELS

_FIELDS = ["Clause", "Category", "Risk_Level", "Timestamp", "Predicted_Category", "Predicted_Risk"]
MIN_WORDS, MAX_CHARS = 4, 1500


def _clean(text: str) -> str:
    text = re.sub(r"\s+", " ", str(text)).strip()
    # neutralise spreadsheet-formula injection if the CSV is opened in Excel
    return text.lstrip("=+-@\t\r ")


def save_feedback(clause: str, correct_category: str, correct_risk: str,
                  predicted_category: str = "", predicted_risk: str = "") -> None:
    """Append one correction to data/user_feedback.csv (raises ValueError if invalid)."""
    clause = _clean(clause)
    if len(clause.split()) < MIN_WORDS or len(clause) > MAX_CHARS:
        raise ValueError(f"Clause must have {MIN_WORDS}-{MAX_CHARS} characters/words.")
    if correct_category not in CATEGORIES:
        raise ValueError(f"Unknown category: {correct_category!r}")
    if correct_risk not in RISK_LEVELS:
        raise ValueError(f"Unknown risk level: {correct_risk!r}")

    path = config.FEEDBACK_PATH
    new_file = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=_FIELDS)
        if new_file:
            w.writeheader()
        w.writerow({"Clause": clause, "Category": correct_category, "Risk_Level": correct_risk,
                    "Timestamp": datetime.now().isoformat(timespec="seconds"),
                    "Predicted_Category": predicted_category, "Predicted_Risk": predicted_risk})


def load_feedback() -> pd.DataFrame:
    path = config.FEEDBACK_PATH
    if not path.exists():
        return pd.DataFrame(columns=_FIELDS)
    return pd.read_csv(path)


def feedback_count() -> int:
    return len(load_feedback())


def retrain_with_feedback() -> dict:
    """Rebuild the dataset (corrections applied), retrain, and refresh the cached models."""
    from src.merge_dataset import build_dataset
    from src.predict import get_models
    from src.train_model import train

    build_dataset(write=True)
    metrics = train(verbose=False)
    get_models.cache_clear()
    return metrics
