"""Merge the hand-labelled CSVs (+ user corrections) into data/final_legal_dataset.csv.

Run from the project root:   python -m src.merge_dataset

User corrections from data/user_feedback.csv are applied LAST, so a correction
overrides an older label for the same clause.

The old random template generator (generate_dataset.py) was removed: it assigned
risk labels with random.choice(), which made the risk model learn noise.
"""
import pandas as pd

from src.config import DATA_DIR, DATASET_PATH, FEEDBACK_PATH

SOURCES = ["hand_labeled_clauses.csv", "additional_clauses.csv"]
COLS = ["Clause", "Category", "Risk_Level"]


def build_dataset(write: bool = True) -> pd.DataFrame:
    frames = []
    for name in SOURCES:
        df = pd.read_csv(DATA_DIR / name)
        df.columns = df.columns.str.strip()
        frames.append(df[COLS])
    if FEEDBACK_PATH.exists():
        fb = pd.read_csv(FEEDBACK_PATH)
        fb.columns = fb.columns.str.strip()
        if set(COLS).issubset(fb.columns):
            frames.append(fb[COLS])          # last => wins
    df = pd.concat(frames, ignore_index=True).dropna()
    for col in COLS:
        df[col] = df[col].astype(str).str.strip()
    df["Category"] = df["Category"].str.replace(" ", "_")
    df["Risk_Level"] = df["Risk_Level"].str.capitalize()
    bad = set(df["Risk_Level"]) - {"Low", "Medium", "High"}
    if bad:
        raise ValueError(f"Unknown risk labels: {bad}")
    df = df.drop_duplicates(subset="Clause", keep="last").reset_index(drop=True)
    if write:
        df.to_csv(DATASET_PATH, index=False)
    return df


def main():
    df = build_dataset()
    print(f"Wrote {len(df)} rows to {DATASET_PATH}")
    print(pd.crosstab(df["Category"], df["Risk_Level"]))


if __name__ == "__main__":
    main()
