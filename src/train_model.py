"""Train the category and risk models.

Run from the project root:   python -m src.train_model

* Both models are single sklearn Pipelines (TF-IDF word+char n-grams -> LogisticRegression), so
  training and inference use identical preprocessing and the probabilities are real.
* CATEGORY model: trained on the project's hand-labelled clauses PLUS a down-weighted sample of
  real, lawyer-annotated clauses from CUAD (data/cuad_clauses.csv, see docs/CUAD.md).
* RISK model: trained ONLY on the project's own risk labels - CUAD has no risk levels.
* Two honest evaluations are stored in models/metrics.json:
    1. in-domain: stratified CV on the hand-labelled data (CUAD is only ever added to TRAINING folds)
    2. real contracts: CUAD clauses from contracts the model never saw (grouped by contract),
       compared with a model trained without CUAD.
"""
import json
import sys
from collections import Counter
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import FeatureUnion, Pipeline

from src.config import (CATEGORY_MODEL_PATH, CUAD_CAP_PER_CATEGORY, CUAD_PATH, CUAD_SAMPLE_WEIGHT,
                        DATASET_PATH, METRICS_PATH, MODEL_DIR, RISK_MODEL_PATH)
from src.preprocess import normalize_text


def build_pipeline() -> Pipeline:
    """Word 1-2grams + character 3-5grams -> logistic regression (cross-validated choice)."""
    features = FeatureUnion([
        ("w", TfidfVectorizer(preprocessor=normalize_text, ngram_range=(1, 2), sublinear_tf=True)),
        ("c", TfidfVectorizer(preprocessor=normalize_text, analyzer="char_wb",
                              ngram_range=(3, 5), sublinear_tf=True)),
    ])
    return Pipeline([
        ("features", features),
        ("clf", LogisticRegression(C=10, max_iter=3000, class_weight="balanced")),
    ])


def _fit(X, y, weights=None) -> Pipeline:
    pipe = build_pipeline()
    if weights is None:
        return pipe.fit(X, y)
    return pipe.fit(X, y, clf__sample_weight=np.asarray(weights, dtype=float))


def load_dataset(path=DATASET_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    needed = {"Clause", "Category", "Risk_Level"}
    if not needed.issubset(df.columns):
        raise ValueError(f"Dataset must contain columns {sorted(needed)}; found {list(df.columns)}")
    df = df.dropna(subset=list(needed)).drop_duplicates(subset="Clause")
    return df.reset_index(drop=True)


def load_cuad(path=CUAD_PATH) -> pd.DataFrame:
    """Real-contract clauses (category labels only). Empty frame if the file is absent."""
    if not path.exists():
        return pd.DataFrame(columns=["Clause", "Category", "Contract_Id"])
    return pd.read_csv(path).dropna(subset=["Clause", "Category"])


def cuad_sample(cuad: pd.DataFrame, cap: int = CUAD_CAP_PER_CATEGORY, seed: int = 0) -> pd.DataFrame:
    """At most `cap` clauses per category (fixed seed -> reproducible)."""
    if cuad.empty:
        return cuad
    return pd.concat([g.sample(min(cap, len(g)), random_state=seed) for _, g in cuad.groupby("Category")])


def _n_splits(y) -> int:
    return max(2, min(5, int(Counter(y).most_common()[-1][1])))


def _evaluate(name, X, y, extra=None, verbose=True):
    """Stratified CV. `extra` = (X_extra, y_extra, weight): added to TRAINING folds only."""
    n_splits = _n_splits(y)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    X, y = X.reset_index(drop=True), y.reset_index(drop=True)
    pred = np.empty(len(y), dtype=object)
    for tr, te in cv.split(X, y):
        if extra is not None and len(extra[0]):
            Xtr = pd.concat([X.iloc[tr], extra[0]])
            ytr = pd.concat([y.iloc[tr], extra[1]])
            w = np.r_[np.ones(len(tr)), np.full(len(extra[0]), extra[2])]
            model = _fit(Xtr, ytr, w)
        else:
            model = _fit(X.iloc[tr], y.iloc[tr])
        pred[te] = model.predict(X.iloc[te])

    labels = sorted(set(y))
    report = classification_report(y, pred, labels=labels, output_dict=True, zero_division=0)
    metrics = {
        "cv_folds": n_splits,
        "accuracy": round(float(accuracy_score(y, pred)), 3),
        "macro_f1": round(float(f1_score(y, pred, labels=labels, average="macro")), 3),
        "majority_baseline": round(Counter(y).most_common(1)[0][1] / len(y), 3),
        "per_class": {l: {k: round(float(report[l][k]), 3) for k in ("precision", "recall", "f1-score")}
                      | {"support": int(report[l]["support"])} for l in labels},
        "confusion": {"labels": labels, "matrix": confusion_matrix(y, pred, labels=labels).tolist()},
    }
    if verbose:
        print(f"\n===== {name.upper()} MODEL ({n_splits}-fold stratified CV) =====")
        print(classification_report(y, pred, labels=labels, zero_division=0))
    return metrics


def _evaluate_hybrid(X, y, verbose=True):
    """Cross-validated accuracy of ML + rules (the system actually shipped)."""
    from src.predict import combine_risk
    from src.rules import evaluate_rules

    cv = StratifiedKFold(n_splits=_n_splits(y), shuffle=True, random_state=42)
    proba = cross_val_predict(build_pipeline(), X, y, cv=cv, method="predict_proba")
    classes = sorted(set(y))
    ml_pred, final = [], []
    for text, p in zip(X, proba):
        probs = dict(zip(classes, p))
        ml = max(probs, key=probs.get)
        ml_pred.append(ml)
        final.append(combine_risk(ml, probs, evaluate_rules(text))[0])
    out = {
        "ml_only_accuracy": round(float(accuracy_score(y, ml_pred)), 3),
        "hybrid_accuracy": round(float(accuracy_score(y, final)), 3),
        "hybrid_macro_f1": round(float(f1_score(y, final, average="macro")), 3),
    }
    if verbose:
        print("\n===== RISK: ML ONLY vs ML + RULES (cross-validated) =====")
        print(out)
    return out


def _evaluate_real_contracts(hand: pd.DataFrame, cuad: pd.DataFrame, verbose=True) -> dict:
    """Category accuracy on REAL contracts the model has never seen (5-fold, grouped by contract).

    Compares the shipped recipe (hand data + CUAD sample) with a model trained on the hand data only.
    The test sample is capped at 120 clauses per category so big categories do not dominate.
    """
    base = _fit(hand["Clause"], hand["Category"])
    truth, with_cuad, without = [], [], []
    for tr, te in GroupKFold(n_splits=5).split(cuad, groups=cuad["Contract_Id"]):
        train_c, test_c = cuad.iloc[tr], cuad.iloc[te]
        cs = cuad_sample(train_c)
        X = pd.concat([hand["Clause"], cs["Clause"]])
        y = pd.concat([hand["Category"], cs["Category"]])
        w = np.r_[np.ones(len(hand)), np.full(len(cs), CUAD_SAMPLE_WEIGHT)]
        model = _fit(X, y, w)
        test = cuad_sample(test_c, cap=120, seed=1)
        truth += list(test["Category"])
        with_cuad += list(model.predict(test["Clause"]))
        without += list(base.predict(test["Clause"]))
    labels = sorted(set(truth))
    out = {
        "test_clauses": len(truth), "contracts": int(cuad["Contract_Id"].nunique()), "categories": labels,
        "accuracy": round(float(accuracy_score(truth, with_cuad)), 3),
        "macro_f1": round(float(f1_score(truth, with_cuad, labels=labels, average="macro")), 3),
        "hand_only_accuracy": round(float(accuracy_score(truth, without)), 3),
        "hand_only_macro_f1": round(float(f1_score(truth, without, labels=labels, average="macro")), 3),
        "settings": {"cuad_cap_per_category": CUAD_CAP_PER_CATEGORY, "cuad_sample_weight": CUAD_SAMPLE_WEIGHT},
    }
    if verbose:
        print("\n===== CATEGORY on REAL CONTRACTS never seen in training =====")
        print(out)
    return out


def train(verbose=True, use_cuad=True) -> dict:
    df = load_dataset()
    X = df["Clause"]
    cuad = load_cuad() if use_cuad else pd.DataFrame(columns=["Clause", "Category", "Contract_Id"])
    cs = cuad_sample(cuad)
    extra = (cs["Clause"], cs["Category"], CUAD_SAMPLE_WEIGHT) if len(cs) else None

    metrics = {
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "rows": int(len(df)),
        "sklearn_version": sklearn.__version__,
        "category_counts": df["Category"].value_counts().to_dict(),
        "risk_counts": df["Risk_Level"].value_counts().to_dict(),
        "cuad": {"used": bool(len(cs)), "clauses_in_file": int(len(cuad)), "clauses_used_in_training": int(len(cs)),
                 "contracts": int(cuad["Contract_Id"].nunique()) if len(cuad) else 0,
                 "sample_weight": CUAD_SAMPLE_WEIGHT, "cap_per_category": CUAD_CAP_PER_CATEGORY},
        "category": _evaluate("category", X, df["Category"], extra, verbose),
        "risk": _evaluate("risk", X, df["Risk_Level"], None, verbose),
    }
    metrics["risk_hybrid"] = _evaluate_hybrid(X, df["Risk_Level"], verbose)
    if len(cuad):
        metrics["real_contracts"] = _evaluate_real_contracts(df, cuad, verbose)

    MODEL_DIR.mkdir(exist_ok=True)
    if extra is not None:
        Xc = pd.concat([X, cs["Clause"]])
        yc = pd.concat([df["Category"], cs["Category"]])
        wc = np.r_[np.ones(len(df)), np.full(len(cs), CUAD_SAMPLE_WEIGHT)]
        cat = _fit(Xc, yc, wc)
    else:
        cat = _fit(X, df["Category"])
    risk = _fit(X, df["Risk_Level"])          # risk labels come from this project only
    joblib.dump(cat, CATEGORY_MODEL_PATH)
    joblib.dump(risk, RISK_MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    if verbose:
        print(f"\nSaved models to {MODEL_DIR}")
    return metrics


if __name__ == "__main__":
    try:
        train()
    except Exception as exc:  # pragma: no cover
        print("Training failed:", exc)
        sys.exit(1)
