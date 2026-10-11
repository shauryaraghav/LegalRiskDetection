"""Clause-level prediction: ML models + rule engine -> explainable result."""
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from typing import Dict, List, Tuple

import joblib
import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from src.config import CATEGORY_MODEL_PATH, RISK_LEVELS, RISK_MODEL_PATH, RISK_RANK
from src.rules import RuleResult, evaluate_rules
from src.suggestions import suggest_for_clause


@dataclass
class ClausePrediction:
    clause: str
    category: str
    category_confidence: float      # model probability, 0-100
    ml_risk: str                    # what the ML model alone predicted
    risk: str                       # final (hybrid) risk
    risk_confidence: float          # model probability of the FINAL label, 0-100
    rule_score: int
    source: str
    explanation: str
    reasons: List[str] = field(default_factory=list)
    key_terms: List[str] = field(default_factory=list)
    suggestions: List[dict] = field(default_factory=list)   # negotiation ideas (Medium/High only)

    def as_dict(self) -> dict:
        return asdict(self)


# ------------------------------------------------------------------ models
@lru_cache(maxsize=1)
def get_models():
    """Load models; (re)train automatically if missing or incompatible."""
    try:
        return joblib.load(CATEGORY_MODEL_PATH), joblib.load(RISK_MODEL_PATH)
    except Exception:
        from src.train_model import train
        train(verbose=False)
        return joblib.load(CATEGORY_MODEL_PATH), joblib.load(RISK_MODEL_PATH)


# ------------------------------------------------------------ hybrid logic
def combine_risk(ml_label: str, probs: Dict[str, float], rules: RuleResult) -> Tuple[str, str]:
    """Merge ML output and rule evidence.

    * strong rule evidence (score >= 3)  -> High
    * moderate evidence (score >= 2)     -> at least Medium, unless the model is
                                            very sure (>=70%) the clause is Low
    * mitigating language + weak model   -> a High is softened to Medium
    * otherwise                          -> trust the ML model
    Rules never silently downgrade a confident ML prediction.
    """
    if rules.score >= 3:
        return "High", "Rule: strong risk phrase"
    if rules.score >= 2:
        if ml_label == "Low" and probs.get("Low", 0) < 0.70:
            return "Medium", "Rule: risk keyword raised the level"
        return ml_label, "ML model (rules agree)" if ml_label != "Low" else "ML model"
    if rules.mitigating and ml_label == "High" and probs.get("High", 0) < 0.60:
        return "Medium", "Rule: mitigating language lowered the level"
    return ml_label, "ML model"


# ------------------------------------------------------------ explanations
def top_terms(pipeline, text: str, label: str, k: int = 3) -> List[str]:
    """Words that pushed the linear model towards `label` (real model evidence)."""
    try:
        feats = pipeline.named_steps["features"]
        clf = pipeline.named_steps["clf"]
        X = feats.transform([text])
        idx = list(clf.classes_).index(label)
        contrib = X.toarray()[0] * clf.coef_[idx if clf.coef_.shape[0] > 1 else 0]
        names = feats.get_feature_names_out()
        out = []
        for i in np.argsort(contrib)[::-1]:
            name = names[i]
            if contrib[i] <= 0 or len(out) >= k:
                break
            if (name.startswith("w__") and " " not in name
                    and name[3:] not in ENGLISH_STOP_WORDS and name[3:] != "num"):
                out.append(name[3:])
        return out
    except Exception:
        return []


def build_explanation(risk: str, rules: RuleResult, terms: List[str]) -> str:
    if rules.reasons:
        text = "This clause " + "; ".join(rules.reasons[:3]) + "."
    elif risk == "High":
        text = "Wording is similar to one-sided or punitive terms in the training examples."
    elif risk == "Medium":
        text = "Contains obligations or restrictions that deserve a closer read."
    else:
        text = "Reads as standard terms with no major risk indicators."
    if rules.mitigating:
        text += " Softening language: " + ", ".join(rules.mitigating[:2]) + "."
    if terms:
        text += " Key model terms: " + ", ".join(terms) + "."
    return text


# ------------------------------------------------------------ main function
def predict_clause(clause: str) -> ClausePrediction:
    cat_model, risk_model = get_models()
    X = [clause]

    cat_probs = cat_model.predict_proba(X)[0]
    category = cat_model.classes_[int(np.argmax(cat_probs))]

    risk_p = risk_model.predict_proba(X)[0]
    risk_probs = {c: float(p) for c, p in zip(risk_model.classes_, risk_p)}
    ml_risk = max(risk_probs, key=risk_probs.get)

    rules = evaluate_rules(clause)
    final, source = combine_risk(ml_risk, risk_probs, rules)
    terms = top_terms(risk_model, clause, final if final in risk_probs else ml_risk)

    return ClausePrediction(
        clause=clause,
        category=str(category),
        category_confidence=round(float(cat_probs.max()) * 100, 1),
        ml_risk=ml_risk,
        risk=final,
        risk_confidence=round(risk_probs.get(final, 0.0) * 100, 1),
        rule_score=rules.score,
        source=source,
        explanation=build_explanation(final, rules, terms),
        reasons=rules.reasons,
        key_terms=terms,
        suggestions=[x.as_dict() for x in suggest_for_clause(str(category), final, rules)],
    )


def risk_rank(level: str) -> int:
    return RISK_RANK.get(level, 0)


if __name__ == "__main__":
    for t in ["The company may terminate employment without prior notice.",
              "Rent shall be paid on or before the 5th of every month."]:
        print(predict_clause(t).as_dict(), "\n")
