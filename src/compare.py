"""Compare two versions of an agreement (e.g. original vs revised draft)."""
import difflib
import re
from typing import Dict, List

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.feature_extraction.text import TfidfVectorizer

from src.agreement_analyzer import analyze_agreement
from src.config import RISK_RANK
from src.risk_analysis import band_for

MATCH_THRESHOLD = 0.35       # minimum cosine similarity to treat two clauses as "the same clause"
REWRITE_THRESHOLD = 0.10     # looser threshold for leftover clauses of the SAME category (a rewritten clause)
SCORE_TOLERANCE = 2.0        # score changes smaller than this are called "about the same"


def _tokens(text: str) -> List[str]:
    """Lower-case word tokens. Numbers are KEPT (30 days vs 60 days is a real change)."""
    return re.findall(r"[a-z0-9]+", text.lower())


def _match(preds_a, preds_b):
    """Optimal one-to-one matching of clauses by TF-IDF cosine similarity."""
    if not preds_a or not preds_b:
        return []
    vec = TfidfVectorizer(tokenizer=_tokens, token_pattern=None, lowercase=False, ngram_range=(1, 2))
    vec.fit([p.clause for p in preds_a + preds_b])
    xa = vec.transform([p.clause for p in preds_a])
    xb = vec.transform([p.clause for p in preds_b])
    sim = (xa @ xb.T).toarray()
    rows, cols = linear_sum_assignment(-sim)
    pairs = [(i, j, float(sim[i, j])) for i, j in zip(rows, cols) if sim[i, j] >= MATCH_THRESHOLD]

    # pass 2: leftovers of the same category are most likely the same clause rewritten
    left_a = [i for i in range(len(preds_a)) if i not in {p[0] for p in pairs}]
    left_b = [j for j in range(len(preds_b)) if j not in {p[1] for p in pairs}]
    if left_a and left_b:
        sub = sim[np.ix_(left_a, left_b)].copy()
        for x, i in enumerate(left_a):
            for y, j in enumerate(left_b):
                if preds_a[i].category != preds_b[j].category:
                    sub[x, y] = 0.0
        r2, c2 = linear_sum_assignment(-sub)
        pairs += [(left_a[x], left_b[y], float(sub[x, y])) for x, y in zip(r2, c2) if sub[x, y] >= REWRITE_THRESHOLD]
    return pairs


def _word_diff(a: str, b: str):
    ta, tb = _tokens(a), _tokens(b)
    removed, added = [], []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, ta, tb).get_opcodes():
        if op in ("replace", "delete"):
            removed += ta[i1:i2]
        if op in ("replace", "insert"):
            added += tb[j1:j2]
    return removed, added


def compare_agreements(text_a: str, text_b: str) -> Dict:
    ra, rb = analyze_agreement(text_a), analyze_agreement(text_b)
    pa, pb = ra["predictions"], rb["predictions"]
    pairs = _match(pa, pb)
    matched_a = {i for i, _, _ in pairs}
    matched_b = {j for _, j, _ in pairs}

    rows = []
    for i, j, sim in pairs:
        a, b = pa[i], pb[j]
        if _tokens(a.clause) == _tokens(b.clause):
            status, removed, added = "unchanged", [], []
        else:
            status = "modified"
            removed, added = _word_diff(a.clause, b.clause)
        delta = RISK_RANK[b.risk] - RISK_RANK[a.risk]
        rows.append({"status": status, "clause_a": a.clause, "clause_b": b.clause,
                     "risk_a": a.risk, "risk_b": b.risk, "category_a": a.category, "category_b": b.category,
                     "trend": "riskier" if delta > 0 else "safer" if delta < 0 else "same",
                     "removed_words": removed, "added_words": added, "similarity": round(sim, 2)})
    for i, p in enumerate(pa):
        if i not in matched_a:
            rows.append({"status": "removed", "clause_a": p.clause, "clause_b": "", "risk_a": p.risk, "risk_b": "",
                         "category_a": p.category, "category_b": "",
                         "trend": "safer" if p.risk != "Low" else "same",
                         "removed_words": [], "added_words": [], "similarity": 0.0})
    for j, p in enumerate(pb):
        if j not in matched_b:
            rows.append({"status": "added", "clause_a": "", "clause_b": p.clause, "risk_a": "", "risk_b": p.risk,
                         "category_a": "", "category_b": p.category,
                         "trend": "riskier" if p.risk != "Low" else "same",
                         "removed_words": [], "added_words": [], "similarity": 0.0})

    # most important first: riskier changes, then safer, then neutral; unchanged last
    def importance(r):
        if r["status"] == "unchanged":
            return (3, 0)
        sev = RISK_RANK.get(r["risk_b"] or r["risk_a"], 0)
        return ({"riskier": 0, "safer": 1, "same": 2}[r["trend"]], -sev)
    rows.sort(key=importance)

    counts = {
        "unchanged": sum(r["status"] == "unchanged" for r in rows),
        "modified": sum(r["status"] == "modified" for r in rows),
        "added": sum(r["status"] == "added" for r in rows),
        "removed": sum(r["status"] == "removed" for r in rows),
        "riskier": sum(r["trend"] == "riskier" for r in rows),
        "safer": sum(r["trend"] == "safer" for r in rows),
    }
    score_a, score_b = ra["score"], rb["score"]
    delta = round(score_b - score_a, 1)
    if ra["total"] == 0 or rb["total"] == 0:
        verdict = "One of the documents has no detectable clauses, so no comparison is possible."
    elif delta > SCORE_TOLERANCE:
        verdict = f"Version B is RISKIER than A: score {score_a:.1f}% -> {score_b:.1f}% (+{delta} points)."
    elif delta < -SCORE_TOLERANCE:
        verdict = f"Version B is SAFER than A: score {score_a:.1f}% -> {score_b:.1f}% ({delta} points)."
    else:
        verdict = f"Both versions have about the same risk: {score_a:.1f}% vs {score_b:.1f}%."

    summary = [verdict]
    new_high = [r for r in rows if r["status"] == "added" and r["risk_b"] == "High"]
    gone_high = [r for r in rows if r["status"] == "removed" and r["risk_a"] == "High"]
    worse = [r for r in rows if r["status"] == "modified" and r["trend"] == "riskier"]
    better = [r for r in rows if r["status"] == "modified" and r["trend"] == "safer"]
    if new_high:
        summary.append(f"{len(new_high)} new High-risk clause(s) were added in B.")
    if worse:
        summary.append(f"{len(worse)} existing clause(s) became riskier in B.")
    if gone_high:
        summary.append(f"{len(gone_high)} High-risk clause(s) from A were removed in B.")
    if better:
        summary.append(f"{len(better)} existing clause(s) became safer in B.")
    summary.append(f"{counts['unchanged']} unchanged, {counts['modified']} modified, "
                   f"{counts['added']} added, {counts['removed']} removed.")

    return {
        "score_a": score_a, "score_b": score_b, "delta": delta,
        "band_a": band_for(score_a) if ra["total"] else "N/A",
        "band_b": band_for(score_b) if rb["total"] else "N/A",
        "high_a": ra["high"], "high_b": rb["high"],
        "counts": counts, "rows": rows, "summary": summary,
    }
