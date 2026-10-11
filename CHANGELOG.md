# Changes in v5 (on top of v4)

**CUAD integration**: 6,362 real lawyer-annotated clauses (469 contracts) extracted by `src/cuad_import.py`; category model trains on a down-weighted sample.
**Evaluation**: new grouped-by-contract test on unseen real contracts (82% vs 54% without CUAD); in-domain CV now adds CUAD to training folds only.
**New category**: `Intellectual_Property` (+14 hand-labelled examples, suggestions fallback).
**Honesty**: rejected weak CUAD-derived risk labels and an "Other" class after testing; documented that the risk model over-flags long real contracts; long-document warning in the app.
**Docs**: docs/CUAD.md, data/CUAD_ATTRIBUTION.md. **Tests**: 72.

# Changes in v4 (on top of v3)

**Feedback loop**: correct predictions in the UI or API; corrections are validated, stored, override older labels and are used on retrain.
**Model Evaluation page**: per-class metrics, confusion matrices, ML vs ML+rules, dataset stats, retrain button (metrics.json now stores these).
**REST API (FastAPI)**: /health /analyze /predict-clause /compare /feedback /metrics.
**Compare contracts**: clause matching across versions, word-level diffs, risk trend and score change.
**Optional AI explanations**: on-demand, opt-in, prompt-injection-aware, safe fallbacks, response caching.
**Security**: untrusted contract text is escaped before being shown as markdown/HTML.
**Tests**: 64 pytest tests (29 new).

# Changes in v3 (on top of v2)

**New - Suggestions**: per-clause negotiation suggestions with example wording, prioritised action plan, "possibly missing protections" check.
**New - Risk score analysis**: per-category breakdown, share-of-risk drivers, distance to risk bands, what-if scenarios, score reliability.
**App**: new *Risk Analysis* and *Suggestions* pages; suggestions shown inside each clause; PDF report now includes both sections.
**Tests**: 35 pytest tests (13 new).

# Changes in v2

**Data**: removed 89 auto-generated rows whose risk labels were `random.choice()` noise; fixed 15 contradictory labels;
added 130 hand-labelled clauses (245 total, every category now has >=18 rows); added labeling rubric.
**Model**: train and inference now use identical preprocessing (single Pipeline); added char n-grams; switched to
Logistic Regression for real probabilities; stratified cross-validation + hybrid evaluation; joblib instead of pickle.
**Confidence**: real probabilities, no 65-95% clamp; "most critical clause" now ranks by risk, rule evidence, then confidence.
**Rules**: whole-word regex, negation, mitigating language, per-rule explanations; rules no longer blindly override confident ML output.
**Splitter**: abbreviation-safe (e.g., Inc., No.), numbered-clause aware, joins wrapped lines, no longer drops clauses containing "date"/"name".
**App**: fixed folder layout & imports, added missing `reportlab` dep, HTML-escaping, None-safe PDF extraction, analyze disabled without text,
paste-text input, case-insensitive highlighting, in-memory PDF (also fixes crash on `&`/`<` in clauses), cached models, disclaimer, no debug prints.
**Removed**: NLTK dependency, `generate_dataset.py` (random labels), duplicated analysis logic.
**Added**: 22 pytest tests.
