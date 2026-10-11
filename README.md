# Legal Risk Detection & Clause Classification System ⚖️

An NLP prototype that splits a legal agreement (PDF / TXT / pasted text) into clauses,
classifies each clause (Payment, Termination, Penalty, Liability, Indemnification, ...),
rates its risk (High / Medium / Low), explains the decision, **suggests how to renegotiate risky clauses
(with example wording)** and gives a **detailed risk-score analysis**. A Streamlit dashboard shows the
results and exports a PDF report.

> ⚠ Educational prototype - **not legal advice**. See [docs/LABELING.md](docs/LABELING.md) for limitations.

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py                       # dashboard (+ Compare Contracts and Model Evaluation pages)
uvicorn src.api:app --reload               # REST API, interactive docs at http://127.0.0.1:8000/docs
python -m src.agreement_analyzer my.txt    # command-line report
python -m pytest -q                        # tests
```

Models are loaded from `models/`. If they are missing or were built with an incompatible
scikit-learn version, they are re-trained automatically on first use.

## New in v3: suggestions & risk-score analysis

**💡 Suggestions** (`src/suggestions.py`)
* Every Medium/High clause gets up to 3 negotiation ideas, each with an *issue*, an *action* and *example wording*
  (e.g. "without notice" -> ask for a 30-day written notice period).
* Suggestions come from the rule that flagged the clause, with a per-category fallback; protections already in the clause
  (a liability cap, a cure period, a grace period) suppress the matching suggestion.
* An agreement-level check reports **possibly missing protections** (notice, liability cap, dispute resolution,
  confidentiality, termination, cure period) for documents with 5+ clauses.
* Suggestions are ordered into an action plan, High risk first.

**📈 Risk score analysis** (`src/risk_analysis.py`)
* Formula: High = 2 points, Medium = 1, Low = 0; score = points / (2 x clauses) x 100; bands Low < 35, Medium 35-60, High >= 60.
* Per-category breakdown: how risky each category is, and how much of the overall score it explains.
* **What-if scenarios**: the score if you renegotiate the 3 worst clauses, soften every High to Medium, or fix every High to Low.
* **Reliability**: average model confidence, number of low-confidence clauses and rule-adjusted clauses.
* Everything is also included in the PDF report and the CLI output.

## New in v5: real-contract data (CUAD)

The category model now also learns from **real, lawyer-annotated clauses** from CUAD (469 SEC-filed contracts, `src/cuad_import.py`).

* On clauses from real contracts the model had never seen, category accuracy rose from **54% to 82%**.
* Added an `Intellectual_Property` category (CUAD has many licence/ownership clauses).
* **What it did not fix:** CUAD has *no risk ratings*, so the risk model is unchanged (~62% ML only, ~65% with rules), and on long real contracts
  it **over-flags** (one test contract: 40% of clauses rated High). The app warns on documents with 60+ clauses.
* Two ideas were tested and rejected with data (weak risk labels from CUAD, and an "Other" class) - see [docs/CUAD.md](docs/CUAD.md).
* Evaluation page has a new *Real contracts (CUAD)* tab. Data attribution: `data/CUAD_ATTRIBUTION.md`.

## New in v4

| Feature | Where | What it does |
|---|---|---|
| **Feedback loop** | Clauses page -> "Wrong prediction? Correct it"; `POST /feedback` | Saves corrections to `data/user_feedback.csv` (validated, spreadsheet-injection safe). *Model Evaluation -> Retrain* merges them into the dataset (corrections override older labels) and retrains. |
| **Model Evaluation page** | `pages/2_Model_Evaluation.py` | Cross-validated accuracy / macro-F1 vs baseline, per-class precision/recall/F1, confusion matrices, ML-only vs ML+rules, dataset balance, trust caveats, retrain button. |
| **REST API** | `src/api.py` | `GET /health`, `POST /analyze`, `POST /predict-clause`, `POST /compare`, `POST /feedback`, `GET /metrics` with validation and size limits. |
| **Compare contracts** | `pages/1_Compare_Contracts.py`, `src/compare.py` | Matches clauses between two versions (optimal TF-IDF matching + a rewrite pass), reports added / removed / modified / unchanged clauses, word-level diffs (numbers count), risk trend per clause and the overall score change. |
| **AI explanations (optional)** | sidebar toggle -> "Explain with AI", `src/llm_explainer.py` | On click only, sends one clause to the Anthropic API and shows a plain-English explanation plus tailored wording. Off by default; needs a key; falls back silently to the built-in explanation. |

API example:

```bash
curl -X POST localhost:8000/predict-clause -H 'Content-Type: application/json' \
     -d '{"clause": "The Company may terminate employment without prior notice."}'
```

**About the AI feature:** the clause is wrapped in tags and the prompt tells the model to treat it as data, never as
instructions; replies are shown as escaped text; errors never expose details. Using it sends clause text to a third-party
API, so leave it off for confidential contracts. The model name is configurable with `LEGAL_LLM_MODEL`.
The code is unit-tested with a fake client; it has not been exercised against the live API in this repository's test suite.

## How it works

```
text -> clause splitter -> TF-IDF (word + char n-grams) -> Logistic Regression  -> category + probabilities
                                                        -> Logistic Regression  -> ML risk ──┐
                                      rule engine (whole-word regex, negation, mitigation) ───┴─> hybrid risk + explanation
```

* **Hybrid decision** (`src/predict.py::combine_risk`): a strong rule phrase ("without notice",
  "non-refundable", "forfeit"...) => High; a moderate keyword raises Low to Medium only if the model is
  not >=70% sure; mitigating language ("mutual", "reasonable", "cure period") can soften a weak High.
* **Explanations** list the rule that fired *and* the words that pushed the linear model toward its answer.
* **Confidence** values are real model probabilities (no clamping).

## Project layout

```
app.py                      Streamlit dashboard (Dashboard, Clauses, Risk Analysis, Suggestions, Insights)
pages/                      extra Streamlit pages: Compare Contracts, Model Evaluation
src/
  config.py                 paths / constants
  clause_splitter.py        abbreviation-, numbering- and wrap-aware clause extraction
  preprocess.py             text normalisation (used inside the model pipeline)
  rules.py                  rule engine
  predict.py                hybrid prediction + explanations
  suggestions.py            negotiation suggestions + missing-protection checks
  risk_analysis.py          score breakdown, what-if scenarios, reliability
  agreement_analyzer.py     agreement-level analysis (library + CLI)
  report.py                 PDF export
  cuad_import.py            builds data/cuad_clauses.csv from CUAD
  compare.py                two-version comparison
  feedback.py               user corrections + retraining
  llm_explainer.py          optional LLM explanations
  api.py                    FastAPI service
  io_utils.py               file reading + markdown escaping
  train_model.py            training + cross-validated evaluation
  merge_dataset.py          builds data/final_legal_dataset.csv
data/                       hand-labelled clause CSVs + cuad_clauses.csv (derived from CUAD) (+ user_feedback.csv once you submit corrections)
models/                     trained pipelines (*.joblib) + metrics.json
tests/                      pytest suite
docs/LABELING.md            labeling rubric and dataset limitations
docs/CUAD.md                how CUAD is used, experiments, limitations
```

## Retrain

```bash
python -m src.merge_dataset     # rebuild the dataset from data/*.csv
python -m src.train_model       # cross-validate, print report, save models
```

## Measured performance

| Task | Accuracy | Macro-F1 |
|------|---------:|---------:|
| Category, own clauses (11 classes, 5-fold CV, 259 clauses) | 0.79 | 0.80 |
| **Category, unseen real contracts (CUAD, grouped by contract)** | **0.82** (0.54 without CUAD) | 0.81 |
| Risk, ML only (own clauses, 5-fold CV) | 0.62 | 0.61 |
| Risk, ML + rules (shipped) | 0.65 | 0.63 |

Risk numbers are modest because risk labels are still few and not lawyer-reviewed. Real-contract figures are slightly optimistic because the CUAD settings were tuned on the same folds.

Metrics move by a few points whenever the data changes (for example after retraining with corrections) because the dataset is small.

## Future work
Lawyer-reviewed risk labels on real clauses · a labelled 'not a key clause' set · party-aware rules · fine-tuned Legal-BERT · OCR for scanned PDFs · multilingual support · contract-type detection · entity extraction · Docker / CI.

## Author
Original project by **Taruni Middela**. v2 fixed data, modelling, rule-engine, splitter, packaging and app issues.
