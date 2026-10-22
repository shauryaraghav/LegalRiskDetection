# Legal Risk Detection System

A contract-review assistant that reads a legal agreement, breaks it into clauses, works out what each clause is about, rates how risky it is for the weaker party, and suggests what to ask for instead.

You can use it three ways: a Streamlit dashboard, a REST API, or the command line.

> **Disclaimer:** this is an educational NLP project. It is not legal advice and is no substitute for a lawyer.

---

## What it does

- **Reads** PDF, TXT or pasted text and splits it into clauses (handles numbering, wrapped lines and abbreviations such as "e.g." or "Inc.").
- **Classifies** each clause into one of 11 categories: Payment, Termination, Penalty, Liability, Indemnification, Confidentiality, Jurisdiction, Restriction, Security Deposit, Intellectual Property and Administrative.
- **Rates risk** as Low, Medium or High using a machine-learning model combined with a rule engine, and explains the decision (the rule that fired and the words that influenced the model).
- **Suggests improvements** for risky clauses, each with an issue, an action to ask for, and example wording. It also flags protections that are missing from the whole document, such as a liability cap or a notice period.
- **Scores the agreement** with a breakdown by category, "what if we renegotiated this?" scenarios, and a reliability indicator.
- **Compares two versions** of a contract and shows which clauses were added, removed or changed, and whether the revision is riskier.
- **Learns from corrections:** wrong predictions can be fixed in the UI and fed back into retraining.
- **Exports** a PDF report.
- **Optional:** an "Explain with AI" button that uses an LLM to explain a single clause in plain language (off by default).

## Getting started

```bash
git clone <your-repo-url>
cd Legal-Risk-Detection-System
pip install -r requirements.txt
```

Trained models are included in `models/`. If they are missing, or were built with a different scikit-learn version, they are retrained automatically the first time you run the app.

### Dashboard

```bash
streamlit run app.py
```

Upload a contract (or paste text) and click **Analyze Agreement**. The sidebar has the Dashboard, Clauses, Risk Analysis, Suggestions and Insights views, plus two extra pages: **Compare Contracts** and **Model Evaluation**.

### REST API

```bash
uvicorn src.api:app --reload
```

Interactive docs are at `http://127.0.0.1:8000/docs`.

| Endpoint | Purpose |
|---|---|
| `GET /health` | Service and model status |
| `POST /analyze` | Full analysis of an agreement |
| `POST /predict-clause` | Category, risk and suggestions for one clause |
| `POST /compare` | Compare two versions of an agreement |
| `POST /feedback` | Submit a corrected label |
| `GET /metrics` | Latest evaluation metrics |

```bash
curl -X POST localhost:8000/predict-clause \
  -H 'Content-Type: application/json' \
  -d '{"clause": "The Company may terminate employment without prior notice."}'
```

### Command line

```bash
python -m src.agreement_analyzer contract.txt
```

## How it works

```
contract text
     │
     ▼
clause splitter
     │
     ▼
TF-IDF features (word 1-2 grams + character 3-5 grams)
     ├──► Logistic Regression ──► category + probability
     └──► Logistic Regression ──► ML risk level ──┐
                                                  ├──► final risk + explanation + suggestions
rule engine (whole-word patterns, negation) ──────┘
```

**Hybrid risk decision.** A strong phrase such as "without notice", "non-refundable" or "forfeit" makes a clause High. A moderate keyword can raise a Low clause to Medium unless the model is at least 70% sure it is Low. Mitigating language ("mutual", "reasonable", "cure period") can soften a High the model was unsure about. Rules never override a confident model prediction in the other direction.

**Risk score.** High counts 2 points, Medium 1, Low 0. The score is `points / (2 × clauses) × 100`, with bands Low (< 35), Medium (35–60) and High (≥ 60).

## Data

| Source | Used for | Size |
|---|---|---|
| Hand-labelled clauses (`data/hand_labeled_clauses.csv`, `data/additional_clauses.csv`) | Category and risk models | 259 clauses |
| [CUAD](https://github.com/TheAtticusProject/cuad) clauses (`data/cuad_clauses.csv`) | Category model only | 6,362 clauses from 469 real contracts |
| User corrections (`data/user_feedback.csv`) | Both models, on retraining | grows with use |

CUAD contains lawyer-annotated clause *types* but no risk ratings, so the risk model is trained only on the hand-labelled data, labelled with the rubric in [`docs/LABELING.md`](docs/LABELING.md). How CUAD is used, and the experiments behind those choices, are in [`docs/CUAD.md`](docs/CUAD.md).

## Results

Measured with cross-validation; contract-level results are grouped by contract so no contract appears in both training and test data.

| Task | Accuracy | Macro-F1 |
|---|---:|---:|
| Category, own clauses (5-fold CV) | 0.79 | 0.80 |
| Category, unseen real contracts (CUAD) | 0.82 (0.54 without CUAD) | 0.81 |
| Risk, ML only (5-fold CV) | 0.62 | 0.61 |
| Risk, ML + rules (as shipped) | 0.65 | 0.63 |

The **Model Evaluation** page shows per-class scores and confusion matrices.

## Limitations

- **Risk labels are the weak point.** They come from about 260 clauses labelled by hand and have not been reviewed by a lawyer.
- **Long real contracts are over-flagged.** The risk model was trained on short clauses; on one 54,000-character agreement it rated about 40% of clauses High. The app shows a warning for documents with 60 or more clauses.
- **Only about 11% of clauses in a real contract are of a kind lawyers flag.** Definitions and boilerplate still receive a category, which dilutes the overall score.
- **The rules are not party-aware.** A mutual indemnity, or a clause that favours the other side, can still be flagged.
- The CUAD tuning was done on the same folds used to report the real-contract figure, so that number is slightly optimistic.
- Scanned (image-only) PDFs are not supported.

## Optional: AI explanations

Turn on **Enable "Explain with AI"** in the sidebar and provide an Anthropic API key (or set `ANTHROPIC_API_KEY`). Only the clause you click is sent, nothing is sent automatically, and the built-in explanation keeps working if the call fails. Contract text is treated as data, never as instructions, and the reply is shown as plain text. Because clause text leaves your machine, leave this off for confidential documents. The model is configurable with `LEGAL_LLM_MODEL`. This path is covered by unit tests with a fake client and has not been run against the live API.

## Project structure

```
app.py                      Streamlit dashboard
pages/                      Compare Contracts, Model Evaluation
src/
  clause_splitter.py        clause extraction
  preprocess.py             text normalisation
  rules.py                  rule engine
  predict.py                hybrid prediction and explanations
  suggestions.py            negotiation suggestions, missing-protection checks
  risk_analysis.py          score breakdown, what-if scenarios, reliability
  agreement_analyzer.py     agreement-level analysis (library and CLI)
  compare.py                two-version comparison
  feedback.py               corrections and retraining
  report.py                 PDF export
  api.py                    FastAPI service
  llm_explainer.py          optional LLM explanations
  cuad_import.py            builds data/cuad_clauses.csv from CUAD
  merge_dataset.py          builds the training dataset
  train_model.py            training and evaluation
  config.py, io_utils.py    shared constants and helpers
data/                       datasets
models/                     trained models and metrics.json
docs/                       labeling rubric, CUAD notes
tests/                      pytest suite (72 tests)
```

## Development

```bash
python -m pytest -q                 # run the tests
python -m src.merge_dataset         # rebuild the dataset from data/*.csv
python -m src.train_model           # retrain and re-run both evaluations (~30 s)
python -m src.cuad_import --download   # re-download CUAD and rebuild data/cuad_clauses.csv
```

See [`CHANGELOG.md`](CHANGELOG.md) for the history of changes.

## Roadmap

Lawyer-reviewed risk labels on real contract clauses, a labelled "not a key clause" class, party-aware rules, a fine-tuned legal language model, OCR for scanned PDFs, contract-type detection, and Docker/CI.

## Acknowledgements and license

This project is released under the MIT License; the original copyright notice is kept in [`LICENSE`](LICENSE).

Real-contract training data comes from **CUAD** (Hendrycks, Burns, Chen & Ball, 2021), released by The Atticus Project under CC BY 4.0. See [`data/CUAD_ATTRIBUTION.md`](data/CUAD_ATTRIBUTION.md).
