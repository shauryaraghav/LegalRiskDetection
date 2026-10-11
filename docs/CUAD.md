# Using CUAD (real, lawyer-annotated contracts)

**CUAD** - *Contract Understanding Atticus Dataset* (Hendrycks, Burns, Chen & Ball, 2021): 510 SEC-filed contracts in which lawyers
highlighted clauses of 41 types. Released by The Atticus Project under CC BY 4.0 (per the authors; please verify the licence at
https://github.com/TheAtticusProject/cuad). Cite: *CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review*, NeurIPS Datasets & Benchmarks 2021.

## What CUAD can and cannot do here
CUAD labels are **clause types**, not **risk levels**. So:

| Part of the project | Uses CUAD? | Why |
|---|---|---|
| Category model | **Yes** (down-weighted sample) | CUAD provides lawyer-labelled real clauses for 8 of our categories |
| Risk model | **No** | CUAD has no Low/Medium/High ratings. Risk labels are still this project's own (not lawyer-reviewed) |
| Confidentiality, Indemnification, Security deposit | No | CUAD does not annotate these types |

The mapping from CUAD types to this project's categories (`src/cuad_import.py::CATEGORY_MAP`) is a judgement made by this project.
Names, dates and fragments (Parties, Document Name, Agreement/Effective/Expiration Date, Warranty Duration) are not used.
Spans of 6-80 words are kept; duplicated boilerplate is removed. Result: 6,362 clauses from 469 contracts in `data/cuad_clauses.csv`.

## Experiments (so the choices are not guesswork)
All "unseen contract" numbers are grouped by contract (no contract appears in both training and test).

| Question | Result | Decision |
|---|---|---|
| Category accuracy on real contracts the model never saw | **54% without CUAD -> 82% with CUAD** (4,017 clauses, 469 contracts) | adopted |
| Does CUAD hurt our own short clauses? | No - per-class F1 improved for Liability (0.71 -> 0.83), Payment (0.70 -> 0.76), Penalty; accuracy 0.78 -> 0.79 | adopted |
| How much CUAD? | 100 clauses per category at weight 0.3 balanced our data (kept) against real-contract gains (more CUAD = better on real contracts, worse on our own) | 100 / 0.3 |
| Weak risk labels derived from CUAD types (e.g. "Uncapped Liability" -> High) | +2 points, within noise, and the labels would be mine, not a lawyer's | **rejected** |
| An "Other / not a key clause" class from unannotated CUAD text | Lost 27-31% of lawyer-flagged clauses, labelled 11-18% of our own clauses "Other", lowered accuracy 0.79 -> 0.71-0.76, doubled precision only from 11% to ~20% (CUAD annotates just 41 types, so *unannotated is not the same as irrelevant*) | **rejected** |

Settings were chosen using these same folds, so the real-contract figure is slightly optimistic.

## Whole-contract behaviour (important)
Tested on 25 full contracts: the clause splitter located 100% of the annotated clauses and the category was right for about 81% of them.
However only ~11% of clauses in a real contract are of a type lawyers flag; the rest (definitions, recitals, boilerplate) still receive a
forced category.

The **risk model over-flags real contracts**: on one 54,000-character distributor agreement the ML model alone rated 40% of the 197 clauses
High. It was trained on 259 short, blunt example clauses, and CUAD cannot correct this because it contains no risk ratings. The rule engine
is also not party-aware (a mutual indemnity or a clause giving the *distributor* discretion is still flagged). The app therefore shows a
warning on documents with 60+ clauses.

## Regenerate
```bash
python -m src.cuad_import --download     # fetch CUADv1.json from GitHub and rebuild data/cuad_clauses.csv
python -m src.train_model                # retrain + re-run both evaluations
```

## What would improve it next
1. Lawyer-reviewed **risk** labels on real contract clauses (the real bottleneck now).
2. A labelled "not a key clause" set (CUAD alone cannot supply one).
3. Party-aware rules (who benefits from the clause).
