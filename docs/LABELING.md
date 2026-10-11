# Labeling rubric

Risk is judged from the point of view of the party who did **not** draft the clause
(typically the tenant / employee / borrower / client / customer).

| Level  | Meaning | Typical signals |
|--------|---------|-----------------|
| High   | One-sided, punitive, uncapped or hard to reverse | termination without notice, forfeiture, non-refundable, unlimited liability, indemnity, compounding interest, waiver of rights, sole discretion |
| Medium | A real obligation, restriction or cost that deserves a careful read | exclusive jurisdiction / arbitration, capped liability, deductions, late fees, non-compete with limits, notice periods |
| Low    | Routine, balanced or administrative | payment within a stated period, mutual amendments in writing, counterparts, governing law, grace periods |

Rules of thumb used when relabeling the original data:
* every clause with the same wording gets the same label (the old data labelled "indemnify" clauses both Medium and High);
* a **cap** on liability is Medium, **unlimited** or **excluded** liability is High;
* standard confidentiality duties are Medium; perpetual / uncapped / penalty-backed ones are High;
* one category per clause (indemnity clauses are `Indemnification`, not `Liability`);
* `Intellectual_Property` covers ownership, assignment and licence terms; one-sided assignment of everything the person creates, or a perpetual irrevocable licence, is High.

## Honest limitations
* **Risk labels** (259 clauses), labelled by one person with a rubric - **not** validated by a lawyer.
* Many clauses are short, simple sentences; real contracts are longer and messier. Category learning is supplemented with real, lawyer-annotated CUAD clauses (see CUAD.md); risk labels are not.
* Cross-validation folds share vocabulary, so real-world accuracy will be lower.
* For production-quality results, use a larger public corpus such as CUAD and have
  the labels reviewed by legal professionals.
