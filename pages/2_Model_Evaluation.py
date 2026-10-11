"""Model evaluation, dataset statistics and the feedback/retrain loop."""
import json

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.config import DISCLAIMER, METRICS_PATH
from src.feedback import feedback_count, load_feedback, retrain_with_feedback

st.set_page_config(page_title="Model Evaluation", layout="wide")
st.title("📊 Model Evaluation")
st.caption("⚠ " + DISCLAIMER)

if not METRICS_PATH.exists():
    st.warning("No metrics found yet. Run `python -m src.train_model` (or use the retrain button below).")
    m = None
else:
    m = json.loads(METRICS_PATH.read_text())


def heatmap(conf: dict, title: str):
    labels, mat = conf["labels"], conf["matrix"]
    n = len(labels)
    fig, ax = plt.subplots(figsize=(max(3.5, 0.6 * n + 2), max(3, 0.55 * n + 1.5)))
    ax.imshow(mat, cmap="Blues")
    ax.set_xticks(range(n), [l.replace("_", " ") for l in labels], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n), [l.replace("_", " ") for l in labels], fontsize=8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title, fontsize=10)
    top = max(max(r) for r in mat) or 1
    for i in range(n):
        for j in range(n):
            ax.text(j, i, mat[i][j], ha="center", va="center", fontsize=8,
                    color="white" if mat[i][j] > top / 2 else "black")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def per_class_table(d: dict) -> pd.DataFrame:
    return pd.DataFrame([{"Class": k.replace("_", " "), "Precision": v["precision"], "Recall": v["recall"],
                          "F1": v["f1-score"], "Support": v["support"]} for k, v in d.items()])


if m:
    hy = m["risk_hybrid"]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Training clauses", m["rows"])
    k2.metric("Category accuracy", f"{m['category']['accuracy']:.0%}", f"baseline {m['category']['majority_baseline']:.0%}", delta_color="off")
    k3.metric("Risk accuracy (ML + rules)", f"{hy['hybrid_accuracy']:.0%}", f"ML only {hy['ml_only_accuracy']:.0%}", delta_color="off")
    k4.metric("Last trained", m.get("trained_at", "n/a")[:16].replace("T", " "))
    st.caption(f"All scores are {m['category']['cv_folds']}-fold stratified cross-validation, so every clause is scored by a model that never saw it. "
               "'Baseline' = always guessing the most common class.")

    tab1, tab2, tab_real, tab3 = st.tabs(["Category model", "Risk model", "Real contracts (CUAD)", "Dataset"])
    with tab1:
        a, b = st.columns([1, 1])
        with a:
            st.dataframe(per_class_table(m["category"]["per_class"]), hide_index=True)
        with b:
            heatmap(m["category"]["confusion"], "Category - confusion matrix")
    with tab2:
        a, b = st.columns([1, 1])
        with a:
            st.dataframe(per_class_table(m["risk"]["per_class"]), hide_index=True)
            st.write(f"**ML only:** {hy['ml_only_accuracy']:.1%} accuracy  \n"
                     f"**ML + rules (used by the app):** {hy['hybrid_accuracy']:.1%} accuracy, macro-F1 {hy['hybrid_macro_f1']:.2f}")
        with b:
            heatmap(m["risk"]["confusion"], "Risk (ML only) - confusion matrix")
        st.caption("Most errors are between neighbouring levels (Medium vs High / Low), which is expected for ordinal risk.")
    with tab_real:
        rc = m.get("real_contracts")
        if not rc:
            st.info("CUAD data is not installed, so there is no real-contract test. Run `python -m src.cuad_import --download` then retrain.")
        else:
            r1, r2, r3 = st.columns(3)
            r1.metric("Category accuracy, unseen real contracts", f"{rc['accuracy']:.0%}",
                      f"{rc['accuracy'] - rc['hand_only_accuracy']:+.0%} vs. without CUAD", delta_color="normal")
            r2.metric("Macro-F1", f"{rc['macro_f1']:.2f}", f"{rc['macro_f1'] - rc['hand_only_macro_f1']:+.2f} vs. without CUAD", delta_color="normal")
            r3.metric("Test clauses / contracts", f"{rc['test_clauses']:,} / {rc['contracts']}")
            st.write(
                f"Clauses were taken from **{rc['contracts']} real SEC-filed contracts** annotated by lawyers (CUAD). "
                "Each fold is tested on contracts the model never saw, and the same test clauses are scored by a model trained "
                f"*without* CUAD (**{rc['hand_only_accuracy']:.0%}** accuracy) - this is how much harder real contracts are than the "
                "short sample clauses the project started with.")
            st.caption("Covers " + ", ".join(c.replace("_", " ") for c in rc["categories"]) + ". CUAD has no confidentiality, indemnification or "
                       "security-deposit clauses, and no risk levels, so those parts rely on this project's own labels. "
                       "Settings were tuned on these folds, so treat the numbers as slightly optimistic.")
            st.warning("Not measured by this test: whole-contract behaviour. Only ~11% of clauses in a real contract are of a type lawyers flag; "
                       "the rest (definitions, boilerplate) still get a forced category, which dilutes the overall score on long documents. "
                       "An 'Other' class built from unannotated CUAD text was tried and rejected (see docs/CUAD.md). "
                       "Also, CUAD has no risk labels, so the RISK model is unchanged: on one real contract it rated ~40% of clauses High, "
                       "which is more than a lawyer would.")
    with tab3:
        a, b = st.columns(2)
        a.write("**Clauses per category**")
        a.bar_chart(pd.Series(m["category_counts"]))
        b.write("**Clauses per risk level**")
        b.bar_chart(pd.Series(m["risk_counts"]))
        cu = m.get("cuad", {})
        if cu.get("used"):
            st.caption(f"The category model additionally trains on {cu['clauses_used_in_training']} real clauses sampled from CUAD "
                       f"(of {cu['clauses_in_file']:,} extracted from {cu['contracts']} contracts; {cu['cap_per_category']} per category, "
                       f"weight {cu['sample_weight']}). The risk model uses only the project's own labels.")

    with st.expander("⚠ How much should you trust these numbers?"):
        st.write(
            "- The dataset is small (a few hundred clauses) and labelled by one person - not lawyer-reviewed.\n"
            "- Cross-validation folds share vocabulary, so accuracy on real, longer contracts will be lower.\n"
            "- The rule layer was designed while looking at this data, so its gain is slightly optimistic.\n"
            "- Use the feedback loop below, or a larger corpus such as CUAD, to improve the model.")

st.markdown("---")
st.subheader("✏ Feedback loop")
n = feedback_count()
st.write(f"**{n}** user correction(s) saved. Corrections are added to the training data and override older labels for the same clause. "
         "Submit corrections from the **Clauses** page of the main app (or the API `/feedback` endpoint).")
if n:
    st.dataframe(load_feedback().tail(10)[["Clause", "Category", "Risk_Level", "Predicted_Category", "Predicted_Risk", "Timestamp"]],
                 hide_index=True)
if st.button(f"🔁 Retrain with {n} correction(s)", disabled=(n == 0)):
    with st.spinner("Rebuilding dataset and retraining…"):
        retrain_with_feedback()
    st.success("Models retrained. The new metrics are shown above.")
    st.rerun()
