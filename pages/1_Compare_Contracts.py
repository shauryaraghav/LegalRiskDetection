"""Compare two versions of an agreement."""
import streamlit as st

from src.compare import compare_agreements
from src.config import DISCLAIMER
from src.io_utils import read_upload

st.set_page_config(page_title="Compare Contracts", layout="wide")
st.title("🔀 Compare Two Contracts")
st.caption("⚠ " + DISCLAIMER)
st.write("Upload an **original** and a **revised** version to see which clauses were added, removed or changed, "
         "and whether the revision made the agreement riskier.")

ICON = {"added": "🆕 Added", "removed": "🗑 Removed", "modified": "✏ Modified", "unchanged": "＝ Unchanged"}
RISK_ICON = {"High": "🔴", "Medium": "🟡", "Low": "🟢", "": "—"}


def input_box(label: str, key: str) -> str:
    st.subheader(label)
    up = st.file_uploader("PDF or TXT", type=["pdf", "txt"], key=f"up_{key}")
    pasted = st.text_area("…or paste text", height=120, key=f"txt_{key}")
    if up is not None:
        try:
            return read_upload(up)
        except Exception as exc:
            st.error(f"Could not read the file: {exc}")
            return ""
    return pasted


col_a, col_b = st.columns(2)
with col_a:
    text_a = input_box("Version A (original)", "a")
with col_b:
    text_b = input_box("Version B (revised)", "b")

if st.button("🔍 Compare versions", disabled=not (text_a.strip() and text_b.strip()), use_container_width=True):
    with st.spinner("Analyzing and matching clauses…"):
        st.session_state.cmp = compare_agreements(text_a, text_b)

c = st.session_state.get("cmp")
if not c:
    st.info("Provide both versions, then click **Compare versions**.")
    st.stop()

m1, m2, m3, m4 = st.columns(4)
m1.metric("Score A", f"{c['score_a']:.1f}%", c["band_a"], delta_color="off")
m2.metric("Score B", f"{c['score_b']:.1f}%", c["band_b"], delta_color="off")
m3.metric("Change", f"{c['delta']:+.1f} pts", "riskier" if c["delta"] > 2 else "safer" if c["delta"] < -2 else "similar",
          delta_color="inverse" if c["delta"] != 0 else "off")
m4.metric("High-risk clauses", f"{c['high_a']} → {c['high_b']}")

if c["delta"] > 2:
    st.error(c["summary"][0])
elif c["delta"] < -2:
    st.success(c["summary"][0])
else:
    st.info(c["summary"][0])
for line in c["summary"][1:]:
    st.write("• " + line)

show_unchanged = st.checkbox("Show unchanged clauses", value=False)
st.markdown("### Clause changes")
shown = 0
for r in c["rows"]:
    if r["status"] == "unchanged" and not show_unchanged:
        continue
    shown += 1
    trend = {"riskier": "⬆ riskier", "safer": "⬇ safer", "same": ""}[r["trend"]]
    risk = f"{RISK_ICON[r['risk_a']]} {r['risk_a'] or '—'} → {RISK_ICON[r['risk_b']]} {r['risk_b'] or '—'}"
    preview = (r["clause_b"] or r["clause_a"])[:80]
    with st.expander(f"{ICON[r['status']]} · {risk} {trend} · {preview}…"):
        if r["clause_a"]:
            st.caption("Version A")
            st.text(r["clause_a"])          # st.text: untrusted contract text is never rendered as markdown/HTML
        if r["clause_b"]:
            st.caption("Version B")
            st.text(r["clause_b"])
        if r["removed_words"]:
            st.write("**Words removed:** " + ", ".join(f"`{w}`" for w in r["removed_words"][:15]))
        if r["added_words"]:
            st.write("**Words added:** " + ", ".join(f"`{w}`" for w in r["added_words"][:15]))
        if r["category_a"] and r["category_b"] and r["category_a"] != r["category_b"]:
            st.write(f"Category changed: {r['category_a']} → {r['category_b']}")
if shown == 0:
    st.success("No differences found between the two versions.")
