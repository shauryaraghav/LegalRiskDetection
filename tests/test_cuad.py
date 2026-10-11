import json

import pandas as pd

from src import config
from src.agreement_analyzer import analyze_agreement
from src.cuad_import import CATEGORY_MAP, extract
from src.predict import predict_clause
from src.suggestions import suggest_for_clause
from src.rules import evaluate_rules
from src.train_model import cuad_sample, load_cuad

LONG = "The Supplier shall indemnify no one and this sentence is deliberately long enough to pass the length filter easily."


def _mini(tmp_path):
    def qa(title, ctype, texts):
        return {"id": f"{title}__{ctype}", "question": "q", "is_impossible": not texts,
                "answers": [{"text": t, "answer_start": 0} for t in texts]}
    docs = [
        {"title": "A", "paragraphs": [{"context": "x", "qas": [
            qa("A", "Governing Law", ["This Agreement shall be governed by the laws of the State of New York without regard to conflicts."]),
            qa("A", "Parties", ["Acme Corporation and Beta Limited both parties here"]),           # unmapped type
            qa("A", "Cap On Liability", ["Too short here"]),                                         # < 6 words
            qa("A", "Non-Compete", ["word " * 120]),                                                  # > 80 words
            qa("A", "License Grant", ["Licensor grants Licensee a non-exclusive, non-transferable license to use the Software."]),
        ]}]},
        {"title": "B", "paragraphs": [{"context": "y", "qas": [
            qa("B", "Governing Law", ["This Agreement shall be governed by the laws of the State of New York without regard to conflicts."]),  # duplicate boilerplate
            qa("B", "Uncapped Liability", ["(a) The Supplier's liability under this Agreement shall be unlimited for any breach of confidentiality."]),
        ]}]},
    ]
    path = tmp_path / "cuad.json"
    path.write_text(json.dumps({"version": "t", "data": docs}))
    return path


def test_extract_maps_filters_and_dedupes(tmp_path):
    df = extract(_mini(tmp_path))
    assert set(df.columns) == {"Clause", "Category", "CUAD_Type", "Contract_Id"}      # no invented risk labels
    assert set(df.CUAD_Type) == {"Governing Law", "License Grant", "Uncapped Liability"}
    assert df.Category.tolist().count("Jurisdiction") == 1                              # duplicate removed
    assert df.loc[df.CUAD_Type == "License Grant", "Category"].iloc[0] == "Intellectual_Property"
    assert not df.Clause.str.startswith("(a)").any()                                    # numbering stripped


def test_every_mapped_category_is_known():
    assert set(CATEGORY_MAP.values()) <= set(config.CATEGORIES)


def test_cuad_sample_cap_and_reproducible():
    df = pd.DataFrame({"Clause": [f"clause number {i} text" for i in range(50)],
                       "Category": ["Payment"] * 30 + ["Liability"] * 20, "Contract_Id": "c"})
    a, b = cuad_sample(df, cap=10), cuad_sample(df, cap=10)
    assert a.Category.value_counts().to_dict() == {"Payment": 10, "Liability": 10}
    assert a.Clause.tolist() == b.Clause.tolist()


def test_missing_cuad_file_gives_empty_frame(tmp_path):
    assert load_cuad(tmp_path / "nope.csv").empty


def test_shipped_metrics_show_cuad_benefit():
    m = json.loads(config.METRICS_PATH.read_text())
    rc = m["real_contracts"]
    assert rc["accuracy"] > rc["hand_only_accuracy"] + 0.15 and rc["test_clauses"] > 1000
    assert m["cuad"]["used"] and "Intellectual_Property" in m["category_counts"]


def test_real_contract_style_clauses():
    assert predict_clause("Licensor hereby grants to Licensee a non-exclusive, non-transferable, worldwide license to use the Software "
                          "solely for its internal business purposes.").category == "Intellectual_Property"
    assert predict_clause("This Agreement shall be governed by and construed in accordance with the laws of the State of Delaware, "
                          "without giving effect to its conflict of laws principles.").category == "Jurisdiction"
    assert predict_clause("In no event shall either party's aggregate liability under this Agreement exceed the total fees "
                          "paid in the twelve months preceding the claim.").category == "Liability"


def test_ip_has_suggestion_fallback():
    s = suggest_for_clause("Intellectual_Property", "High", evaluate_rules("The Company owns everything the Employee ever thinks of"))
    assert s and "ownership" in s[0].issue.lower()


def test_long_document_warning():
    text = "\n".join(f"{i}. The tenant shall pay the monthly rent on the first day of month {i}." for i in range(70))
    notes = " ".join(analyze_agreement(text)["analysis"]["summary"])
    assert "Long document" in notes
    short = "1. The tenant shall pay the monthly rent on the first day of each month.\n2. Either party may terminate on 60 days written notice."
    assert "Long document" not in " ".join(analyze_agreement(short)["analysis"]["summary"])
