import json

from src.config import METRICS_PATH
from src.io_utils import md_escape


def test_metrics_have_evaluation_fields():
    m = json.loads(METRICS_PATH.read_text())
    for task in ("category", "risk"):
        conf = m[task]["confusion"]
        assert len(conf["labels"]) == len(conf["matrix"]) and sum(map(sum, conf["matrix"])) == m["rows"]
        assert all({"precision", "recall", "f1-score", "support"} <= set(v) for v in m[task]["per_class"].values())
    assert "hybrid_accuracy" in m["risk_hybrid"] and m["trained_at"]


def test_md_escape_blocks_links_and_images():
    out = md_escape("![x](http://evil/p.png) **bold** <b>")
    assert "![" not in out.replace("\\![", "") and "\\[" in out and "\\*" in out
