import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from src import api, config, feedback  # noqa: E402

client = TestClient(api.app)
TEXT = ("1. The Client shall pay invoices within 30 days.\n2. The Company may terminate this Agreement without notice.\n"
        "3. The Client shall indemnify and hold harmless the Provider.")


def test_health():
    assert client.get("/health").json() == {"status": "ok", "models_loaded": True}


def test_analyze():
    r = client.post("/analyze", json={"text": TEXT})
    assert r.status_code == 200
    d = r.json()
    assert d["total"] == 3 and d["high"] >= 1 and "analysis" in d and "action_plan" in d and d["disclaimer"]
    assert all("suggestions" in p for p in d["predictions"])


def test_analyze_validation():
    assert client.post("/analyze", json={"text": ""}).status_code == 422
    assert client.post("/analyze", json={"text": "Hello."}).status_code == 422       # no clauses detected
    assert client.post("/analyze", json={}).status_code == 422


def test_predict_clause():
    d = client.post("/predict-clause", json={"clause": "The Company may terminate without notice."}).json()
    assert d["risk"] == "High" and d["category"] == "Termination"


def test_compare():
    d = client.post("/compare", json={"text_a": TEXT, "text_b": TEXT}).json()
    assert d["delta"] == 0 and d["counts"]["unchanged"] == 3


def test_metrics():
    d = client.get("/metrics").json()
    assert "risk_hybrid" in d and "confusion" in d["category"]


def test_feedback(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "FEEDBACK_PATH", tmp_path / "fb.csv")
    ok = client.post("/feedback", json={"clause": "The tenant shall pay rent monthly in advance",
                                        "correct_category": "Payment", "correct_risk": "Low"})
    assert ok.status_code == 200 and ok.json() == {"saved": True, "total_feedback": 1}
    bad = client.post("/feedback", json={"clause": "The tenant shall pay rent monthly in advance",
                                         "correct_category": "Nope", "correct_risk": "Low"})
    assert bad.status_code == 422
    assert client.post("/feedback", json={"clause": "The tenant shall pay rent monthly in advance",
                                          "correct_category": "Payment", "correct_risk": "Extreme"}).status_code == 422
