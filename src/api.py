"""REST API.   Run with:   uvicorn src.api:app --reload      (docs at http://127.0.0.1:8000/docs)"""
import json
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.agreement_analyzer import analyze_agreement, serialize_result
from src.compare import compare_agreements
from src.config import DISCLAIMER, METRICS_PATH
from src.feedback import feedback_count, save_feedback
from src.predict import get_models, predict_clause

MAX_CHARS = 300_000

app = FastAPI(title="Legal Risk Detection API", version="4.0",
              description="Clause classification, risk scoring and suggestions. " + DISCLAIMER)


class AnalyzeRequest(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_CHARS)


class ClauseRequest(BaseModel):
    clause: str = Field(min_length=3, max_length=5000)


class CompareRequest(BaseModel):
    text_a: str = Field(min_length=1, max_length=MAX_CHARS)
    text_b: str = Field(min_length=1, max_length=MAX_CHARS)


class FeedbackRequest(BaseModel):
    clause: str = Field(min_length=3, max_length=1500)
    correct_category: str
    correct_risk: Literal["Low", "Medium", "High"]
    predicted_category: str = ""
    predicted_risk: str = ""


@app.get("/health")
def health():
    try:
        get_models()
        return {"status": "ok", "models_loaded": True}
    except Exception:
        return {"status": "degraded", "models_loaded": False}


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    result = analyze_agreement(req.text)
    if result["total"] == 0:
        raise HTTPException(status_code=422, detail="No clauses were detected in the text.")
    return serialize_result(result)


@app.post("/predict-clause")
def predict(req: ClauseRequest):
    return predict_clause(req.clause).as_dict()


@app.post("/compare")
def compare(req: CompareRequest):
    return compare_agreements(req.text_a, req.text_b)


@app.post("/feedback")
def feedback(req: FeedbackRequest):
    try:
        save_feedback(req.clause, req.correct_category, req.correct_risk,
                      req.predicted_category, req.predicted_risk)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return {"saved": True, "total_feedback": feedback_count()}


@app.get("/metrics")
def metrics():
    if not METRICS_PATH.exists():
        raise HTTPException(status_code=404, detail="No metrics yet. Run: python -m src.train_model")
    return json.loads(METRICS_PATH.read_text())
