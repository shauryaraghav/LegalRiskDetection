"""Central paths and constants."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

DATASET_PATH = DATA_DIR / "final_legal_dataset.csv"
CATEGORY_MODEL_PATH = MODEL_DIR / "category_model.joblib"
RISK_MODEL_PATH = MODEL_DIR / "risk_model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"
FEEDBACK_PATH = DATA_DIR / "user_feedback.csv"
CUAD_PATH = DATA_DIR / "cuad_clauses.csv"

CATEGORIES = ["Administrative", "Confidentiality", "Indemnification", "Intellectual_Property", "Jurisdiction", "Liability",
              "Payment", "Penalty", "Restriction", "Security_Deposit", "Termination"]
RISK_LEVELS = ["Low", "Medium", "High"]          # ordered
RISK_RANK = {"Low": 0, "Medium": 1, "High": 2}

DISCLAIMER = (
    "This tool is an educational NLP prototype. It is not legal advice and "
    "must not replace review by a qualified lawyer."
)

# CUAD (real, lawyer-annotated contracts) is used for the CATEGORY model only.
# Settings chosen by experiment (see docs/CUAD.md): a few real examples per category, down-weighted
# so they broaden the model's vocabulary without drowning the project's own labelled data.
CUAD_CAP_PER_CATEGORY = 100
CUAD_SAMPLE_WEIGHT = 0.3
