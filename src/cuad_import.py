"""Import clause examples from CUAD (Contract Understanding Atticus Dataset).

CUAD: Hendrycks, Burns, Chen & Ball (2021), "CUAD: An Expert-Annotated NLP Dataset for Legal
Contract Review", NeurIPS Datasets & Benchmarks.  510 SEC-filed contracts, annotated by lawyers
with 41 clause types.  https://github.com/TheAtticusProject/cuad  (released under CC BY 4.0).

IMPORTANT - what CUAD can and cannot give us
* CUAD labels are CLAUSE TYPES, not risk levels.  So it improves the CATEGORY model.
* CUAD types are mapped onto this project's categories below (a judgement made by this project, not by CUAD).
* Weak risk labels derived from CUAD types were tried and gave only ~+2 points (within noise),
  so they are NOT used.  The risk model is still trained on this project's own labels.

Usage:
    python -m src.cuad_import --download            # fetch CUADv1.json from GitHub, build data/cuad_clauses.csv
    python -m src.cuad_import --json path/CUADv1.json
"""
import argparse
import json
import re
import urllib.request
import zipfile
from io import BytesIO
from pathlib import Path

import pandas as pd

from src.config import DATA_DIR

CUAD_ZIP_URL = "https://github.com/The-Atticus-Project/cuad/raw/main/data.zip"
OUT_PATH = DATA_DIR / "cuad_clauses.csv"

# CUAD clause type -> this project's category.   None = not used (names, dates, fragments).
CATEGORY_MAP = {
    "Governing Law": "Jurisdiction",
    "Cap On Liability": "Liability",
    "Uncapped Liability": "Liability",
    "Insurance": "Liability",
    "Liquidated Damages": "Penalty",
    "Termination For Convenience": "Termination",
    "Notice Period To Terminate Renewal": "Termination",
    "Post-Termination Services": "Termination",
    "Renewal Term": "Administrative",
    "Anti-Assignment": "Administrative",
    "Change Of Control": "Administrative",
    "Audit Rights": "Administrative",
    "Third Party Beneficiary": "Administrative",
    "Non-Compete": "Restriction",
    "Exclusivity": "Restriction",
    "No-Solicit Of Employees": "Restriction",
    "No-Solicit Of Customers": "Restriction",
    "Competitive Restriction Exception": "Restriction",
    "Volume Restriction": "Restriction",
    "Price Restrictions": "Restriction",
    "Most Favored Nation": "Restriction",
    "Rofr/Rofo/Rofn": "Restriction",
    "Covenant Not To Sue": "Restriction",
    "Non-Disparagement": "Restriction",
    "Minimum Commitment": "Payment",
    "Revenue/Profit Sharing": "Payment",
    "License Grant": "Intellectual_Property",
    "Non-Transferable License": "Intellectual_Property",
    "Affiliate License-Licensee": "Intellectual_Property",
    "Affiliate License-Licensor": "Intellectual_Property",
    "Irrevocable Or Perpetual License": "Intellectual_Property",
    "Unlimited/All-You-Can-Eat-License": "Intellectual_Property",
    "Ip Ownership Assignment": "Intellectual_Property",
    "Joint Ip Ownership": "Intellectual_Property",
    "Source Code Escrow": "Intellectual_Property",
    # unused: Document Name, Parties, Agreement Date, Effective Date, Expiration Date, Warranty Duration
}

MIN_WORDS, MAX_WORDS, MAX_CHARS = 6, 80, 600


def _clean(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"^\(?[0-9a-zA-Z]{1,4}[.)]\s+(?=[A-Z])", "", text)      # leading "(a)" / "3.1" numbering
    return text.strip(" ;:,")


def extract(json_path: Path) -> pd.DataFrame:
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))["data"]
    rows = []
    for doc in data:
        title = doc["title"]
        for para in doc["paragraphs"]:
            for qa in para["qas"]:
                ctype = qa["id"].split("__")[-1]
                category = CATEGORY_MAP.get(ctype)
                if not category:
                    continue
                for ans in qa["answers"]:
                    text = _clean(ans["text"])
                    n = len(text.split())
                    if MIN_WORDS <= n <= MAX_WORDS and len(text) <= MAX_CHARS and text[:1].isalpha():
                        rows.append({"Clause": text, "Category": category, "CUAD_Type": ctype,
                                     "Contract_Id": title})
    df = pd.DataFrame(rows)
    # same text appearing in several contracts (boilerplate) -> keep one copy
    df["_key"] = df["Clause"].str.lower().str.replace(r"\W+", " ", regex=True)
    df = df.drop_duplicates("_key").drop(columns="_key").reset_index(drop=True)
    return df


def download_json() -> Path:
    cache = DATA_DIR / "_cache"
    cache.mkdir(exist_ok=True)
    target = cache / "CUADv1.json"
    if not target.exists():
        print("Downloading CUAD (~18 MB) ...")
        with urllib.request.urlopen(CUAD_ZIP_URL, timeout=120) as resp:
            blob = resp.read()
        with zipfile.ZipFile(BytesIO(blob)) as z:
            target.write_bytes(z.read("CUADv1.json"))
    return target


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--json", type=str)
    args = ap.parse_args()
    path = download_json() if args.download else Path(args.json or "")
    if not path.exists():
        ap.error("give --download or --json path/to/CUADv1.json")
    df = extract(path)
    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(df)} clauses from {df.Contract_Id.nunique()} contracts to {OUT_PATH}")
    print(df.Category.value_counts().to_string())


if __name__ == "__main__":
    main()
