"""Text normalisation.

This function is used as the TfidfVectorizer ``preprocessor`` inside the
trained pipelines, so training and inference are guaranteed to see exactly the
same text (the old version trained on raw text but predicted on lemmatised
text).  It has no NLTK dependency, so nothing needs to be downloaded.
"""
import re

_NUM_WORDS = r"(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|fifteen|twenty|thirty|forty|fifty|sixty|ninety)"


def normalize_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"non[\s-]+refundable", "nonrefundable", text)
    text = re.sub(r"(?<=\w)['\u2019]s\b", "", text)          # possessives
    text = re.sub(r"\d+(?:\.\d+)?", " num ", text)            # 30, 5.5 -> num
    text = re.sub(rf"\b{_NUM_WORDS}\b", " num ", text)        # thirty -> num
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()
