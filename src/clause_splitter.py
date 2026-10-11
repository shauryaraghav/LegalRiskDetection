"""Split agreement text into individual clauses."""
import re
from typing import List

# Abbreviations whose trailing dot must not end a sentence.
_ABBREV = re.compile(
    r"\b(e\.g|i\.e|etc|vs|inc|ltd|llc|co|corp|pvt|no|nos|mr|mrs|ms|dr|jr|sr|st|rs|sec|art|cl|approx|viz)\.",
    re.IGNORECASE,
)
_DOT = "\u2024"  # one-dot leader, used as a placeholder

# Leading clause numbering: "1.", "3.1", "3.1.2)", "(a)", "a)", "(iv)"
_LEADING_NUM = re.compile(r"^\s*(?:\(?\d+(?:\.\d+)*[.)]?|\(?[a-zA-Z]{1,3}[.)])\s+")
# A line that starts a new numbered clause
_NUM_LINE = re.compile(r"(?m)^\s*(?:\d+(?:\.\d+)*[.)]?|\([a-zA-Z0-9]{1,4}\)|[a-z][.)]|[•\-\*])\s+")

_NOISE = re.compile(
    r"^(?:signed|signature|witness(?:ed)?|in witness whereof|date[d]?\s*[:/]|name\s*[:/]|"
    r"place\s*[:/]|page\s+\d+(?:\s+of\s+\d+)?$|_{3,}|\.{4,}|sample\b|please (?:put|sign))",
    re.IGNORECASE,
)

MIN_WORDS = 4


def _protect(text: str) -> str:
    return _ABBREV.sub(lambda m: m.group(0).replace(".", _DOT), text)


def extract_clauses(text: str) -> List[str]:
    """Return a list of cleaned clause strings."""
    if not text or not text.strip():
        return []

    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Mark the start of numbered / bulleted lines, then un-wrap soft line breaks.
    text = _NUM_LINE.sub(lambda m: "\n\n" + m.group(0).lstrip(), text)
    text = re.sub(r"\n{2,}", "\x00", text)      # hard breaks
    text = text.replace("\n", " ")              # soft wraps
    text = re.sub(r"[ \t]+", " ", text)

    clauses: List[str] = []
    for block in text.split("\x00"):
        block = _protect(block.strip())
        if not block:
            continue
        # Split on sentence ends (followed by a capital/digit/quote) and semicolons.
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(\"\u201c])|;\s+", block)
        for part in parts:
            clause = part.replace(_DOT, ".").strip()
            clause = _LEADING_NUM.sub("", clause).strip().rstrip(".;:, ").strip()
            if len(clause.split()) < MIN_WORDS:
                continue
            if _NOISE.match(clause):
                continue
            clauses.append(clause)
    return clauses
