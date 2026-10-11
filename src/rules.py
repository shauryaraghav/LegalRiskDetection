"""Keyword / pattern rule engine.

Improvements over the original:
* whole-word regex matching ("fine" no longer matches "refine"/"define")
* negation handling ("no penalty", "without penalty")
* mitigating language ("mutual", "reasonable", "cure period") lowers the score
* every match carries a plain-English reason that is shown to the user
"""
import re
from dataclasses import dataclass, field
from typing import List

# (regex, weight, reason)
# weight 3 = on its own enough to call a clause High risk;
# weight 2 = a strong signal that the model should weigh (at least Medium).
_STRONG = [
    (r"\bwithout\s+(?:any\s+|prior\s+|written\s+|advance\s+)*notice\b", 3, "allows action without notice"),
    (r"\b(?:immediate(?:ly)?\s+terminat\w+|terminat\w+\s+immediately)\b", 3, "allows immediate termination"),
    (r"\bnon[\s-]?refundable\b", 3, "payment is non-refundable"),
    (r"\bsole\s+discretion\b", 3, "gives one party sole discretion"),
    (r"\bno\s+liability\b|\bexcludes?\s+all\s+liability\b|\bunlimited\s+liability\b", 3, "excludes or removes limits on liability"),
    (r"\bindemnif\w+|\bhold\s+harmless\b", 2, "creates an indemnity obligation"),
    (r"\birrevocabl[ey]\b|\bwaives?\b", 2, "waives or gives up rights"),
    (r"\bforfeit\w*\b", 3, "involves forfeiture"),
    (r"\bliquidated\s+damages\b|\bpunitive\b", 3, "imposes liquidated or punitive damages"),
    (r"\bcompound\w*\b", 3, "interest compounds"),
    (r"\bautomatically\s+terminat\w+", 2, "terminates automatically"),
    (r"\bbinding\s+decision\b|\bfinal\s+and\s+binding\b", 2, "decision is final and binding"),
    (r"\bin\s+perpetuity\b", 2, "obligation lasts forever"),
]
_MEDIUM = [
    (r"\bpenalt(?:y|ies)\b", 2, "includes a financial penalty"),
    (r"\bbreach(?:es|ed)?\b", 2, "defines consequences of breach"),
    (r"\bliab(?:le|ility)\b", 2, "assigns liability"),
    (r"\blate\s+(?:fee|charge|payment)s?\b|\badditional\s+charges?\b", 2, "adds late-payment charges"),
    (r"\bwithhold\w*\b|\bdeduct\w*\b", 2, "allows withholding or deductions"),
    (r"\bexclusive(?:ly)?\b", 2, "contains exclusive rights or exclusive jurisdiction"),
    (r"\bbinding\s+arbitration\b", 2, "requires binding arbitration"),
    (r"\bvoid\b", 2, "can make the agreement void"),
]
_MITIGATING = [
    (r"\bmutual(?:ly)?\b", "mutual / balanced"),
    (r"\breasonable\b", "uses a reasonableness standard"),
    (r"\bcure\s+period\b|\bremed(?:y|ied)\s+(?:the\s+)?breach\b", "offers a cure period"),
    (r"\bwritten\s+notice\b|\b\w+\s+days?\s+(?:written\s+)?notice\b", "requires advance notice"),
    (r"\bup\s+to\s+a\s+cap\b|\bshall\s+not\s+exceed\b|\blimited\s+to\b", "caps the exposure"),
    (r"\bgrace\s+period\b", "provides a grace period"),
]
_NEGATED = re.compile(
    r"\b(?:no|without|free\s+of|not\s+subject\s+to)\s+(?:any\s+)?(?:penalt(?:y|ies)|late\s+fees?|charges?)\b",
    re.IGNORECASE,
)

_STRONG_C = [(re.compile(p, re.I), w, r) for p, w, r in _STRONG]
_MEDIUM_C = [(re.compile(p, re.I), w, r) for p, w, r in _MEDIUM]
_MIT_C = [(re.compile(p, re.I), r) for p, r in _MITIGATING]


@dataclass
class RuleResult:
    score: int = 0
    strong: List[str] = field(default_factory=list)
    medium: List[str] = field(default_factory=list)
    mitigating: List[str] = field(default_factory=list)

    @property
    def reasons(self) -> List[str]:
        return self.strong + self.medium


def evaluate_rules(text: str) -> RuleResult:
    res = RuleResult()
    negated = bool(_NEGATED.search(text))

    for rx, w, reason in _STRONG_C:
        if rx.search(text):
            res.strong.append(reason)
            res.score += w
    for rx, w, reason in _MEDIUM_C:
        if rx.search(text):
            if negated and ("penalt" in reason or "late" in reason):
                continue
            res.medium.append(reason)
            res.score += w
    for rx, reason in _MIT_C:
        if rx.search(text):
            res.mitigating.append(reason)

    # Mitigating language softens, but never erases, a strong flag.
    if res.mitigating and res.score > 0:
        res.score = max(res.score - 2 * min(len(res.mitigating), 2), 0 if not res.strong else 1)
    return res


def get_rule_score(text: str) -> int:
    """Backward-compatible helper."""
    return evaluate_rules(text).score
