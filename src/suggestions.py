"""Plain-English negotiation suggestions.

Suggestions are driven by (a) the rule that flagged the clause and (b) the clause
category as a fallback.  They are general drafting ideas - not legal advice.
"""
import re
from dataclasses import asdict, dataclass
from typing import Dict, List

from src.rules import RuleResult


@dataclass
class Suggestion:
    issue: str          # what is risky
    action: str         # what to ask for
    safer_wording: str  # example replacement language

    def as_dict(self) -> dict:
        return asdict(self)


# keyed by the reason strings produced in src/rules.py
_BY_REASON: Dict[str, Suggestion] = {
    "allows action without notice": Suggestion(
        "No notice is required before this right can be used.",
        "Ask for a minimum written notice period (for example 15-30 days).",
        "Either party may do so by giving at least thirty (30) days' prior written notice."),
    "allows immediate termination": Suggestion(
        "Termination can take effect immediately.",
        "Limit immediate termination to serious cases and add a cure period for everything else.",
        "A party may terminate immediately only for a material breach that remains uncured fifteen (15) days after written notice."),
    "payment is non-refundable": Suggestion(
        "Money paid cannot be recovered.",
        "Negotiate a refund of amounts for services not delivered, or tie the non-refundable part to work actually done.",
        "Fees are refundable pro rata for any services not performed."),
    "gives one party sole discretion": Suggestion(
        "One party can decide alone, with no standard to challenge.",
        "Replace 'sole discretion' with 'reasonable discretion' and require written reasons and notice.",
        "The Company may do so acting reasonably and after giving written notice with reasons."),
    "excludes or removes limits on liability": Suggestion(
        "Liability is either excluded entirely or unlimited.",
        "Ask for a mutual liability cap (for example fees paid in the last 12 months).",
        "Each party's total liability shall not exceed the fees paid in the twelve (12) months preceding the claim."),
    "creates an indemnity obligation": Suggestion(
        "An indemnity can expose you to open-ended third-party claims.",
        "Make it mutual, limit it to claims caused by the indemnifying party's own breach or negligence, add a cap, and require prompt notice.",
        "Each party shall indemnify the other only against third-party claims caused by its own breach or negligence, subject to the liability cap."),
    "waives or gives up rights": Suggestion(
        "You give up legal rights.",
        "Avoid blanket waivers; limit them to specific, named rights and keep statutory rights intact.",
        "This waiver applies only to [specific right] and does not affect any right that cannot be waived by law."),
    "involves forfeiture": Suggestion(
        "Money or property can be lost entirely.",
        "Limit forfeiture to actual, documented loss and require an itemised statement.",
        "Amounts may be retained only to the extent of documented loss, supported by an itemised written statement."),
    "imposes liquidated or punitive damages": Suggestion(
        "Fixed damages may be out of proportion to real loss.",
        "Ask for a cap and for the amount to be a genuine pre-estimate of loss.",
        "Liquidated damages shall be a genuine pre-estimate of loss and shall not exceed [5-10]% of the contract value."),
    "interest compounds": Suggestion(
        "Compounding interest makes overdue amounts grow quickly.",
        "Prefer simple interest at a stated reasonable rate, after a grace period.",
        "Simple interest at [x]% per annum applies to amounts overdue by more than ten (10) days."),
    "terminates automatically": Suggestion(
        "The agreement ends automatically, with no chance to fix the problem.",
        "Replace automatic termination with a written-notice-and-cure process.",
        "Either party may terminate if the triggering event is not remedied within thirty (30) days of written notice."),
    "decision is final and binding": Suggestion(
        "The decision cannot be challenged.",
        "Ask for a neutral decision-maker and a limited right of appeal or review.",
        "The decision shall be made by a mutually appointed neutral and is subject to review under applicable law."),
    "obligation lasts forever": Suggestion(
        "The obligation has no end date.",
        "Limit the duration (for example 2-5 years after termination), with a longer period only for genuine trade secrets.",
        "These obligations continue for three (3) years after termination of this Agreement."),
    "includes a financial penalty": Suggestion(
        "A penalty applies.",
        "Ask for a stated cap, a grace period and a written reminder before it applies.",
        "No penalty applies until ten (10) days after written reminder, and penalties shall not exceed [x]% in total."),
    "defines consequences of breach": Suggestion(
        "Consequences of breach are triggered immediately.",
        "Add written notice and a cure period before remedies apply.",
        "Remedies apply only if the breach is not cured within fifteen (15) days of written notice."),
    "assigns liability": Suggestion(
        "Liability is placed on one party.",
        "Check who bears the risk; ask for it to be mutual and capped.",
        "Each party is liable for losses caused by its own negligence, up to the agreed cap."),
    "adds late-payment charges": Suggestion(
        "Late charges apply.",
        "Negotiate a grace period (5-10 days) and a reasonable rate.",
        "A late fee of [x]% applies only to amounts unpaid ten (10) days after the due date."),
    "allows withholding or deductions": Suggestion(
        "Amounts can be withheld or deducted.",
        "Require deductions to be itemised, documented and notified in writing.",
        "Deductions are permitted only for documented loss and must be itemised in a written statement."),
    "contains exclusive rights or exclusive jurisdiction": Suggestion(
        "Rights or the choice of court are exclusive to one side or place.",
        "Prefer a neutral or non-exclusive venue and require mediation first.",
        "The parties shall first attempt mediation; courts at [neutral venue] shall have non-exclusive jurisdiction."),
    "requires binding arbitration": Suggestion(
        "Disputes go to binding arbitration.",
        "Check the seat, rules, cost sharing and how the arbitrator is chosen.",
        "Arbitration shall be before a sole arbitrator appointed by mutual consent, with costs shared equally."),
    "can make the agreement void": Suggestion(
        "The agreement can be declared void.",
        "Clarify exactly what triggers this and add notice and cure.",
        "The Agreement may be treated as void only for [specified event] after written notice and a cure period."),
}

# fallback by category when no specific rule fired: (medium text, high text)
_BY_CATEGORY: Dict[str, Suggestion] = {
    "Payment": Suggestion("Payment terms may be unfavourable.", "Make amounts, due dates, refunds and tax treatment explicit.",
                          "Invoices are payable within thirty (30) days of receipt; disputed amounts may be withheld pending resolution."),
    "Termination": Suggestion("Termination rights may be uneven.", "Make termination rights mutual, with notice and a cure period.",
                              "Either party may terminate on [30] days' written notice; for breach, after a [15]-day cure period."),
    "Penalty": Suggestion("A penalty or charge applies.", "Ask for a cap, a grace period and a prior reminder.",
                          "Charges apply only after a [10]-day grace period and are capped at [x]%."),
    "Security_Deposit": Suggestion("Deposit handling is unclear.", "Ask for a refund deadline, itemised deductions and a written receipt.",
                                   "The deposit shall be refunded within [30] days of vacating, less itemised documented deductions."),
    "Restriction": Suggestion("A restriction limits your freedom.", "Narrow its scope, duration and geography, and allow consent not to be unreasonably withheld.",
                              "This restriction applies for [6] months within [defined area]; consent shall not be unreasonably withheld."),
    "Liability": Suggestion("Liability allocation may be one-sided.", "Ask for a mutual cap and exclude indirect losses for both sides.",
                            "Each party's liability is capped at the fees paid in the previous twelve months."),
    "Administrative": Suggestion("One party may be able to change terms.", "Require written, mutually signed amendments and notice of changes.",
                                 "This Agreement may be amended only in writing signed by both parties."),
    "Confidentiality": Suggestion("Confidentiality scope or duration is broad.", "Limit the duration, add standard exclusions (public information, legally required disclosure) and make it mutual.",
                                  "Obligations last [3] years and do not apply to public information or disclosures required by law."),
    "Intellectual_Property": Suggestion("Ownership or licence terms may favour the other side.",
                                        "Clarify who owns what, keep your pre-existing IP, limit the licence scope and make any revocation subject to notice and refund.",
                                        "Each party keeps its pre-existing intellectual property; the licence is limited to [stated purpose] and may be revoked only for uncured material breach."),
    "Jurisdiction": Suggestion("Dispute forum may favour one side.", "Prefer a neutral venue and require mediation first.",
                               "Disputes shall first go to mediation, then to the courts of [neutral venue]."),
    "Indemnification": Suggestion("Indemnity scope is broad.", "Make it mutual, capped and limited to third-party claims.",
                                  "Each party indemnifies the other only for third-party claims caused by its own breach, subject to the cap."),
}

# suppress a suggestion when the clause already contains the matching protection
_SUPPRESS = {
    "assigns liability": "caps the exposure",
    "defines consequences of breach": "offers a cure period",
    "adds late-payment charges": "provides a grace period",
}

MAX_PER_CLAUSE = 3


def suggest_for_clause(category: str, risk: str, rules: RuleResult) -> List[Suggestion]:
    """Return up to 3 suggestions for a Medium/High clause (none for Low)."""
    if risk == "Low":
        return []
    out: List[Suggestion] = []
    # strongest evidence first
    for reason in rules.strong + rules.medium:
        sup = _SUPPRESS.get(reason)
        if sup and sup in rules.mitigating:
            continue
        s = _BY_REASON.get(reason)
        if s and s not in out:
            out.append(s)
        if len(out) == MAX_PER_CLAUSE:
            break
    if not out:
        fb = _BY_CATEGORY.get(category)
        if fb:
            out.append(fb)
    return out


# --------------------------------------------------- agreement-level checks
_CHECKS = [
    ("Notice requirements", r"\bnotice\b",
     "No notice requirement was found.",
     "Add a written-notice requirement for termination, changes and default."),
    ("Liability cap", r"\bnot\s+exceed\b|\blimited\s+to\b|\bcap(?:ped)?\b|\bmaximum\s+liability\b",
     "No cap on liability was found.",
     "Consider a mutual cap on liability (for example fees paid in the last 12 months)."),
    ("Dispute resolution", r"\bgoverned\s+by\b|\bjurisdiction\b|\barbitrat\w+|\bcourts?\b|\bmediation\b",
     "No governing law or dispute-resolution clause was found.",
     "Add governing law and a dispute process (negotiation, mediation, then courts or arbitration)."),
    ("Confidentiality", r"\bconfidential\w*\b",
     "No confidentiality clause was found.",
     "If sensitive information is shared, add a mutual confidentiality clause."),
    ("Termination", r"\bterminat\w+\b",
     "No termination clause was found.",
     "Add clear termination rights, notice periods and consequences."),
    ("Cure / grace period", r"\bcure\b|\bgrace\s+period\b|\bremed(?:y|ied)\b",
     "No cure or grace period was found.",
     "Add a cure period before default remedies, termination or penalties apply."),
]
MIN_CLAUSES_FOR_MISSING_CHECK = 5   # tiny snippets would trigger false alarms


def missing_protections(text: str, total_clauses: int) -> List[dict]:
    """Common protections that do not appear anywhere in the document."""
    if total_clauses < MIN_CLAUSES_FOR_MISSING_CHECK:
        return []
    found = []
    for name, pattern, issue, action in _CHECKS:
        if not re.search(pattern, text, re.IGNORECASE):
            found.append({"topic": name, "issue": issue, "action": action})
    return found
