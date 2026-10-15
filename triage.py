"""Airisto RAG - Component 5: triage + booking handoff (standard library only).

Everything here is rule-based. The LLM never decides a verdict.

  decide(record, state)        -> Decision   verdict/track/booking flags for a matched record
  decide_no_match(text, state) -> Decision   for unknown codes: no verdict, offer a technician
  booking_url(...)             -> wa.me link with brand, model, code, issue pre-filled, or None

Rules (from ARCHITECTURE.md section 7):
  * self_fix + user says the fix did not work  ->  technician_required / repair   (escalation)
  * self_fix                                   ->  no booking button yet
  * needs_service / technician_required / urgent -> booking button
  * urgent                                     ->  switch-off warning comes first
"""
import os
import re
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import quote

from schema import ErrorCodeRecord

SWITCH_OFF_WARNING = "Switch the AC off at the main switch now and do not run it until a technician checks it."

# Phrases meaning "I tried your fix and the problem is still there" (English + Hinglish).
FIX_FAILED_PATTERNS = [
    r"\bstill (showing|shows|blinking|flashing|not|same|there|happening|the same)\b",
    r"\b(did ?n'?o?t|didnt|does ?n'?o?t|doesnt) work\b",
    r"\bnot (fixed|working still|solved|resolved)\b",
    r"\b(same|that) (error|problem|issue|code)\b",
    r"\b(error|problem|issue|code) (came|is back|again)\b",
    r"\bcame back\b",
    r"\b(already )?tried (that|it|this|everything)\b",
    r"\b(abhi|ab) bhi\b",
    r"\bphir se\b",
    r"\b(theek|thik|sahi|ṭhīk) nahi (hua|hui|hai)\b",
    r"\bkaam nahi (kiya|kar raha)\b",
    r"\bsame problem\b",
    r"\bfix nahi\b",
]
_FIX_FAILED = re.compile("|".join(FIX_FAILED_PATTERNS), re.IGNORECASE)

# Danger words in a free-text message. Used when no record matched, so we can still
# tell the user to switch off. English + Hinglish.
SAFETY_PATTERNS = [
    r"\bsmoke\b", r"\bsmoking\b", r"\bburn(t|ing)?\b", r"\bspark(s|ing)?\b", r"\bfire\b",
    r"\bshock\b", r"\bcurrent (lag|maar|aa)\w*", r"\bjal(ne|ta|a)\b", r"\bdhuan\b", r"\bdhuaan\b",
    r"\bchingari\b", r"\bgas leak\b",
]
_SAFETY = re.compile("|".join(SAFETY_PATTERNS), re.IGNORECASE)


def fix_failed_in(text: str) -> bool:
    return bool(text and _FIX_FAILED.search(text))


def safety_concern_in(text: str) -> bool:
    return bool(text and _SAFETY.search(text))


@dataclass
class ConversationState:
    model: str = ""              # model name/number if the user gave it
    fix_failed: bool = False     # user said the self-fix did not work

    @classmethod
    def from_message(cls, text: str, model: str = "", previous: Optional["ConversationState"] = None):
        """Update state from a new user message. fix_failed is sticky once set."""
        sticky = previous.fix_failed if previous else False
        model = model or (previous.model if previous else "")
        return cls(model=model, fix_failed=sticky or fix_failed_in(text))


@dataclass
class Decision:
    verdict: Optional[str]       # None only for a no-match
    track: str                   # none | service | repair
    show_booking: bool
    escalated: bool = False      # self_fix turned into technician_required by rule
    switch_off_first: bool = False
    warnings: list = field(default_factory=list)
    reason: str = ""


def decide(record: ErrorCodeRecord, state: Optional[ConversationState] = None) -> Decision:
    state = state or ConversationState()
    verdict, track, escalated = record.verdict, record.track, False

    if verdict == "self_fix" and state.fix_failed:
        verdict, track, escalated = "technician_required", "repair", True

    warnings = list(record.warnings)
    urgent = verdict == "urgent"
    if urgent:  # the switch-off warning always comes first
        warnings.insert(0, SWITCH_OFF_WARNING)

    reason = "escalated: self-fix did not work" if escalated else f"from record ({record.verdict})"
    return Decision(verdict=verdict, track=track, show_booking=verdict != "self_fix",
                    escalated=escalated, switch_off_first=urgent, warnings=warnings, reason=reason)


def decide_no_match(text: str = "", state: Optional[ConversationState] = None) -> Decision:
    """No reviewed record matched. We do not invent a verdict; we offer a technician.
    If the message mentions smoke/burning/sparks we treat it as urgent anyway."""
    if safety_concern_in(text):
        return Decision(verdict="urgent", track="repair", show_booking=True, switch_off_first=True,
                        warnings=[SWITCH_OFF_WARNING], reason="safety words in message")
    return Decision(verdict=None, track="repair", show_booking=True, reason="no matching record")


# ---------- booking ----------

def _wa_link(number: str, message: str) -> Optional[str]:
    digits = re.sub(r"\D", "", number or "")
    if not digits:
        return None
    return f"https://wa.me/{digits}?text={quote(message, safe='')}"


def _config(whatsapp_number, discount_code):
    return (whatsapp_number if whatsapp_number is not None else os.environ.get("WHATSAPP_NUMBER", ""),
            discount_code if discount_code is not None else os.environ.get("DISCOUNT_CODE", ""))


def booking_url(record: ErrorCodeRecord, decision: Decision, state: Optional[ConversationState] = None,
                whatsapp_number: Optional[str] = None, discount_code: Optional[str] = None) -> Optional[str]:
    """WhatsApp link with the case pre-filled, or None if no booking is needed/possible."""
    if not decision.show_booking:
        return None
    number, code = _config(whatsapp_number, discount_code)
    model = (state.model if state else "") or record.model_series
    kind = {"service": "Service", "repair": "Repair"}.get(decision.track, "Repair")
    if decision.verdict == "urgent":
        kind += " (URGENT)"
    lines = ["Hi Airisto, I need an AC technician.",
             f"Brand: {record.brand}"]
    if model:
        lines.append(f"Model: {model}")
    lines += [f"Error code: {record.code}",
              f"Issue: {record.meaning}",
              f"Type: {kind}"]
    if decision.escalated:
        lines.append("Note: I tried the suggested checks and the problem is still there.")
    if code:
        lines.append(f"Discount code: {code}")
    return _wa_link(number, "\n".join(lines))


def booking_url_no_match(decision: Decision, brand: str = "", code_typed: str = "", text: str = "",
                         state: Optional[ConversationState] = None,
                         whatsapp_number: Optional[str] = None, discount_code: Optional[str] = None) -> Optional[str]:
    """Booking link when nothing matched: carries what the customer told us."""
    if not decision.show_booking:
        return None
    number, code = _config(whatsapp_number, discount_code)
    lines = ["Hi Airisto, I need an AC technician."]
    if brand:
        lines.append(f"Brand: {brand}")
    if state and state.model:
        lines.append(f"Model: {state.model}")
    if code_typed:
        lines.append(f"Error code: {code_typed}")
    if text:
        lines.append(f"Problem: {text.strip()[:200]}")
    if decision.verdict == "urgent":
        lines.append("Type: Repair (URGENT)")
    if code:
        lines.append(f"Discount code: {code}")
    return _wa_link(number, "\n".join(lines))
