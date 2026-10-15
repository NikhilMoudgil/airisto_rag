"""Airisto RAG - Component 4: generator.

Builds the response dict from a reviewed record. The card fields (meaning, checks,
verdict, warnings, source) are copied straight from the record / Decision, never
written by the LLM. The LLM only writes the short friendly `wrapper`, and its text is
checked before use; if it fails a check, or no LLM is configured, a fixed template is used.

    build_response(query, result, decision, lang, llm=None, book_url=None) -> dict

`llm` is any callable  llm(system_prompt: str, user_prompt: str) -> str.
`groq_llm()` makes one from GROQ_API_KEY / LLM_MODEL (pip install groq).
"""
import json
import os
import re
from typing import Callable, Optional

from retriever import RetrievalResult

# ---------- language ----------

_DEVANAGARI = re.compile(r"[\u0900-\u097F]")
_HINGLISH_WORDS = {
    "nahi", "nahin", "hai", "hain", "raha", "rahi", "rahe", "kya", "kyu", "kyun", "ka", "ki", "ke",
    "mera", "meri", "mere", "paani", "pani", "tapak", "gir", "chal", "thanda", "hawa", "bahar",
    "wala", "wali", "aa", "aur", "abhi", "bhi", "phir", "theek", "thik", "kaam", "karo", "kar",
    "ho", "hua", "hui", "se", "mein", "me", "ko", "dikha", "jalne", "smell",
}


def detect_language(text: str) -> str:
    """'hi' (Devanagari), 'hinglish' (Hindi in Roman letters) or 'en'. Rule-based."""
    if _DEVANAGARI.search(text or ""):
        return "hi"
    words = re.findall(r"[a-z]+", (text or "").lower())
    hits = sum(w in _HINGLISH_WORDS for w in words)
    return "hinglish" if hits >= 2 or (words and hits / len(words) >= 0.25) else "en"


# ---------- fixed templates (used when no LLM, or the LLM text is rejected) ----------

WRAPPERS = {
    "en": {
        "self_fix": "Good news: this one can often be sorted out with a simple check. Try the steps below.",
        "needs_service": "This looks like something a service visit should handle. You can book one below.",
        "technician_required": "This needs a technician to check it properly. Please don't try to fix it yourself.",
        "urgent": "This is serious. Please switch the AC off first, then book a technician below.",
        "escalated": "Thanks for trying that. Since the problem is still there, a technician should take a look.",
        "ask_brand": "Which brand is your AC? The same code can mean different things on different brands.",
        "ask_model": "This code differs by model series. Which series is your AC?",
        "none": "I'm not sure about this one, and I don't want to guess. A technician can check it properly.",
        "none_urgent": "Please switch the AC off at the main switch now. A technician should check this before you use it again.",
        "empty": "Tell me your AC brand and the error code on the display, and I'll look it up.",
    },
    "hinglish": {
        "self_fix": "Achhi baat hai, ye aksar ek simple check se theek ho jata hai. Neeche diye steps try kijiye.",
        "needs_service": "Ye service visit se theek hoga. Neeche se booking kar sakte hain.",
        "technician_required": "Isko technician ko dekhna padega. Please khud theek karne ki koshish mat kijiye.",
        "urgent": "Ye serious hai. Pehle AC band kijiye, phir neeche se technician book kijiye.",
        "escalated": "Try karne ke liye shukriya. Problem abhi bhi hai, isliye technician ko dekhna chahiye.",
        "ask_brand": "Aapka AC kis brand ka hai? Ek hi code alag brands mein alag matlab rakhta hai.",
        "ask_model": "Ye code model series ke hisaab se alag hai. Aapka AC kaunsi series ka hai?",
        "none": "Mujhe iske baare mein pakka nahi pata, aur main andaza nahi lagana chahta. Technician sahi se check kar sakta hai.",
        "none_urgent": "Please AC ko abhi main switch se band kijiye. Dobara chalane se pehle technician se check karwayein.",
        "empty": "Apne AC ka brand aur display par aaya error code batayein, main check karta hoon.",
    },
    "hi": {
        "self_fix": "अच्छी बात है, यह अक्सर एक आसान जाँच से ठीक हो जाता है। नीचे दिए स्टेप आज़माएँ।",
        "needs_service": "यह सर्विस विज़िट से ठीक होगा। आप नीचे से बुकिंग कर सकते हैं।",
        "technician_required": "इसे टेक्नीशियन को देखना होगा। कृपया खुद ठीक करने की कोशिश न करें।",
        "urgent": "यह गंभीर है। पहले AC बंद करें, फिर नीचे से टेक्नीशियन बुक करें।",
        "escalated": "आज़माने के लिए धन्यवाद। समस्या अभी भी है, इसलिए टेक्नीशियन को देखना चाहिए।",
        "ask_brand": "आपका AC किस ब्रांड का है? एक ही कोड अलग ब्रांड में अलग मतलब रखता है।",
        "ask_model": "यह कोड मॉडल सीरीज़ के अनुसार अलग है। आपका AC कौन सी सीरीज़ का है?",
        "none": "मुझे इसके बारे में पक्का पता नहीं है और मैं अंदाज़ा नहीं लगाना चाहता। टेक्नीशियन सही जाँच कर सकता है।",
        "none_urgent": "कृपया AC को अभी मेन स्विच से बंद करें। दोबारा चलाने से पहले टेक्नीशियन से जाँच करवाएँ।",
        "empty": "अपने AC का ब्रांड और डिस्प्ले पर आया एरर कोड बताइए, मैं देखता हूँ।",
    },
}

# What to say when nothing matched. Generic and deliberately conservative.
GENERAL_SAFE_ADVICE = [
    "Switch the AC off with the remote, wait 5 minutes, then switch it on again.",
    "Do not open the unit or touch any wiring.",
    "Note the brand, model and exact code on the display so the technician can prepare.",
]

# ---------- LLM wrapper, with guard rails ----------

SYSTEM_PROMPT = (
    "You write ONE short, friendly message (max 2 sentences, under 280 characters) above an AC "
    "diagnosis card. Reply in the language given. Rules: use only the facts supplied; never state "
    "a different verdict than the one supplied; never tell the user to open the unit, handle gas "
    "or refrigerant, touch a capacitor or wiring; do not repeat the checklist or warnings; no "
    "emojis; no markdown."
)

# If the LLM mentions any of these we throw its text away and use the template.
UNSAFE_WRAPPER = re.compile(
    r"open (up )?(the |your )?(unit|ac|panel|cover|casing|indoor|outdoor)|refrigerant|capacitor|"
    r"\bgas\b|screwdriver|multimeter|bypass|jumper|wiring|\bwires?\b|खोल",
    re.IGNORECASE,
)
MAX_WRAPPER_CHARS = 320


def wrapper_ok(text: str) -> bool:
    return bool(text) and len(text) <= MAX_WRAPPER_CHARS and not UNSAFE_WRAPPER.search(text) \
        and "```" not in text and "{" not in text


def groq_llm(api_key: Optional[str] = None, model: Optional[str] = None) -> Callable:
    """Build an llm callable backed by Groq. Model name comes from LLM_MODEL (not hard-coded)."""
    api_key = api_key or os.environ.get("GROQ_API_KEY", "")
    model = model or os.environ.get("LLM_MODEL", "")
    if not api_key or not model:
        raise RuntimeError("Set GROQ_API_KEY and LLM_MODEL (see .env.example)")
    try:
        from groq import Groq
    except ImportError as e:
        raise ImportError("Run: pip install groq") from e
    client = Groq(api_key=api_key)

    def call(system: str, user: str) -> str:
        r = client.chat.completions.create(
            model=model, temperature=0.3, max_tokens=150,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
        return (r.choices[0].message.content or "").strip()

    return call


def _wrapper(key: str, lang: str, facts: dict, llm: Optional[Callable]) -> str:
    template = WRAPPERS.get(lang, WRAPPERS["en"])[key]
    if llm is None:
        return template
    prompt = f"Language: {lang}\nFacts: {json.dumps(facts, ensure_ascii=False)}\nWrite the message."
    try:
        text = (llm(SYSTEM_PROMPT, prompt) or "").strip()
    except Exception:  # network/API trouble must never break the answer
        return template
    return text if wrapper_ok(text) else template


# ---------- response ----------

def _empty(lang: str, **extra) -> dict:
    base = {"brand": None, "code": None, "meaning": None, "user_checks": [], "verdict": None,
            "warnings": [], "source": None, "wrapper": "", "book_url": None, "match_type": "none",
            "model_series": None, "lang": lang, "needs": None, "options": [], "reason": ""}
    base.update(extra)
    return base


def build_response(query: str, result: RetrievalResult, decision, lang: Optional[str] = None,
                   llm: Optional[Callable] = None, book_url: Optional[str] = None) -> dict:
    lang = lang or detect_language(query)

    # 1. we need one more fact (brand / model series) before answering
    if result.match_type == "ambiguous":
        key = "ask_brand" if result.clarification == "brand" else "ask_model"
        return _empty(lang, wrapper=WRAPPERS.get(lang, WRAPPERS["en"])[key], needs=result.clarification,
                      options=list(result.options), reason=result.reason,
                      warnings=list(decision.warnings) if decision else [],
                      brand=result.parsed.get("brand"), code=result.parsed.get("code"))

    # 2. a trustworthy record: exact, or a semantic match with a clear winner
    if result.top and result.match_type in ("exact", "semantic") and \
            (result.match_type == "exact" or result.confident) and decision is not None:
        rec = result.top.record
        key = "escalated" if decision.escalated else decision.verdict
        facts = {"brand": rec.brand, "code": rec.code, "meaning": rec.meaning,
                 "verdict": decision.verdict, "escalated": decision.escalated}
        return {
            "brand": rec.brand,
            "code": rec.code,
            "meaning": rec.meaning,
            "user_checks": list(rec.user_checks),
            "verdict": decision.verdict,
            "warnings": list(decision.warnings),
            "source": rec.source,
            "wrapper": _wrapper(key, lang, facts, llm),
            "book_url": book_url if decision.show_booking else None,
            "match_type": result.match_type,
            "model_series": rec.model_series or None,
            "lang": lang, "needs": None, "options": [], "reason": result.reason,
        }

    # 3. nothing trustworthy: say so, give safe general advice, offer a technician
    urgent = decision is not None and decision.verdict == "urgent"
    key = "none_urgent" if urgent else ("empty" if result.reason == "empty_query" else "none")
    possible = []
    if result.match_type == "semantic":  # weak semantic hits: offer, never assert
        possible = [{"brand": m.record.brand, "code": m.record.code, "meaning": m.record.meaning}
                    for m in result.matches[:3]]
    resp = _empty(lang, wrapper=WRAPPERS.get(lang, WRAPPERS["en"])[key], reason=result.reason,
                  user_checks=[] if key == "empty" else list(GENERAL_SAFE_ADVICE),
                  warnings=list(decision.warnings) if decision else [],
                  verdict=decision.verdict if decision else None,
                  book_url=book_url if (decision and decision.show_booking) else None,
                  brand=result.parsed.get("brand"), code=result.parsed.get("code_typed") or result.parsed.get("code"))
    resp["possible_matches"] = possible
    return resp
