"""Airisto RAG - pipeline: retrieve -> triage -> generate (one function the UI/API will call).

    python pipeline.py                      interactive chat (template wrapper, no API key needed)
    python pipeline.py --llm                use Groq for the wrapper (needs GROQ_API_KEY, LLM_MODEL)
    python pipeline.py "TestCo T3 not working" --json
"""
import argparse
import json
import sys
from typing import Callable, Optional

from embedders import get_embedder
from generator import build_response, detect_language
from retriever import Retriever
from triage import (ConversationState, booking_url, booking_url_no_match, decide, decide_no_match,
                    safety_concern_in)


def answer(query: str, retriever: Retriever, state: Optional[ConversationState] = None,
           brand: Optional[str] = None, code: Optional[str] = None, model_series: Optional[str] = None,
           llm: Optional[Callable] = None, lang: Optional[str] = None,
           whatsapp_number: Optional[str] = None, discount_code: Optional[str] = None):
    """Return (response_dict, new_state). Pass new_state back in on the next turn so
    'still showing' after a self-fix escalates by rule."""
    state = ConversationState.from_message(query, model=model_series or "", previous=state)
    res = retriever.retrieve(query, brand=brand, code=code, model_series=model_series)
    lang = lang or detect_language(query)
    cfg = {"whatsapp_number": whatsapp_number, "discount_code": discount_code}

    trusted = res.top is not None and (res.match_type == "exact" or (res.match_type == "semantic" and res.confident))
    if trusted:
        decision = decide(res.top.record, state)
        url = booking_url(res.top.record, decision, state, **cfg)
    elif res.match_type == "ambiguous":
        # we can't answer yet, but danger words still get a switch-off warning
        decision = decide_no_match(query, state) if safety_concern_in(query) else None
        url = None
    elif res.reason == "empty_query":
        decision, url = None, None
    else:
        decision = decide_no_match(query, state)
        url = booking_url_no_match(decision, brand=res.parsed.get("brand") or "",
                                   code_typed=res.parsed.get("code_typed") or "", text=query, state=state, **cfg)
    return build_response(query, res, decision, lang, llm=llm, book_url=url), state


def _print(resp: dict):
    print(f"\n  {resp['wrapper']}")
    if resp["brand"] and resp["meaning"]:
        print(f"  [{resp['match_type']}] {resp['brand']} {resp['code']}: {resp['meaning']}  -> {resp['verdict']}")
    for w in resp["warnings"]:
        print(f"  ! {w}")
    for c in resp["user_checks"]:
        print(f"  - {c}")
    if resp["needs"]:
        print(f"  (need {resp['needs']}: {', '.join(resp['options'])})")
    if resp.get("possible_matches"):
        print("  Possible matches: " + "; ".join(f"{m['brand']} {m['code']}" for m in resp["possible_matches"]))
    if resp["book_url"]:
        print(f"  Book: {resp['book_url']}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Airisto error-code assistant (CLI).")
    ap.add_argument("query", nargs="?")
    ap.add_argument("--index", default="index")
    ap.add_argument("--embedder", default="hashing", choices=["hashing", "st"])
    ap.add_argument("--model", default="")
    ap.add_argument("--llm", action="store_true", help="use Groq for the wrapper text")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    llm = None
    if args.llm:
        from generator import groq_llm
        llm = groq_llm()
    retriever = Retriever.from_dir(args.index, get_embedder(args.embedder, args.model))

    state = None
    def run(q):
        nonlocal state
        resp, state = answer(q, retriever, state, llm=llm)
        print(json.dumps(resp, ensure_ascii=False, indent=2) if args.json else "", end="")
        if not args.json:
            _print(resp)

    if args.query:
        run(args.query)
        return 0
    print("type a message, or 'q' to quit")
    while True:
        try:
            q = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in ("q", "quit", "exit"):
            break
        if q:
            run(q)
    return 0


if __name__ == "__main__":
    sys.exit(main())
