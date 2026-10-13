"""Airisto RAG - Component 3: retriever.

Given a customer's message (and optionally brand / code / model chosen in the UI),
decide what to return:

  exact      brand + code (+ model) pin down exactly one record. Score 1.0, no AI involved.
  ambiguous  we need one more fact first: the brand, or the model series.
  semantic   no code given: closest records by meaning, only above a similarity threshold.
  none       nothing trustworthy found. The caller must say 'not sure' and offer a technician.

Safety rules baked in:
  * A code the database does not have for a known brand is NEVER guessed from similar text.
  * A code with no brand is never answered; we ask for the brand first (E4 means
    different things on different brands).
  * Semantic matches below the threshold are discarded, not shown.

Usage:
    python retriever.py "TestCo T3"                 (one query)
    python retriever.py                             (interactive)
    python retriever.py "paani tapak raha hai" --brand FakeAir --debug
    python retriever.py "..." --embedder st --index index_st
"""
import argparse
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Optional

from embedders import cosine, get_embedder
from indexer import load_index, record_from_dict
from schema import ErrorCodeRecord, normalize_code

# Provisional. The right values depend on real data: calibrate with `--debug`
# and the evaluation set (Component 6). Override with SIMILARITY_THRESHOLD.
DEFAULT_THRESHOLDS = {"hashing": 0.30, "st": 0.35}
CODE_CONTEXT_WORDS = {"error", "code", "fault", "err", "errorcode", "showing", "shows",
                      "display", "displays", "displaying", "blinking", "flashing"}
CODE_SHAPE = re.compile(r"^[A-Za-z]{1,3}[-_.]?\d{1,3}[A-Za-z]?$")  # E4, CH07, F-12, E1A
REFRIGERANT = re.compile(r"^[Rr]\d{2,3}[A-Za-z]?$")                 # R32, R410A are gases, not codes


def _key(text: str) -> str:
    """Lowercase letters+digits only: 'Blue Star' -> 'bluestar', 'CH-07' -> 'ch07'."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


@dataclass
class Match:
    record: ErrorCodeRecord
    score: float
    match_type: str  # "exact" | "semantic"


@dataclass
class RetrievalResult:
    match_type: str  # "exact" | "semantic" | "ambiguous" | "none"
    matches: list = field(default_factory=list)
    reason: str = ""
    clarification: Optional[str] = None  # "brand" | "model_series" | None
    options: list = field(default_factory=list)  # choices to offer for the clarification
    parsed: dict = field(default_factory=dict)  # brand/code/series we understood
    best_score: float = 0.0
    confident: bool = False  # semantic only: clear winner, safe to show as THE answer

    @property
    def top(self) -> Optional[Match]:
        return self.matches[0] if self.matches else None


class Retriever:
    def __init__(self, index: dict, embedder, threshold: Optional[float] = None,
                 top_k: int = 5, margin: float = 0.05):
        if not index["entries"]:
            raise ValueError("index is empty")
        self.embedder = embedder
        self.top_k = top_k
        self.margin = margin
        self.threshold = self._resolve_threshold(threshold, embedder.name)
        self.entries = [
            {"record": record_from_dict(e["record"]), "vector": e["vector"]}
            for e in index["entries"]
        ]
        self._build_vocab()

    @classmethod
    def from_dir(cls, index_dir: str, embedder, **kwargs):
        return cls(load_index(index_dir, embedder), embedder, **kwargs)

    @staticmethod
    def _resolve_threshold(explicit, embedder_name: str) -> float:
        if explicit is not None:
            return explicit
        env = os.environ.get("SIMILARITY_THRESHOLD")
        if env:
            return float(env)
        family = "st" if embedder_name.startswith("st:") else "hashing"
        return DEFAULT_THRESHOLDS[family]

    # ---------- vocabulary learned from the index ----------

    def _build_vocab(self):
        self.brands = {}   # brand key -> display name
        self.codes = set()  # normalised codes
        self.series = {}   # brand key -> {series key: display}
        for e in self.entries:
            r = e["record"]
            bk = _key(r.brand)
            self.brands[bk] = r.brand
            self.codes.add(r.code_norm)
            if r.model_series:
                self.series.setdefault(bk, {})[_key(r.model_series)] = r.model_series

    # ---------- understanding the query ----------

    @staticmethod
    def _tokens(query: str) -> list:
        return re.findall(r"[A-Za-z0-9][A-Za-z0-9.\-_]*", query)

    @staticmethod
    def _spans(tokens: list):
        """Yield (key, original_text, is_pure_digit, prev_token_lower) for tokens and
        adjacent token pairs, so 'CH 07' is seen as 'ch07' and 'Blue Star' as 'bluestar'."""
        for i, tok in enumerate(tokens):
            prev = tokens[i - 1].lower() if i else ""
            yield _key(tok), tok, tok.isdigit(), prev
            if i + 1 < len(tokens):
                yield _key(tok + tokens[i + 1]), tok + " " + tokens[i + 1], False, prev

    def _code_attempt(self, tokens: list) -> Optional[str]:
        """A code-shaped token that is NOT in the database ('T9', 'CH 99'). If the customer
        clearly typed one, we must say 'unknown code', not guess from similar text.
        Refrigerant names (R32), brands and known model series never count."""
        all_series = {k for d in self.series.values() for k in d}
        for i, tok in enumerate(tokens):
            options = [(tok, i + 1)]
            if tok.isalpha() and len(tok) <= 3 and i + 1 < len(tokens) \
                    and tokens[i + 1].isdigit() and len(tokens[i + 1]) <= 3:
                options.append((tok + " " + tokens[i + 1], i + 2))  # 'CH 07'
            for text, next_i in options:
                joined = text.replace(" ", "")
                if not CODE_SHAPE.match(joined) or REFRIGERANT.match(joined):
                    continue
                if _key(joined) in self.brands or _key(joined) in all_series:
                    continue
                prev_word = tokens[i - 1].lower() if i else ""
                next_word = tokens[next_i].lower() if next_i < len(tokens) else ""
                near_context = prev_word in CODE_CONTEXT_WORDS or next_word in CODE_CONTEXT_WORDS
                typed_like_code = any(c.isupper() for c in joined)
                if near_context or typed_like_code or len(tokens) <= 3:
                    return joined.upper()
        return None

    def parse_query(self, query: str) -> dict:
        """Rule-based extraction of brand, code and model series. No AI, fully predictable."""
        tokens = self._tokens(query)
        spans = list(self._spans(tokens))
        brand = None
        for k, _, _, _ in sorted(spans, key=lambda s: -len(s[0])):
            if k in self.brands:
                brand = self.brands[k]
                break
        series = None
        if brand:
            for k, _, _, _ in spans:
                if k in self.series.get(_key(brand), {}):
                    series = self.series[_key(brand)][k]
                    break
        code = None
        for k, text, is_digit, prev in sorted(spans, key=lambda s: -len(s[0])):
            norm = normalize_code(k)
            if norm not in self.codes or k == _key(brand or "") or k in self.brands:
                continue
            if is_digit and not (prev in CODE_CONTEXT_WORDS or len(tokens) <= 3):
                continue  # a bare number like '1.5 ton' is not an error code
            if text.isalpha() and not text.isupper() and len(text) <= 2:
                continue  # a lowercase 'e' or 'ch' with no digits is just a word
            if series and k == _key(series):
                continue
            code = norm
            break
        code_typed = None if code else self._code_attempt(tokens)
        return {"brand": brand, "code": code, "model_series": series, "code_typed": code_typed}

    # ---------- retrieval ----------

    def retrieve(self, query: str, brand: Optional[str] = None, code: Optional[str] = None,
                 model_series: Optional[str] = None) -> RetrievalResult:
        query = (query or "").strip()
        parsed = self.parse_query(query) if query else \
            {"brand": None, "code": None, "model_series": None, "code_typed": None}
        # explicit choices from the UI (brand chips, model dropdown) win over parsing
        if brand:
            bk = _key(brand)
            if bk not in self.brands:
                return RetrievalResult("none", reason="unknown_brand", parsed={"brand": brand})
            parsed["brand"] = self.brands[bk]
        if code:
            parsed["code"] = normalize_code(code)
        if model_series:
            parsed["model_series"] = model_series.strip()
        if not query and not parsed["code"]:
            return RetrievalResult("none", reason="empty_query", parsed=parsed)

        if parsed["code"]:
            return self._by_code(parsed)
        if parsed["code_typed"]:  # they typed a code we do not have: never guess
            return RetrievalResult("none", reason="unknown_code", parsed=parsed)
        return self._semantic(query, parsed)

    def _by_code(self, parsed: dict) -> RetrievalResult:
        code, brand, series = parsed["code"], parsed["brand"], parsed["model_series"]
        cands = [e["record"] for e in self.entries if e["record"].code_norm == code]
        if not cands:
            return RetrievalResult("none", reason="unknown_code", parsed=parsed)

        if not brand:
            brands = sorted({r.brand for r in cands})
            return RetrievalResult("ambiguous", reason="need_brand", clarification="brand",
                                   options=brands, parsed=parsed)

        cands = [r for r in cands if _key(r.brand) == _key(brand)]
        if not cands:
            return RetrievalResult("none", reason="unknown_code", parsed=parsed)

        if series:
            same = [r for r in cands if _key(r.model_series) == _key(series)]
            generic = [r for r in cands if not r.model_series]
            cands = same or generic or cands
        else:
            cands = [r for r in cands if not r.model_series] or cands

        if len(cands) == 1:
            return RetrievalResult("exact", matches=[Match(cands[0], 1.0, "exact")],
                                   reason="exact", parsed=parsed, best_score=1.0, confident=True)
        options = sorted({r.model_series for r in cands if r.model_series})
        return RetrievalResult("ambiguous", matches=[Match(r, 1.0, "exact") for r in cands],
                               reason="need_model_series", clarification="model_series",
                               options=options, parsed=parsed)

    def _semantic(self, query: str, parsed: dict) -> RetrievalResult:
        brand = parsed["brand"]
        pool = [e for e in self.entries if not brand or _key(e["record"].brand) == _key(brand)]
        qv = self.embedder.embed_query(query)
        scored = sorted(((cosine(qv, e["vector"]), e["record"]) for e in pool),
                        key=lambda x: -x[0])[: self.top_k]
        best = scored[0][0] if scored else 0.0
        kept = [(s, r) for s, r in scored if s >= self.threshold]
        if not kept:
            return RetrievalResult("none", reason="no_semantic_match", parsed=parsed, best_score=best)
        matches = [Match(r, s, "semantic") for s, r in kept]
        gap_ok = len(kept) == 1 or kept[0][0] - kept[1][0] >= self.margin
        result = RetrievalResult("semantic", matches=matches, reason="semantic",
                                 parsed=parsed, best_score=best, confident=gap_ok)
        if not brand and len({m.record.brand for m in matches}) > 1:
            result.clarification, result.options = "brand", sorted({m.record.brand for m in matches})
        return result


# ---------- command line ----------

def _print_result(res: RetrievalResult, debug: bool):
    print(f"  -> {res.match_type.upper()} ({res.reason})"
          + (f"  confident={res.confident}" if res.match_type == "semantic" else ""))
    for m in res.matches:
        r = m.record
        print(f"     [{m.score:.2f}] {r.brand} {r.model_series or ''} {r.code}: {r.meaning}")
        print(f"            verdict={r.verdict}  track={r.track}")
    if res.clarification:
        print(f"     need {res.clarification}: pick one of {res.options}")
    if debug:
        print(f"     parsed={res.parsed}  best_score={res.best_score:.3f}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Try the Airisto retriever.")
    ap.add_argument("query", nargs="?", help="omit for interactive mode")
    ap.add_argument("--index", default="index")
    ap.add_argument("--embedder", default="hashing", choices=["hashing", "st"])
    ap.add_argument("--model", default="")
    ap.add_argument("--brand", default=None)
    ap.add_argument("--code", default=None)
    ap.add_argument("--model-series", default=None)
    ap.add_argument("--debug", action="store_true", help="show parsed fields and best score")
    args = ap.parse_args(argv)

    emb = get_embedder(args.embedder, args.model)
    ret = Retriever.from_dir(args.index, emb)
    print(f"index ready: {len(ret.entries)} records, embedder={emb.name}, threshold={ret.threshold}")

    def run(q):
        res = ret.retrieve(q, args.brand, args.code, args.model_series)
        print(f"\n> {q}")
        _print_result(res, args.debug)

    if args.query:
        run(args.query)
        return 0
    print("type a query, or 'q' to quit")
    while True:
        try:
            q = input("\nquery> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in ("q", "quit", "exit"):
            break
        if q:
            run(q)
    return 0


if __name__ == "__main__":
    sys.exit(main())
