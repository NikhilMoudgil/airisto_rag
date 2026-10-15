# Airisto Error-Code RAG: Project Context (as of 15 Oct 2026)

## What this is
Airisto is a hyper-local AC/appliance care business in the Tricity (Chandigarh, Mohali, Panchkula, Kharar, Zirakpur). Customers use WhatsApp. This project is a RAG assistant that tells a customer or technician what an AC error code means, what is safe to try, and whether a technician is needed. If one is needed, it hands off to WhatsApp booking with the details pre-filled (flat Rs 50 off for bookings via the link).

**Core rule:** the diagnosis comes from a reviewed record, never from the LLM's memory. The LLM only writes a short friendly wrapper line. Users write English and Hinglish.

## Stack and conventions
- Python, **standard library only** for core code. Optional: `sentence-transformers` (real embeddings), `groq` (wrapper text).
- Search is brute-force cosine over one JSON file (`index/index.json`), not ChromaDB. Fine for up to a few thousand records.
- Tests use `unittest`. Run all: `python -m unittest discover -p "test_*.py"` (56 passing).
- Planned later: FastAPI backend, Supabase.

## Request flow
message -> **retrieve** (exact brand+code, else semantic, else none) -> **triage** (rule-based verdict) -> **generate** (card from record + guarded LLM wrapper) -> **booking link** if a technician is needed.

## What is DONE
| Component | File | Notes |
|---|---|---|
| 1 Schema + CSV validator | `schema.py` | Verdicts: `self_fix`, `needs_service`, `technician_required`, `urgent`. Tracks: none/service/repair. Only records with `reviewed_by` + `last_reviewed` are published. `normalize_code("E 4")` -> `E4`. |
| Embedders | `embedders.py` | `HashingEmbedder` (no install, typo-tolerant, no real meaning) and `SentenceTransformerEmbedder` (default `paraphrase-multilingual-MiniLM-L12-v2`). Index stores the embedder name; mismatches are refused. |
| 2 Indexer | `indexer.py` | CSV -> `index/index.json`. Refuses to build on validation errors. Skips unreviewed unless `--include-unreviewed`. Embeds brand, model, code variants (CH 07/CH07/CH-07), meaning, symptoms, causes. |
| 3 Retriever | `retriever.py` | Match types: `exact`, `ambiguous` (needs brand or model series), `semantic`, `none`. Never guesses an unknown code for a known brand. Never answers a code without a brand. Rejects refrigerant names (R32) and bare numbers as codes. Semantic results below threshold are dropped; `confident` means a clear winner (margin 0.05). |
| 4 Generator | `generator.py` | Card fields copied from the record. LLM text is rejected (fixed template used) if too long or if it mentions opening the unit, refrigerant, capacitor, wiring, etc. Templates in English, Hinglish, Hindi. No-match gives generic safe advice. |
| 5 Triage + booking | `triage.py` | Rule-based. A self-fix that the user says failed ("still showing", "abhi bhi") escalates to `technician_required`, and stays escalated. Urgent always shows the switch-off warning first. Smoke/burning/sparks words in a no-match message are treated as urgent. Builds `wa.me` links. |
| Pipeline | `pipeline.py` | `answer(query, retriever, state, ...)` returns `(response_dict, new_state)`. CLI: `python pipeline.py [--llm] [--embedder st --index index_st]`. |

Response shape: `brand, code, meaning, user_checks, verdict, warnings, source, wrapper, book_url, match_type`, plus `model_series, lang, needs, options, reason, possible_matches`.

Config (`.env.example`): `GROQ_API_KEY`, `LLM_MODEL`, `EMBEDDING_MODEL`, `WHATSAPP_NUMBER` (digits with country code), `DISCOUNT_CODE`, `SIMILARITY_THRESHOLD`.

## Data
- `data/test_records_FAKE.csv`: 8 fake records (TestCo T1-T5 incl. two T5 model series, FakeAir F-12, one unreviewed F-99). Used only for tests.
- `data/error_codes_template.csv`: format template. List columns are separated by ` | `.
- **No real, technician-reviewed codes exist yet.**

## Known state and caveats
- A sentence-transformers index (`index_st/index.json`, 8 records, 384 dim) was built, but **with `--include-unreviewed`**, so it contains the unreviewed F-99. Rebuild without the flag for real use.
- In that index, unrelated records have 0.45-0.81 cosine similarity. The provisional thresholds (`hashing` 0.30, `st` 0.35) are almost certainly too low for the real model. **They must be tuned on an evaluation set, not guessed.**
- The real embedder and the Groq call have not been run in the dev sandbox (no network). Only the hashing embedder and stub LLMs were tested.
- `test_indexer.py` was previously a duplicate of `test_retriever.py`; it has been replaced with real indexer tests.

## NEXT steps
1. **Component 6: evaluation** (`eval/test_queries.csv`, `eval/run_eval.py`). About 100 queries covering misspellings, Hinglish, symptom-only queries and deliberately unknown codes. Track: right record (exact vs semantic), right verdict, honest "not sure", and zero unsafe advice. Use it to set `SIMILARITY_THRESHOLD` and the margin.
2. Streamlit UI (`app_streamlit.py`), then API (`api.py`, FastAPI), then SEO pages (`seo/generate_pages.py`).

## Open items for the owner
- Choose the 5-6 launch brands and collect the first batch of real codes, reviewed by a technician.
- Final WhatsApp Business number and discount code wording.

## Rules any new code must keep
- Never tell users to open a sealed unit, handle refrigerant, or touch a capacitor.
- Never let the LLM change the verdict. Triage sets it.
- Never publish unreviewed records. Never guess an unknown code.
- Reply in the user's language.
