# Airisto Error-Code RAG: Basic Blueprint (v1)

## 1. Purpose

Tell a customer or technician what an AC error code means, what is safe to try, and whether a technician is needed. If one is needed, hand the customer to WhatsApp booking with the context pre-filled.

**Core design rule:** the diagnosis comes from a reviewed record, not from the model's memory. The LLM understands the question and phrases the reply. It does not invent facts.

## 2. Request flow

```
User message ("LG AC CH 07 not cooling", Hinglish ok)
        |
        v
[1] Query understanding
    extract: brand, code, symptoms, language
        |
        v
[2] Retrieval
    a) exact match on (brand, normalized code)  --> 1 record
    b) else vector search (brand-filtered)      --> top 5 records
    c) else below threshold                     --> NO MATCH
        |
        v
[3] Triage (rule-based)
    verdict + track come from the record
    follow-up "still showing" escalates the verdict
        |
        v
[4] Response builder
    diagnosis card filled from record fields
    LLM writes a short friendly wrapper in the user's language
        |
        v
[5] Booking handoff (only if verdict needs a technician)
    WhatsApp link with brand, code, diagnosis pre-filled

NO MATCH path: say so honestly, give general safe advice, offer technician booking.
```

## 3. Components (build order)

| # | Component | File | Input | Output |
|---|---|---|---|---|
| 1 | Schema + validator (done) | `schema.py` | CSV | validated records |
| 2 | Indexer | `indexer.py` | validated records | database + vector index |
| 3 | Retriever | `retriever.py` | brand, code, free text | ranked records or no-match |
| 4 | Generator | `generator.py` | query + records | wrapper text in user's language |
| 5 | Triage + booking | `triage.py` | record, conversation state | final verdict, WhatsApp URL |
| 6 | Evaluation | `eval/run_eval.py` | test queries | accuracy report |
| 7 | Streamlit UI | `app_streamlit.py` | user input | diagnosis card |
| 8 | API | `api.py` | HTTP request | JSON response |
| 9 | SEO pages | `seo/generate_pages.py` | records | static page per code |

## 4. Data and indexing

- Source of truth: `data/error_codes.csv`, checked by `schema.py`.
- Only records with `reviewed_by` and `last_reviewed` set are published to users.
- Text embedded per record: brand + model series + code + meaning + likely causes + a short 'symptoms' phrasing. Put the code in several forms (E4, E 4) so fuzzy queries match.
- Store the full record as metadata so retrieval returns the complete record, not a snippet.
- Prototype store: ChromaDB (persistent folder). Later option: Supabase with pgvector.
- Embedding model: use a multilingual one so Hinglish works. Candidates to test in Component 6: `paraphrase-multilingual-MiniLM-L12-v2`, `multilingual-e5-small`.

## 5. Retrieval logic

1. Normalize the code (`normalize_code` in `schema.py`).
2. If brand and code are known, do an exact match. Return that one record.
3. Otherwise run vector search, filtered by brand when known, `top_k = 5`.
4. If the best similarity is below a threshold, return NO MATCH. Tune the threshold on the test set; do not guess it.
5. If several model-specific records exist for one code, prefer the matching `model_series`, else ask the user for the model.

## 6. Generation contract

Inputs to the LLM: user message, retrieved record(s), language. Rules:

- Use only the supplied records. If none, say you are not sure.
- Never tell users to open a sealed unit, handle refrigerant or touch a capacitor.
- Never change the verdict. It is set by triage.
- Reply in the user's language.

Response shape (same for API and UI):

```json
{
  "brand": "...",
  "code": "...",
  "meaning": "...",
  "user_checks": ["..."],
  "verdict": "self_fix | needs_service | technician_required | urgent",
  "warnings": ["..."],
  "source": "...",
  "wrapper": "short friendly text from the LLM",
  "book_url": "https://wa.me/... or null",
  "match_type": "exact | semantic | none"
}
```

## 7. Triage and booking

| Verdict | Service track | Booking button |
|---|---|---|
| self_fix | none | No (offer after a failed fix) |
| needs_service | service (scheduled) | Yes |
| technician_required | repair | Yes |
| urgent | repair (fast) | Yes, with a switch-off warning first |

- **Escalation:** if the user says the self-fix did not work, change `self_fix` to `technician_required` by rule, not by LLM judgment.
- **Booking link:** `https://wa.me/<WHATSAPP_NUMBER>?text=<url-encoded message>`. The message carries brand, model, code, verdict and the website discount code for the flat ₹50 off.

## 8. Folder structure

```
airisto_rag/
  schema.py            (done)
  error_codes_template.csv
  ARCHITECTURE.md
  data/error_codes.csv
  indexer.py
  retriever.py
  generator.py
  triage.py
  eval/test_queries.csv
  eval/run_eval.py
  app_streamlit.py
  api.py
  seo/generate_pages.py
  requirements.txt
  .env.example
```

## 9. Configuration (`.env.example`)

```
GROQ_API_KEY=
LLM_MODEL=              # set the model name here, not in code
EMBEDDING_MODEL=
CHROMA_PATH=./chroma_db
WHATSAPP_NUMBER=
SIMILARITY_THRESHOLD=   # tuned in Component 6
```

## 10. Key function signatures

```python
# indexer.py
def build_index(records: list[ErrorCodeRecord]) -> None

# retriever.py
def retrieve(query: str, brand: str | None = None, code: str | None = None) -> RetrievalResult

# triage.py
def decide(record: ErrorCodeRecord, state: ConversationState) -> Decision
def booking_url(record: ErrorCodeRecord, decision: Decision) -> str | None

# generator.py
def build_response(query: str, result: RetrievalResult, decision: Decision, lang: str) -> dict
```

## 11. Evaluation (Component 6)

About 100 test queries, each with the expected record and verdict. Include misspellings ('E 4'), Hinglish, symptom-only queries, and codes that are deliberately not in the data. Track:

- Right record retrieved (exact and semantic separately)
- Right verdict shown
- Honest 'not sure' on unknown codes
- Zero cases where the bot tells the user to do something unsafe

## 12. Not in v1

Photo upload of the screen or remote, technician mode with ranked causes, fine-tuned models, geyser and purifier codes, user accounts. Revisit after real usage logs exist.

## 13. Open items

- The 5-6 launch brands and the first batch of real, technician-reviewed codes.
- Final WhatsApp Business number and discount code wording.
