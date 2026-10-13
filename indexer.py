"""Airisto RAG - Component 2: indexer.

Turns a validated CSV of error-code records into one index file
(index/index.json) holding, per record: the full record, the text that was
embedded, and its vector.

Design choices:
  * Brute-force cosine search over a JSON file. For a few hundred to a few
    thousand records this is fast, has no database to run, and is easy to debug.
    Move to pgvector/Chroma only if the data outgrows it.
  * The index stores the embedder's name. The retriever must use the same one,
    otherwise vectors are not comparable (load_index checks this).
  * Bad data never reaches customers: the build is refused if the CSV has
    validation errors, and unreviewed records are left out unless asked.

Usage:
    python indexer.py data/error_codes.csv
    python indexer.py data/test_records_FAKE.csv --include-unreviewed
    python indexer.py data/error_codes.csv --embedder st
"""
import argparse
import json
import os
import re
import sys
from dataclasses import asdict
from datetime import date, datetime, timezone

from embedders import get_embedder
from schema import ErrorCodeRecord, load_csv, normalize_code

INDEX_VERSION = 1
INDEX_FILE = "index.json"


# ---------- text that gets embedded ----------

def code_variants(code: str) -> list:
    """All the ways a customer might type a code: 'CH 07' -> CH 07, CH07, CH-07."""
    norm = normalize_code(code)
    spaced = re.sub(r"([A-Za-z]+)(\d+)", r"\1 \2", norm)
    hyphen = spaced.replace(" ", "-")
    seen, out = set(), []
    for v in (code.strip(), norm, spaced, hyphen):
        if v and v.lower() not in seen:
            seen.add(v.lower())
            out.append(v)
    return out


def index_text(rec: ErrorCodeRecord) -> str:
    """One searchable passage per record. Written so brand, code spellings,
    meaning, symptoms and causes all contribute to matching."""
    parts = [f"Brand: {rec.brand}."]
    if rec.model_series:
        parts.append(f"Model: {rec.model_series}.")
    parts.append(f"Error code: {', '.join(code_variants(rec.code))}.")
    parts.append(f"Meaning: {rec.meaning}.")
    if rec.symptoms:
        parts.append("Customers say: " + "; ".join(rec.symptoms) + ".")
    if rec.likely_causes:
        parts.append("Likely causes: " + "; ".join(rec.likely_causes) + ".")
    return " ".join(parts)


# ---------- record <-> dict (JSON friendly) ----------

def record_to_dict(rec: ErrorCodeRecord) -> dict:
    d = asdict(rec)
    d["last_reviewed"] = rec.last_reviewed.isoformat() if rec.last_reviewed else None
    return d


def record_from_dict(d: dict) -> ErrorCodeRecord:
    d = dict(d)
    lr = d.get("last_reviewed")
    d["last_reviewed"] = date.fromisoformat(lr) if lr else None
    return ErrorCodeRecord(**d)


# ---------- build / save / load ----------

def build_index(records: list, embedder, include_unreviewed: bool = False):
    """Return (index_dict, skipped_unreviewed_ids)."""
    kept, skipped = [], []
    for rec in records:
        if rec.is_published or include_unreviewed:
            kept.append(rec)
        else:
            skipped.append(rec.record_id)
    texts = [index_text(r) for r in kept]
    vectors = embedder.embed_documents(texts) if texts else []
    entries = [
        {"id": r.record_id, "text": t, "record": record_to_dict(r), "vector": v}
        for r, t, v in zip(kept, texts, vectors)
    ]
    index = {
        "version": INDEX_VERSION,
        "embedder": embedder.name,
        "dim": embedder.dim,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "count": len(entries),
        "entries": entries,
    }
    return index, skipped


def save_index(index: dict, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, INDEX_FILE)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False)
    os.replace(tmp, path)  # atomic: a crash never leaves a half-written index
    return path


def load_index(out_dir: str, embedder=None) -> dict:
    """Load the index. If an embedder is given, refuse a mismatched one."""
    path = os.path.join(out_dir, INDEX_FILE)
    with open(path, encoding="utf-8") as f:
        index = json.load(f)
    if index.get("version") != INDEX_VERSION:
        raise ValueError(f"index version {index.get('version')} != expected {INDEX_VERSION}; rebuild it")
    if embedder is not None and index["embedder"] != embedder.name:
        raise ValueError(
            f"index was built with '{index['embedder']}' but you are using '{embedder.name}'. "
            "Rebuild the index or use the same embedder."
        )
    return index


# ---------- command line ----------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build the Airisto error-code index.")
    ap.add_argument("csv", help="validated error-code CSV")
    ap.add_argument("--out", default="index", help="output folder (default: index)")
    ap.add_argument("--embedder", default="hashing", choices=["hashing", "st"])
    ap.add_argument("--model", default="", help="sentence-transformers model name")
    ap.add_argument("--query-prefix", default="")
    ap.add_argument("--doc-prefix", default="")
    ap.add_argument("--include-unreviewed", action="store_true",
                    help="also index records no technician has reviewed (testing only)")
    args = ap.parse_args(argv)

    records, errors = load_csv(args.csv)
    if errors:
        print(f"Refusing to build: {len(errors)} data error(s) in {args.csv}")
        for e in errors:
            print("  -", e)
        return 1

    embedder = get_embedder(args.embedder, args.model, args.query_prefix, args.doc_prefix)
    index, skipped = build_index(records, embedder, args.include_unreviewed)
    if index["count"] == 0:
        print("Nothing to index: no reviewed records. Add reviewed_by + last_reviewed, "
              "or use --include-unreviewed for testing.")
        return 1

    path = save_index(index, args.out)
    print(f"Indexed {index['count']} record(s) with {embedder.name} -> {path}")
    if skipped:
        print(f"Left out {len(skipped)} unreviewed record(s): {', '.join(skipped)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
