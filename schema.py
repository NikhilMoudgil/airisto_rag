"""Airisto RAG - Component 1: record schema + CSV validator (standard library only).

Usage:
    python schema.py error_codes.csv

CSV list columns (likely_causes, user_checks, warnings, typical_parts, symptoms)
are separated with ' | '.
"""
import csv
import re
import sys
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

VERDICTS = ("self_fix", "needs_service", "technician_required", "urgent")
TRACKS = ("none", "service", "repair")
LIST_FIELDS = ("likely_causes", "user_checks", "warnings", "typical_parts", "symptoms")
REQUIRED_TEXT = ("brand", "code", "meaning", "source")


def normalize_code(code: str) -> str:
    """'e 4', 'E-4', 'e4' -> 'E4' so lookups match however users type it."""
    return re.sub(r"[\s\-_.]", "", code).upper()


@dataclass
class ErrorCodeRecord:
    brand: str
    code: str
    meaning: str
    verdict: str
    track: str
    source: str
    likely_causes: list = field(default_factory=list)
    user_checks: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    typical_parts: list = field(default_factory=list)
    symptoms: list = field(default_factory=list)  # how users describe it, incl. Hinglish
    model_series: str = ""
    reviewed_by: str = ""
    last_reviewed: Optional[date] = None

    def validate(self) -> list:
        """Return a list of problems (empty list = record is valid)."""
        problems = []
        for name in REQUIRED_TEXT:
            if not getattr(self, name).strip():
                problems.append(f"{name} must not be blank")
        if self.verdict not in VERDICTS:
            problems.append(f"verdict must be one of {VERDICTS}, got '{self.verdict}'")
        if self.track not in TRACKS:
            problems.append(f"track must be one of {TRACKS}, got '{self.track}'")
        if not self.likely_causes:
            problems.append("likely_causes needs at least one entry")
        if self.verdict == "self_fix":
            if self.track != "none":
                problems.append("self_fix records must have track=none")
            if not self.user_checks:
                problems.append("self_fix records need user_checks")
        if self.verdict == "needs_service" and self.track != "service":
            problems.append("needs_service records must have track=service")
        if self.verdict in ("technician_required", "urgent") and self.track != "repair":
            problems.append(f"{self.verdict} records must have track=repair")
        if self.verdict == "urgent" and not self.warnings:
            problems.append("urgent records need at least one warning")
        return problems

    @property
    def code_norm(self) -> str:
        return normalize_code(self.code)

    @property
    def record_id(self) -> str:
        parts = [self.brand, self.model_series or "all", self.code_norm]
        return "-".join(re.sub(r"\s+", "", p).lower() for p in parts)

    @property
    def is_published(self) -> bool:
        """Only records reviewed by a technician should go live."""
        return bool(self.reviewed_by and self.last_reviewed)


def _split(value: str) -> list:
    return [x.strip() for x in (value or "").split("|") if x.strip()]


def _row_to_record(row: dict) -> ErrorCodeRecord:
    get = lambda k: (row.get(k) or "").strip()
    reviewed = get("last_reviewed")
    return ErrorCodeRecord(
        brand=get("brand"),
        model_series=get("model_series"),
        code=get("code"),
        meaning=get("meaning"),
        likely_causes=_split(row.get("likely_causes")),
        user_checks=_split(row.get("user_checks")),
        verdict=get("verdict"),
        track=get("track"),
        warnings=_split(row.get("warnings")),
        typical_parts=_split(row.get("typical_parts")),
        symptoms=_split(row.get("symptoms")),
        source=get("source"),
        reviewed_by=get("reviewed_by"),
        last_reviewed=date.fromisoformat(reviewed) if reviewed else None,
    )


def load_csv(path: str):
    """Return (records, errors). Errors are human-readable strings."""
    records, errors, seen = [], [], {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for line_no, row in enumerate(csv.DictReader(f), start=2):
            try:
                rec = _row_to_record(row)
            except ValueError as e:  # e.g. bad date format
                errors.append(f"row {line_no}: {e} (dates must be YYYY-MM-DD)")
                continue
            problems = rec.validate()
            if problems:
                errors.append(f"row {line_no}: " + "; ".join(problems))
                continue
            key = (rec.brand.lower(), rec.model_series.lower(), rec.code_norm)
            if key in seen:
                errors.append(f"row {line_no}: duplicate of row {seen[key]} ({rec.record_id})")
                continue
            seen[key] = line_no
            records.append(rec)
    return records, errors


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python schema.py error_codes.csv")
    recs, errs = load_csv(sys.argv[1])
    published = sum(r.is_published for r in recs)
    print(f"{len(recs)} valid records ({published} reviewed/publishable), {len(errs)} errors")
    for e in errs:
        print("  -", e)
    sys.exit(1 if errs else 0)
