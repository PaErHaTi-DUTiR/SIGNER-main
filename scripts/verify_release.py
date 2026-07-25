#!/usr/bin/env python3
"""Run lightweight integrity checks for the public SIGNER release."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "ca_stner_sample" / "sample.jsonl"
LANGUAGES = {"kk", "ky", "tg", "tk", "uz"}
LABELS = {"PER", "LOC", "ORG"}
TAGS = {"O"} | {f"{prefix}-{label}" for prefix in "BI" for label in LABELS}
SECRET_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9._-]{16,}"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
)


def verify_sample() -> list[str]:
    errors = []
    records = []
    with SAMPLE.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                errors.append(f"{SAMPLE}:{line_number}: invalid JSON: {exc}")

    languages = {record.get("language") for record in records}
    if languages != LANGUAGES:
        errors.append(f"sample languages are {sorted(languages)}, expected {sorted(LANGUAGES)}")

    ids = set()
    for record in records:
        sample_id = record.get("id")
        if sample_id in ids:
            errors.append(f"duplicate sample id: {sample_id}")
        ids.add(sample_id)

        tokens = record.get("tokens", [])
        tags = record.get("ner_tags", [])
        if len(tokens) != len(tags):
            errors.append(f"{sample_id}: token/tag length mismatch")
        invalid_tags = set(tags) - TAGS
        if invalid_tags:
            errors.append(f"{sample_id}: invalid tags {sorted(invalid_tags)}")

        previous_label = None
        for index, tag in enumerate(tags):
            if tag.startswith("I-") and previous_label != tag[2:]:
                errors.append(f"{sample_id}: invalid BIO transition at token {index}")
            previous_label = None if tag == "O" else tag[2:]

        text = record.get("text", "")
        for entity in record.get("entities", []):
            start, end = int(entity["start"]), int(entity["end"])
            if entity["label"] not in LABELS:
                errors.append(f"{sample_id}: invalid entity label {entity['label']}")
            if text[start:end] != entity["text"]:
                errors.append(
                    f"{sample_id}: offset mismatch {text[start:end]!r} != {entity['text']!r}"
                )
    return errors


def verify_no_secrets() -> list[str]:
    errors = []
    excluded = {".git"}
    for path in ROOT.rglob("*"):
        if not path.is_file() or any(part in excluded for part in path.parts):
            continue
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".pdf"}:
            continue
        if path.stat().st_size > 2_000_000:
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for pattern in SECRET_PATTERNS:
            if pattern.search(content):
                errors.append(f"possible secret in {path.relative_to(ROOT)}")
    return errors


def main() -> None:
    errors = verify_sample() + verify_no_secrets()
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)
    print("Release checks passed: schema, offsets, BIO tags, languages, and secret scan.")


if __name__ == "__main__":
    main()

