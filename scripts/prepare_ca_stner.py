#!/usr/bin/env python3
"""Convert the public CA-STNER layout into SIGNER's canonical JSONL format."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from signer.data.schema import BaseSample  # noqa: E402

LANGUAGES = ("kk", "ky", "tg", "tk", "uz")
SPLITS = ("train", "dev", "test")
LABEL_TO_SUBTYPE = {
    "PER": "person",
    "ORG": "organization",
    "LOC": "geo",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--validate", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"missing split: {path}")
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
    return records


def convert_record(record: dict, language: str) -> dict:
    text = record["text"]
    if not isinstance(text, str):
        text = text.get("transcript", text.get("normalized_transcript", ""))

    entities = []
    for index, entity in enumerate(record.get("entities", [])):
        label = entity["label"]
        if label not in LABEL_TO_SUBTYPE:
            raise ValueError(f"unsupported label {label!r} in {record['id']}")
        entities.append(
            {
                "entity_id": f"{record['id']}_e{index}",
                "start": int(entity["start"]),
                "end": int(entity["end"]),
                "text": entity["text"],
                "label": label,
                "subtype": LABEL_TO_SUBTYPE[label],
            }
        )

    source_audio_path = str(record["audio"]["path"])
    if not source_audio_path.startswith(f"{language}/"):
        source_audio_path = f"{language}/{source_audio_path}"

    return {
        "id": record["id"],
        "language": language,
        "split": record["split"],
        "audio": {
            "path": source_audio_path,
            "sample_rate": int(record["audio"].get("sample_rate", 16000)),
            "duration_sec": float(record["audio"].get("duration_sec", 0.0)),
        },
        "text": {
            "transcript": text,
            "normalized_transcript": text,
        },
        "entities": entities,
        "metadata": {
            "source": record.get("metadata", {}).get("source", "ca_stner"),
            "quality_flag": "ok",
            "text_audio_consistency": "ok",
            "annotation_status": "adjudicated",
            "annotator_id": "human_verified",
            "adjudication_status": "adjudicated",
        },
    }


def write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    combined: dict[str, list[dict]] = defaultdict(list)
    seen_ids: dict[str, str] = {}

    for language in LANGUAGES:
        for split in SPLITS:
            public_records = read_jsonl(input_root / language / f"{split}.jsonl")
            canonical_records = []
            for public_record in public_records:
                sample_id = str(public_record["id"])
                if sample_id in seen_ids:
                    raise ValueError(
                        f"duplicate sample id {sample_id!r} in {split}; "
                        f"previously seen in {seen_ids[sample_id]}"
                    )
                seen_ids[sample_id] = split
                canonical = convert_record(public_record, language)
                if args.validate:
                    BaseSample.from_dict(canonical).validate()
                canonical_records.append(canonical)
                combined[split].append(canonical)
            write_jsonl(
                canonical_records,
                output_root / "by_language" / language / f"{split}.jsonl",
            )
            print(f"{language}/{split}: {len(canonical_records)} records")

    for split in SPLITS:
        write_jsonl(combined[split], output_root / f"{split}.jsonl")
        print(f"combined/{split}: {len(combined[split])} records")


if __name__ == "__main__":
    main()

