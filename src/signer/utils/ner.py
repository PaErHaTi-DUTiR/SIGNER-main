"""NER decoding and metric helpers."""

from __future__ import annotations

from typing import Iterable

from signer.constants import ID_TO_TAG


def decode_bio_tags(tag_ids: Iterable[int]) -> list[tuple[int, int, str]]:
    ids = [int(tag_id) for tag_id in tag_ids]
    spans: list[tuple[int, int, str]] = []
    start = None
    current_label = None
    for index, tag_id in enumerate(ids):
        tag = ID_TO_TAG[tag_id]
        if tag == "O":
            if start is not None and current_label is not None:
                spans.append((start, index, current_label))
                start = None
                current_label = None
            continue
        prefix, label = tag.split("-", 1)
        if prefix == "B" or label != current_label:
            if start is not None and current_label is not None:
                spans.append((start, index, current_label))
            start = index
            current_label = label
        elif prefix == "I" and start is None:
            start = index
            current_label = label
    if start is not None and current_label is not None:
        spans.append((start, len(ids), current_label))
    return spans


def compute_span_prf(predictions: Iterable[set[tuple]], references: Iterable[set[tuple]]) -> dict[str, float]:
    predicted_total = 0
    reference_total = 0
    matched_total = 0
    for predicted, reference in zip(predictions, references):
        predicted_total += len(predicted)
        reference_total += len(reference)
        matched_total += len(predicted & reference)
    precision = matched_total / predicted_total if predicted_total else 0.0
    recall = matched_total / reference_total if reference_total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "micro_f1": f1}
