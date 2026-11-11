"""Isolated candidate repair for duplicate B starts in the restored BIO chain.

This is not an alternative character-overlap projector. Token ownership follows
the legacy rules exactly; only repeated B tags within the same entity run change
to I. Callers must explicitly opt in. The server's signer package is untouched.
"""
from collections.abc import Mapping, Sequence
from typing import Any


def legacy_owner(start: int, end: int, entities: Sequence[Any]) -> int | None:
    """First owner under legacy order/conditions, including boundary limitations."""
    if end <= start:
        return None
    for index, entity in enumerate(entities):
        if start == entity.start and end <= entity.end:
            return index
        if entity.start < start < entity.end:
            return index
    return None


def repair_duplicate_b_starts(
    encoded: Mapping[str, Any], entities: Sequence[Any],
    tag_to_id: Mapping[str, int], ignore_index: int = -100,
) -> dict[str, Any]:
    """Return a new mapping; preserve every field except a new labels list.

    Does not invent supervision for previously unrepresented entities, repair
    truncation, trim offsets, collapse tokens, change the head, or change gold.
    A different adjacent entity of the same type retains its own B tag.
    Inconsistent input raises instead of silently changing the protocol.
    """
    offsets = encoded['offset_mapping']
    labels = list(encoded['token_labels'])
    if len(offsets) != len(labels):
        raise ValueError('offset/label length mismatch')
    previous_owner = None
    for index, ((start, end), label) in enumerate(zip(offsets, labels)):
        if label == ignore_index:
            continue  # Legacy evaluation also compacts ignored positions.
        owner = legacy_owner(start, end, entities)
        if owner is None:
            if label != tag_to_id['O']:
                raise ValueError('Non-O label without a legacy entity owner')
        else:
            entity = entities[owner]
            begin = tag_to_id[f'B-{entity.label}']
            inside = tag_to_id[f'I-{entity.label}']
            expected = begin if start == entity.start and end <= entity.end else inside
            if label != expected:
                raise ValueError('Input is not legacy BIO supervision')
            if label == begin and owner == previous_owner:
                labels[index] = inside
        previous_owner = owner
    return {**encoded, 'token_labels': labels}
