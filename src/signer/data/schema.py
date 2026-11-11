"""Schema validation for CA-SpeechNER records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from signer.constants import LABELS, LANGUAGES

SUBTYPE_TO_LABEL = {
    "person": "PER",
    "gpe": "LOC",
    "geo": "LOC",
    "facility": "LOC",
    "organization": "ORG",
    "event": "MISC",
    "work": "MISC",
    "product": "MISC",
    "law": "MISC",
    "title": "MISC",
}

TEXT_AUDIO_CONSISTENCY_VALUES = {"ok", "minor_mismatch", "review_or_drop", "drop"}
ANNOTATION_STATUS_VALUES = {"unlabeled", "labeled", "adjudicated", "rejected"}
ADJUDICATION_STATUS_VALUES = {"not_reviewed", "single_pass", "double_annotated", "adjudicated"}


@dataclass(frozen=True)
class EntitySpan:
    entity_id: str
    start: int
    end: int
    text: str
    label: str
    subtype: str

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EntitySpan":
        return cls(
            entity_id=str(payload["entity_id"]),
            start=int(payload["start"]),
            end=int(payload["end"]),
            text=str(payload["text"]),
            label=str(payload["label"]),
            subtype=str(payload["subtype"]),
        )

    def validate(self, normalized_transcript: str) -> None:
        if self.label not in LABELS:
            raise ValueError(f"Unsupported label: {self.label}")
        if self.subtype not in SUBTYPE_TO_LABEL:
            raise ValueError(f"Unsupported subtype: {self.subtype}")
        if SUBTYPE_TO_LABEL[self.subtype] != self.label:
            raise ValueError(f"Subtype {self.subtype} does not map to label {self.label}")
        if not 0 <= self.start < self.end <= len(normalized_transcript):
            raise ValueError(f"Invalid entity offsets: {self.start}, {self.end}")
        if normalized_transcript[self.start : self.end] != self.text:
            raise ValueError(
                f"Entity text mismatch for {self.entity_id}: "
                f"{normalized_transcript[self.start:self.end]!r} != {self.text!r}"
            )


@dataclass(frozen=True)
class BaseSample:
    sample_id: str
    language: str
    split: str
    audio: dict[str, Any]
    text: dict[str, Any]
    entities: tuple[EntitySpan, ...]
    metadata: dict[str, Any]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BaseSample":
        return cls(
            sample_id=str(payload["id"]),
            language=str(payload["language"]),
            split=str(payload["split"]),
            audio=dict(payload["audio"]),
            text=dict(payload["text"]),
            entities=tuple(EntitySpan.from_dict(entity) for entity in payload["entities"]),
            metadata=dict(payload.get("metadata", {})),
        )

    def validate(self) -> None:
        if self.language not in LANGUAGES:
            raise ValueError(f"Unsupported language: {self.language}")
        if self.split not in {"train", "dev", "test"}:
            raise ValueError(f"Unsupported split: {self.split}")
        if int(self.audio["sample_rate"]) != 16000:
            raise ValueError(f"Unexpected sample rate: {self.audio['sample_rate']}")
        normalized = str(self.text["normalized_transcript"])
        if not normalized:
            raise ValueError("normalized_transcript must be non-empty")
        for entity in self.entities:
            entity.validate(normalized)
        self._validate_non_overlapping_entities()
        self._validate_metadata()

    def _validate_non_overlapping_entities(self) -> None:
        ordered = sorted(self.entities, key=lambda entity: (entity.start, entity.end, entity.entity_id))
        previous_end = -1
        previous_entity_id = ""
        for entity in ordered:
            if entity.start < previous_end:
                raise ValueError(
                    "Overlapping or nested entities are not allowed: "
                    f"{previous_entity_id} overlaps with {entity.entity_id}"
                )
            previous_end = entity.end
            previous_entity_id = entity.entity_id

    def _validate_metadata(self) -> None:
        required_fields = {
            "source",
            "quality_flag",
            "text_audio_consistency",
            "annotation_status",
        }
        missing = sorted(required_fields - set(self.metadata.keys()))
        if missing:
            raise ValueError(f"Missing metadata fields: {missing}")

        consistency = str(self.metadata["text_audio_consistency"])
        if consistency not in TEXT_AUDIO_CONSISTENCY_VALUES:
            raise ValueError(f"Unsupported text_audio_consistency: {consistency}")

        annotation_status = str(self.metadata["annotation_status"])
        if annotation_status not in ANNOTATION_STATUS_VALUES:
            raise ValueError(f"Unsupported annotation_status: {annotation_status}")

        adjudication_status = self.metadata.get("adjudication_status")
        if adjudication_status is not None and adjudication_status not in ADJUDICATION_STATUS_VALUES:
            raise ValueError(f"Unsupported adjudication_status: {adjudication_status}")

        if "annotator_id" in self.metadata and not isinstance(self.metadata["annotator_id"], str):
            raise ValueError("annotator_id must be a string")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.sample_id,
            "language": self.language,
            "split": self.split,
            "audio": self.audio,
            "text": self.text,
            "entities": [entity.__dict__ for entity in self.entities],
            "metadata": self.metadata,
        }
