"""Dataset definitions for CA-SpeechNER."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset
from transformers import AutoTokenizer

from signer.constants import IGNORE_INDEX, LANGUAGE_TO_ID, TAG_TO_ID, VIEW_NAME_TO_ID
from signer.data.schema import BaseSample
from signer.utils.io import read_jsonl


def _normalize_probabilities(probabilities: dict[str, float]) -> tuple[list[str], list[float]]:
    total = sum(probabilities.values())
    names = list(probabilities.keys())
    return names, [probabilities[name] / total for name in names]


def encode_entities_with_bio(
    tokenizer: AutoTokenizer,
    normalized_transcript: str,
    entities: tuple,
    max_length: int,
) -> dict[str, Any]:
    encoded = tokenizer(
        normalized_transcript,
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
    )
    offsets = encoded["offset_mapping"]
    labels = [IGNORE_INDEX] * len(offsets)
    for index, (start, end) in enumerate(offsets):
        if end <= start:
            continue
        labels[index] = TAG_TO_ID["O"]
        for entity in entities:
            if start == entity.start and end <= entity.end:
                labels[index] = TAG_TO_ID[f"B-{entity.label}"]
                break
            if entity.start < start < entity.end:
                labels[index] = TAG_TO_ID[f"I-{entity.label}"]
                break
    encoded["token_labels"] = labels
    return encoded


class CASpeechNERDataset(Dataset):
    """Dataset that derives a view from each CA-SpeechNER base sample."""

    def __init__(
        self,
        jsonl_path: str | Path,
        dataset_root: str | Path,
        tokenizer_name_or_path: str,
        max_text_length: int = 256,
        default_view: str = "speech-text",
        sample_views: bool = False,
        view_probabilities: dict[str, float] | None = None,
        speech_feature_cache_dir: str | Path | None = None,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.speech_feature_cache_dir = Path(speech_feature_cache_dir) if speech_feature_cache_dir else None
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, use_fast=True)
        self.max_text_length = max_text_length
        self.default_view = default_view
        self.sample_views = sample_views
        self.view_names, self.view_weights = _normalize_probabilities(
            view_probabilities
            or {"text-only": 0.3, "speech-only": 0.3, "speech-text": 0.4}
        )
        self.samples = [BaseSample.from_dict(record) for record in read_jsonl(jsonl_path)]
        for sample in self.samples:
            sample.validate()

    def __len__(self) -> int:
        return len(self.samples)

    def _choose_view(self) -> str:
        if not self.sample_views:
            return self.default_view
        return random.choices(self.view_names, weights=self.view_weights, k=1)[0]

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        view_name = self._choose_view()
        normalized = sample.text["normalized_transcript"]
        encoded = encode_entities_with_bio(
            tokenizer=self.tokenizer,
            normalized_transcript=normalized,
            entities=sample.entities,
            max_length=self.max_text_length,
        )
        input_ids = encoded["input_ids"]
        placeholder_ids = [
            token if token in self.tokenizer.all_special_ids else self.tokenizer.unk_token_id
            for token in input_ids
        ]
        item = {
            "sample_id": sample.sample_id,
            "language_id": LANGUAGE_TO_ID[sample.language],
            "language": sample.language,
            "view_name": view_name,
            "view_type": VIEW_NAME_TO_ID[view_name],
            "audio_path": str((self.dataset_root / sample.audio["path"]).resolve()),
            "transcript": normalized,
            "entities": [entity.__dict__ for entity in sample.entities],
            "input_ids": input_ids,
            "placeholder_input_ids": placeholder_ids,
            "attention_mask": encoded["attention_mask"],
            "token_labels": encoded["token_labels"],
            "offset_mapping": encoded["offset_mapping"],
        }
        if self.speech_feature_cache_dir is not None:
            item["speech_feature_cache_path"] = str(self.speech_feature_cache_dir / f"{sample.sample_id}.npz")
        return item
