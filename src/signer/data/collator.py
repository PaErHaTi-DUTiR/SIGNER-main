"""Batch collation for SIGNER training and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pydub import AudioSegment
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoFeatureExtractor, WhisperFeatureExtractor

from signer.constants import IGNORE_INDEX


@dataclass
class SignerBatchCollator:
    whisper_name_or_path: str
    speech_encoder_type: str = "whisper"
    sample_rate: int = 16000
    max_audio_seconds: float = 30.0

    def __post_init__(self) -> None:
        encoder_type = self.speech_encoder_type.lower()
        if encoder_type == "whisper":
            self.feature_extractor = WhisperFeatureExtractor.from_pretrained(self.whisper_name_or_path)
        elif encoder_type in {"wav2vec2", "mms"}:
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.whisper_name_or_path)
        else:
            raise ValueError(f"unsupported speech_encoder_type: {self.speech_encoder_type}")
        zero_waveform = torch.zeros(int(self.sample_rate * self.max_audio_seconds), dtype=torch.float32)
        empty = self.feature_extractor(
            zero_waveform.numpy(),
            sampling_rate=self.sample_rate,
            return_attention_mask=True,
            return_tensors="pt",
        )
        self.empty_audio_features, _ = self._extract_features_from_output(empty)
        self.empty_audio_mask = torch.zeros(self.empty_audio_features.shape[0], dtype=torch.bool)

    def _extract_features_from_output(self, features) -> tuple[torch.Tensor, torch.Tensor]:
        if "input_features" in features:
            audio_features = features["input_features"][0].transpose(0, 1)
        elif "input_values" in features:
            audio_features = features["input_values"][0].unsqueeze(-1)
        else:
            raise KeyError("feature extractor output must contain input_features or input_values")

        if "attention_mask" in features:
            audio_mask = features["attention_mask"][0].bool()
        else:
            audio_mask = torch.ones(audio_features.shape[0], dtype=torch.bool)
        return audio_features, audio_mask

    def _decode_audio(self, audio_path: str) -> np.ndarray:
        if audio_path.endswith(".npy"):
            waveform = np.load(audio_path).astype(np.float32)
            if waveform.size == 0:
                return np.zeros(int(self.sample_rate * self.max_audio_seconds), dtype=np.float32)
            max_abs = max(float(np.max(np.abs(waveform))), 1.0)
            if max_abs > 1.0:
                waveform = waveform / max_abs
            max_samples = int(self.sample_rate * self.max_audio_seconds)
            return waveform[:max_samples]
        segment = AudioSegment.from_file(audio_path)
        segment = segment.set_channels(1).set_frame_rate(self.sample_rate)
        samples = np.array(segment.get_array_of_samples())
        if samples.size == 0:
            return np.zeros(int(self.sample_rate * self.max_audio_seconds), dtype=np.float32)
        max_abs = max(float(np.iinfo(samples.dtype).max), 1.0)
        waveform = samples.astype(np.float32) / max_abs
        max_samples = int(self.sample_rate * self.max_audio_seconds)
        return waveform[:max_samples]

    def _load_audio(self, audio_path: str) -> tuple[torch.Tensor, torch.Tensor]:
        waveform = self._decode_audio(audio_path)
        features = self.feature_extractor(
            waveform,
            sampling_rate=self.sample_rate,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return self._extract_features_from_output(features)

    def _load_cached_speech(self, cache_path: str, sample_id: str) -> tuple[torch.Tensor, torch.Tensor]:
        path = Path(cache_path)
        if not path.exists():
            raise FileNotFoundError(f"missing cached speech feature for sample_id={sample_id}: {path}")
        try:
            cached = np.load(path)
        except Exception as exc:  # pragma: no cover - keeps sample_id in low-level numpy failures.
            raise ValueError(f"failed to load cached speech feature for sample_id={sample_id}: {path}") from exc
        required = {"speech_states", "speech_mask"}
        missing = required.difference(cached.files)
        if missing:
            raise KeyError(f"cached speech feature for sample_id={sample_id} is missing keys: {sorted(missing)}")
        speech_states = np.asarray(cached["speech_states"], dtype=np.float32)
        speech_mask = np.asarray(cached["speech_mask"], dtype=bool)
        if speech_states.ndim != 2:
            raise ValueError(
                f"cached speech_states must be 2D for sample_id={sample_id}; got shape={speech_states.shape}"
            )
        if speech_mask.ndim != 1:
            raise ValueError(f"cached speech_mask must be 1D for sample_id={sample_id}; got shape={speech_mask.shape}")
        if speech_states.shape[0] != speech_mask.shape[0]:
            raise ValueError(
                "cached speech_states and speech_mask length mismatch "
                f"for sample_id={sample_id}: states={speech_states.shape}, mask={speech_mask.shape}"
            )
        if speech_states.shape[0] == 0 or speech_states.shape[1] == 0:
            raise ValueError(f"cached speech_states is empty for sample_id={sample_id}: shape={speech_states.shape}")
        return torch.tensor(speech_states, dtype=torch.float32), torch.tensor(speech_mask, dtype=torch.bool)

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        sample_ids = [item["sample_id"] for item in batch]
        transcripts = [item["transcript"] for item in batch]
        languages = [item["language"] for item in batch]
        language_ids = torch.tensor([item["language_id"] for item in batch], dtype=torch.long)
        view_type = torch.tensor([item["view_type"] for item in batch], dtype=torch.long)

        text_input_ids = []
        placeholder_input_ids = []
        text_attention_mask = []
        token_labels = []
        label_mask = []
        audio_features = []
        audio_mask = []
        speech_states = []
        use_cached_speech = False
        offset_mappings = []

        for item in batch:
            ids = item["input_ids"] if item["view_name"] != "speech-only" else item["placeholder_input_ids"]
            text_input_ids.append(torch.tensor(ids, dtype=torch.long))
            placeholder_input_ids.append(torch.tensor(item["placeholder_input_ids"], dtype=torch.long))
            text_attention_mask.append(torch.tensor(item["attention_mask"], dtype=torch.bool))
            token_labels.append(torch.tensor(item["token_labels"], dtype=torch.long))
            offset_mappings.append(item["offset_mapping"])
            label_mask.append(
                torch.tensor([label != IGNORE_INDEX for label in item["token_labels"]], dtype=torch.bool)
            )

            cache_path = item.get("speech_feature_cache_path")
            if item["view_name"] != "text-only" and cache_path:
                cached_states, cached_mask = self._load_cached_speech(cache_path, item["sample_id"])
                speech_states.append(cached_states)
                audio_mask.append(cached_mask)
                audio_features.append(self.empty_audio_features.clone())
                use_cached_speech = True
            elif item["view_name"] == "text-only":
                audio_features.append(self.empty_audio_features.clone())
                audio_mask.append(self.empty_audio_mask.clone())
            else:
                features, mask = self._load_audio(item["audio_path"])
                audio_features.append(features)
                audio_mask.append(mask)

        batch_out = {
            "sample_ids": sample_ids,
            "transcripts": transcripts,
            "languages": languages,
            "language_ids": language_ids,
            "view_type": view_type,
            "audio_features": pad_sequence(audio_features, batch_first=True, padding_value=0.0),
            "audio_mask": pad_sequence(audio_mask, batch_first=True, padding_value=False),
            "text_input_ids": pad_sequence(text_input_ids, batch_first=True, padding_value=1),
            "placeholder_input_ids": pad_sequence(placeholder_input_ids, batch_first=True, padding_value=1),
            "text_attention_mask": pad_sequence(text_attention_mask, batch_first=True, padding_value=False),
            "token_labels": pad_sequence(token_labels, batch_first=True, padding_value=IGNORE_INDEX),
            "label_mask": pad_sequence(label_mask, batch_first=True, padding_value=False),
            "offset_mappings": offset_mappings,
            "entity_spans": [item["entities"] for item in batch],
        }
        if use_cached_speech:
            if len(speech_states) != len(batch):
                raise ValueError("cached and non-cached speech states cannot be mixed in one batch")
            batch_out["speech_states"] = pad_sequence(speech_states, batch_first=True, padding_value=0.0)
        return batch_out
