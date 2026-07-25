"""Configuration objects for the SIGNER implementation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from signer.constants import LABELS


@dataclass
class DataConfig:
    dataset_root: str = "data/ca_stner"
    train_file: str = "../ca_stner_signer/train.jsonl"
    dev_file: str = "../ca_stner_signer/dev.jsonl"
    test_file: str = "../ca_stner_signer/test.jsonl"
    sample_rate: int = 16000
    max_audio_seconds: float = 30.0
    max_text_length: int = 256
    train_default_view: str = "speech-text"
    view_for_eval: str = "speech-text"
    sample_training_views: bool = True
    speech_feature_cache_dir: str | None = None
    train_view_probs: dict[str, float] = field(
        default_factory=lambda: {
            "text-only": 0.3,
            "speech-only": 0.3,
            "speech-text": 0.4,
        }
    )

    def resolve(self, root: str | Path) -> "DataConfig":
        base = Path(root)
        clone = DataConfig(**asdict(self))
        clone.dataset_root = str((base / self.dataset_root).resolve())
        if self.speech_feature_cache_dir:
            cache_path = Path(self.speech_feature_cache_dir)
            if not cache_path.is_absolute():
                cache_path = base / cache_path
            clone.speech_feature_cache_dir = str(cache_path.resolve())
        return clone


@dataclass
class ModelConfig:
    whisper_name_or_path: str = "openai/whisper-small"
    speech_encoder_type: str = "whisper"
    text_model_name_or_path: str = "xlm-roberta-base"
    hidden_size: int = 768
    num_labels: int = 2 * len(LABELS) + 1
    dropout: float = 0.1
    use_speech_branch: bool = True
    load_speech_encoder: bool = True
    freeze_speech_encoder: bool = False
    freeze_text_encoder: bool = False
    whisper_unfreeze_top_n_layers: int = 0  # 0=fully frozen when freeze_speech_encoder=True
    share_modality_embedding: bool = False
    fusion_type: str = "global"
    alignment_window_size: int = 16
    decoder_type: str = "softmax"
    loss_type: str = "cross_entropy"
    label_smoothing: float = 0.0
    focal_gamma: float = 2.0
    use_entity_prior: bool = False
    entity_prior_loss_weight: float = 0.0
    entity_prior_threshold: float = 0.0

    def resolve(self, root: str | Path) -> "ModelConfig":
        base = Path(root)
        clone = ModelConfig(**asdict(self))
        whisper_path = base / self.whisper_name_or_path
        if whisper_path.exists():
            clone.whisper_name_or_path = str(whisper_path.resolve())
        text_path = base / self.text_model_name_or_path
        if text_path.exists():
            clone.text_model_name_or_path = str(text_path.resolve())
        return clone


@dataclass
class TrainingConfig:
    batch_size: int = 2
    eval_batch_size: int = 2
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    num_epochs: int = 3
    max_steps: int | None = None
    gradient_clip_norm: float = 1.0
    gradient_accumulation_steps: int = 1
    consistency_weight: float = 0.2
    aux_text_loss_weight: float = 0.0
    audio_dropout: float = 0.3
    text_dropout: float = 0.3
    device: str = "cuda"
    output_dir: str = "outputs/runs/default"
    log_every_steps: int = 20
    num_workers: int = 0
    seed: int = 42
    positive_example_oversampling: float = 1.0
    non_o_loss_weight: float = 1.0
    init_checkpoint: str | None = None
    speech_encoder_lr_multiplier: float = 1.0  # multiply base LR for partial-unfreeze Whisper layers

    def resolve(self, root: str | Path) -> "TrainingConfig":
        base = Path(root)
        clone = TrainingConfig(**asdict(self))
        if self.init_checkpoint:
            checkpoint_path = base / self.init_checkpoint
            if checkpoint_path.exists():
                clone.init_checkpoint = str(checkpoint_path.resolve())
        return clone


@dataclass
class SignerConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SignerConfig":
        return cls(
            data=DataConfig(**payload.get("data", {})),
            model=ModelConfig(**payload.get("model", {})),
            training=TrainingConfig(**payload.get("training", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "data": asdict(self.data),
            "model": asdict(self.model),
            "training": asdict(self.training),
        }


def dump_config(config: SignerConfig, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config.to_dict(), handle, sort_keys=False, allow_unicode=True)


def load_config(config_path: str | Path, project_root: str | Path | None = None) -> SignerConfig:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    config = SignerConfig.from_dict(payload)
    if project_root is not None:
        config.data = config.data.resolve(project_root)
        config.model = config.model.resolve(project_root)
        config.training = config.training.resolve(project_root)
    return config
