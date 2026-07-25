"""Speech encoder wrapper for SIGNER."""

from __future__ import annotations

import torch.nn as nn
import torch
from transformers import AutoConfig, Wav2Vec2ForCTC, Wav2Vec2Model, WhisperModel


class WhisperSpeechEncoder(nn.Module):
    def __init__(
        self,
        model_name_or_path: str,
        output_dim: int,
        freeze_encoder: bool = False,
        unfreeze_top_n_layers: int = 0,
    ) -> None:
        super().__init__()
        whisper_model = WhisperModel.from_pretrained(model_name_or_path)
        self.encoder = whisper_model.encoder
        hidden_size = whisper_model.config.d_model
        self.projection = nn.Identity() if hidden_size == output_dim else nn.Linear(hidden_size, output_dim)

        # Default: freeze everything, then selectively unfreeze top N transformer layers.
        # unfreeze_top_n_layers=0 and freeze_encoder=False => fully trainable (original behavior).
        if freeze_encoder:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False
            if unfreeze_top_n_layers > 0 and hasattr(self.encoder, "layers"):
                total = len(self.encoder.layers)
                for layer in self.encoder.layers[max(0, total - unfreeze_top_n_layers):]:
                    for parameter in layer.parameters():
                        parameter.requires_grad = True
                # Also unfreeze the final layer norm so unfrozen layers feed useful gradients.
                if hasattr(self.encoder, "layer_norm"):
                    for parameter in self.encoder.layer_norm.parameters():
                        parameter.requires_grad = True

        self._fully_frozen = freeze_encoder and (unfreeze_top_n_layers == 0)

    def forward(self, audio_features, audio_attention_mask=None):
        encoder_inputs = audio_features.transpose(1, 2)
        if self._fully_frozen:
            self.encoder.eval()
            with torch.no_grad():
                outputs = self.encoder(input_features=encoder_inputs)
        else:
            outputs = self.encoder(input_features=encoder_inputs)
        return self.projection(outputs.last_hidden_state)


class Wav2Vec2SpeechEncoder(nn.Module):
    def __init__(self, model_name_or_path: str, output_dim: int, freeze_encoder: bool = False) -> None:
        super().__init__()
        config = AutoConfig.from_pretrained(model_name_or_path)
        if getattr(config, "architectures", None) and "Wav2Vec2ForCTC" in config.architectures:
            self.encoder = Wav2Vec2ForCTC.from_pretrained(model_name_or_path).wav2vec2
        else:
            self.encoder = Wav2Vec2Model.from_pretrained(model_name_or_path)
        self.freeze_encoder = freeze_encoder
        hidden_size = self.encoder.config.hidden_size
        self.projection = nn.Identity() if hidden_size == output_dim else nn.Linear(hidden_size, output_dim)
        if freeze_encoder:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False

    def forward(self, audio_features, audio_attention_mask=None):
        input_values = audio_features.squeeze(-1)
        if self.freeze_encoder:
            self.encoder.eval()
            with torch.no_grad():
                outputs = self.encoder(input_values=input_values, attention_mask=audio_attention_mask)
        else:
            outputs = self.encoder(input_values=input_values, attention_mask=audio_attention_mask)
        return self.projection(outputs.last_hidden_state)


def build_speech_encoder(
    *,
    speech_encoder_type: str,
    model_name_or_path: str,
    output_dim: int,
    freeze_encoder: bool = False,
    unfreeze_top_n_layers: int = 0,
) -> nn.Module:
    encoder_type = speech_encoder_type.lower()
    if encoder_type == "whisper":
        return WhisperSpeechEncoder(model_name_or_path, output_dim, freeze_encoder, unfreeze_top_n_layers)
    if encoder_type in {"wav2vec2", "mms"}:
        return Wav2Vec2SpeechEncoder(model_name_or_path, output_dim, freeze_encoder)
    raise ValueError(f"unsupported speech_encoder_type: {speech_encoder_type}")
