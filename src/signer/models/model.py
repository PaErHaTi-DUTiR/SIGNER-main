"""Main SIGNER token classification model."""

from __future__ import annotations

import torch.nn as nn
import torch
import torch.nn.functional as F

from signer.config import ModelConfig
from signer.constants import TAG_TO_ID, VIEW_NAME_TO_ID
from signer.models.crf import LinearChainCRF
from signer.models.fusion import build_fusion
from signer.models.speech_encoder import build_speech_encoder
from signer.models.text_branch import XLMRTextBranch
from signer.training.losses import masked_cross_entropy_loss, masked_focal_loss


class SignerTokenClassifier(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.use_speech_branch = config.use_speech_branch
        self.text_branch = XLMRTextBranch(
            model_name_or_path=config.text_model_name_or_path,
            output_dim=config.hidden_size,
            freeze_encoder=config.freeze_text_encoder,
        )
        if self.use_speech_branch:
            self.speech_encoder = build_speech_encoder(
                speech_encoder_type=config.speech_encoder_type,
                model_name_or_path=config.whisper_name_or_path,
                output_dim=config.hidden_size,
                freeze_encoder=config.freeze_speech_encoder,
                unfreeze_top_n_layers=config.whisper_unfreeze_top_n_layers,
            ) if config.load_speech_encoder else None
            self.fusion = build_fusion(
                fusion_type=config.fusion_type,
                hidden_size=config.hidden_size,
                dropout=config.dropout,
                alignment_window_size=config.alignment_window_size,
            )
        else:
            self.speech_encoder = None
            self.fusion = None
        self.modality_embedding = nn.Embedding(3, config.hidden_size)
        self.share_modality_embedding = config.share_modality_embedding
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.decoder_type = config.decoder_type
        self.loss_type = config.loss_type
        self.label_smoothing = config.label_smoothing
        self.focal_gamma = config.focal_gamma
        self.use_entity_prior = config.use_entity_prior
        self.entity_prior_loss_weight = config.entity_prior_loss_weight
        self.entity_prior_threshold = config.entity_prior_threshold
        self.crf = LinearChainCRF(config.num_labels) if config.decoder_type == "crf" else None
        self.entity_prior = nn.Linear(config.hidden_size, 1) if config.use_entity_prior else None

    def _text_uncertainty_features(self, text_states: torch.Tensor, batch: dict) -> torch.Tensor:
        text_view = torch.full_like(batch["view_type"], VIEW_NAME_TO_ID["text-only"])
        if self.share_modality_embedding:
            text_view = torch.zeros_like(text_view)
        text_only_states = text_states + self.modality_embedding(text_view).unsqueeze(1)
        with torch.no_grad():
            logits = self.classifier(text_only_states)
            probs = torch.softmax(logits, dim=-1)
            top2 = torch.topk(probs, k=2, dim=-1).values
            top1 = top2[..., 0]
            margin = top2[..., 0] - top2[..., 1]
            entropy = -(probs.clamp_min(1e-8) * probs.clamp_min(1e-8).log()).sum(dim=-1)
            entropy = entropy / torch.log(torch.tensor(float(probs.size(-1)), device=probs.device))
            entity_confidence = 1.0 - probs[..., TAG_TO_ID["O"]]
            low_confidence = 1.0 - top1
            features = torch.stack(
                [
                    entropy.clamp(0.0, 1.0),
                    (1.0 - margin).clamp(0.0, 1.0),
                    entity_confidence.clamp(0.0, 1.0),
                    low_confidence.clamp(0.0, 1.0),
                ],
                dim=-1,
            )
        return features

    def forward(self, batch: dict, class_weights=None) -> dict:
        text_states = self.text_branch(
            input_ids=batch["text_input_ids"],
            attention_mask=batch["text_attention_mask"].long(),
        )
        if self.use_speech_branch:
            speech_mask = batch["audio_mask"]
            if not speech_mask.any():
                fused_states = text_states
            else:
                if "speech_states" in batch:
                    speech_states = batch["speech_states"]
                else:
                    if self.speech_encoder is None:
                        raise ValueError("speech_states cache is required when load_speech_encoder is false")
                    speech_states = self.speech_encoder(
                        audio_features=batch["audio_features"],
                        audio_attention_mask=speech_mask.long(),
                    )
                if speech_mask.size(1) != speech_states.size(1):
                    speech_mask = speech_mask[:, : speech_states.size(1)]
                    if speech_mask.size(1) < speech_states.size(1):
                        pad = speech_states.size(1) - speech_mask.size(1)
                        speech_mask = nn.functional.pad(speech_mask, (0, pad), value=False)
                fusion_kwargs = {"text_mask": batch["text_attention_mask"]}
                if getattr(self.fusion, "uses_text_uncertainty", False):
                    fusion_kwargs["text_uncertainty"] = self._text_uncertainty_features(text_states, batch)
                fused_states = self.fusion(text_states, speech_states, speech_mask, **fusion_kwargs)
        else:
            fused_states = text_states
        view_type = batch["view_type"]
        if self.share_modality_embedding:
            view_type = torch.zeros_like(view_type)
        fused_states = fused_states + self.modality_embedding(view_type).unsqueeze(1)
        logits = self.classifier(self.dropout(fused_states))
        output = {"logits": logits}
        if self.entity_prior is not None:
            mask = batch["text_attention_mask"].float()
            pooled = (fused_states * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            output["entity_prior_logit"] = self.entity_prior(self.dropout(pooled)).squeeze(-1)
        if "token_labels" in batch:
            labels = batch["token_labels"]
            label_mask = batch["label_mask"]
            if self.crf is not None:
                safe_labels = labels.masked_fill(~label_mask, TAG_TO_ID["O"])
                loss = self.crf.neg_log_likelihood(logits, safe_labels, label_mask)
            elif self.loss_type == "focal":
                loss = masked_focal_loss(
                    logits=logits,
                    labels=labels,
                    label_mask=label_mask,
                    class_weights=class_weights,
                    gamma=self.focal_gamma,
                    label_smoothing=self.label_smoothing,
                )
            else:
                loss = masked_cross_entropy_loss(
                    logits=logits,
                    labels=labels,
                    label_mask=label_mask,
                    class_weights=class_weights,
                    label_smoothing=self.label_smoothing,
                )
            if self.entity_prior is not None and self.entity_prior_loss_weight > 0.0:
                has_entity = ((labels != TAG_TO_ID["O"]) & label_mask).any(dim=1).float()
                prior_loss = F.binary_cross_entropy_with_logits(output["entity_prior_logit"], has_entity)
                loss = loss + self.entity_prior_loss_weight * prior_loss
                output["entity_prior_loss"] = prior_loss
            output["loss"] = loss
        return output

    def decode(self, batch: dict) -> list[list[int]]:
        output = self.forward(batch)
        logits = output["logits"]
        mask = batch.get("label_mask", batch["text_attention_mask"]).bool()
        if self.crf is not None:
            predictions = self.crf.decode(logits, mask)
        else:
            predictions = logits.argmax(dim=-1).tolist()
        if self.entity_prior is not None and self.entity_prior_threshold > 0.0:
            probs = torch.sigmoid(output["entity_prior_logit"]).detach().cpu().tolist()
            for index, prob in enumerate(probs):
                if prob < self.entity_prior_threshold:
                    predictions[index] = [TAG_TO_ID["O"] if valid else TAG_TO_ID["O"] for valid in mask[index].cpu().tolist()]
        return predictions
