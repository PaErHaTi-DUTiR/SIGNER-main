"""
AnyMod-SIGNER: Three-mode unified speech-text NER model.

Training views:
  ST (speech+text) → TARF fusion → NER  +  KL(ST‖T) + KL(ST‖S)
  T  (text only)   → XLM-R       → NER  +  KL(T‖ST)
  S  (speech word) → S-head      → NER  +  KL(S‖ST)

Inference modes:
  T-NER  : forward_text(H_t)               ← zero audio dependency
  S-NER  : forward_speech(H_s_word)        ← audio only
  ST-NER : forward(H_t, H_s_frames)        ← joint, same as SIGNER[ST]

The cross-view KL losses implement mutual enhancement:
  • ST teaches T and S → T-NER and S-NER benefit from acoustic fusion
  • S teaches ST via reverse KL → forces TARF to be speech-aware beyond text
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from .fusion import build_fusion


NUM_LABELS = 9  # O B-PER I-PER B-LOC I-LOC B-ORG I-ORG B-MISC I-MISC
PAD_LABEL_ID = -100


class SpeechWordHead(nn.Module):
    """Project word-aligned Whisper features [n, d_speech] to NER logits [n, L]."""

    def __init__(self, d_speech: int = 768, d_model: int = 768,
                 num_labels: int = NUM_LABELS, dropout: float = 0.1) -> None:
        super().__init__()
        self.proj = nn.Linear(d_speech, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_labels)

    def forward(self, H_s_word: torch.Tensor) -> torch.Tensor:
        """H_s_word: [B, n, d_speech] → logits [B, n, num_labels]."""
        x = self.drop(self.norm(F.gelu(self.proj(H_s_word))))
        return self.classifier(x)


class AnyModSIGNER(nn.Module):
    """
    Three-mode SIGNER: T-NER / S-NER / ST-NER.

    Parameters
    ----------
    text_model_name_or_path : str
    hidden_size : int
    num_labels : int
    fusion_type : str   (passed to build_fusion; TARF by default)
    dropout : float
    """

    def __init__(
        self,
        text_model_name_or_path: str,
        hidden_size: int = 768,
        num_labels: int = NUM_LABELS,
        fusion_type: str = "tarf",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.text_encoder = AutoModel.from_pretrained(text_model_name_or_path)
        self.fusion = build_fusion(
            fusion_type=fusion_type, hidden_size=hidden_size,
            dropout=dropout, alignment_window_size=0,
        )
        self.speech_word_head = SpeechWordHead(
            d_speech=hidden_size, d_model=hidden_size,
            num_labels=num_labels, dropout=dropout,
        )
        self.text_classifier = nn.Linear(hidden_size, num_labels)
        self.drop = nn.Dropout(dropout)
        self.num_labels = num_labels

    # ── forward helpers ──────────────────────────────────────────────────────

    def encode_text(self, input_ids, attention_mask):
        out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state  # [B, n, d]

    def forward_text(self, H_t: torch.Tensor) -> torch.Tensor:
        """T-NER: text states → logits.  No audio."""
        return self.text_classifier(self.drop(H_t))

    def forward_speech(
        self,
        H_s_word: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """S-NER: word-aligned Whisper features → logits.
        valid_mask [B, n] bool: False positions are set to zero logits."""
        logits = self.speech_word_head(H_s_word)
        if valid_mask is not None:
            logits = logits * valid_mask.unsqueeze(-1).float()
        return logits

    def forward_joint(
        self,
        H_t: torch.Tensor,
        H_s_frames: torch.Tensor,
        speech_mask: torch.Tensor,
    ) -> torch.Tensor:
        """ST-NER: TARF fusion → logits."""
        H_fused = self.fusion(H_t, H_s_frames, speech_mask)
        return self.text_classifier(self.drop(H_fused))

    # ── training forward ─────────────────────────────────────────────────────

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        word_ids: torch.Tensor | None = None,        # [B, seq_len] int, -1=special
        # ST branch
        speech_frames: torch.Tensor | None = None,
        speech_mask: torch.Tensor | None = None,
        # S branch (word-aligned features)
        speech_word_feats: torch.Tensor | None = None,
        speech_word_valid: torch.Tensor | None = None,
        # Loss weights
        kl_weight: float = 0.5,
        s_weight: float = 0.3,
        aux_t_weight: float = 0.3,
        non_o_weight: float = 3.0,
    ) -> dict[str, torch.Tensor]:
        """
        Multi-view training forward.  Returns dict with 'loss' and per-view logits.
        """
        # Build per-class loss weights
        class_weights = torch.ones(self.num_labels, device=input_ids.device)
        class_weights[1:] = non_o_weight  # all non-O tags

        # ── Text encoding (shared) ───────────────────────────────────────────
        H_t = self.encode_text(input_ids, attention_mask)

        # ── T-NER view ───────────────────────────────────────────────────────
        logits_t = self.forward_text(H_t)
        loss_t = F.cross_entropy(
            logits_t.view(-1, self.num_labels),
            labels.view(-1),
            weight=class_weights,
            ignore_index=PAD_LABEL_ID,
        )

        losses = {"loss_t": loss_t}
        logits_dict = {"logits_t": logits_t}
        total_loss = loss_t

        # ── ST-NER view (if speech frames provided) ──────────────────────────
        if speech_frames is not None and speech_mask is not None:
            logits_st = self.forward_joint(H_t, speech_frames, speech_mask)
            loss_st = F.cross_entropy(
                logits_st.view(-1, self.num_labels),
                labels.view(-1),
                weight=class_weights,
                ignore_index=PAD_LABEL_ID,
            )
            losses["loss_st"] = loss_st

            # KL: ST teaches T
            kl_st_t = _sym_kl(logits_st.detach(), logits_t, labels)
            losses["kl_st_t"] = kl_st_t
            total_loss = total_loss + loss_st + kl_weight * kl_st_t
            logits_dict["logits_st"] = logits_st
        else:
            logits_st = None

        # ── S-NER view (if word-aligned feats provided) ──────────────────────
        if speech_word_feats is not None:
            logits_s_word = self.forward_speech(speech_word_feats, speech_word_valid)
            # Expand word-level logits/valid → token-level using word_ids
            logits_s, tok_valid = _expand_word_to_token(
                logits_s_word, speech_word_valid, word_ids, labels.shape
            )
            loss_s = _masked_ce(logits_s, labels, tok_valid, class_weights)
            losses["loss_s"] = loss_s

            # KL: ST (or T) teaches S
            teacher = logits_st.detach() if logits_st is not None else logits_t.detach()
            kl_st_s = _sym_kl(teacher, logits_s, labels)
            losses["kl_st_s"] = kl_st_s

            total_loss = total_loss + s_weight * loss_s + kl_weight * kl_st_s
            logits_dict["logits_s"] = logits_s

        losses["loss"] = total_loss
        return {**losses, **logits_dict}


# ── helpers ──────────────────────────────────────────────────────────────────

def _expand_word_to_token(
    logits_word: torch.Tensor,          # [B, n_words, L]
    valid_word: torch.Tensor | None,    # [B, n_words] bool
    word_ids: torch.Tensor | None,      # [B, seq_len] int, -1=special
    target_shape: tuple,                # (B, seq_len)
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Broadcast word-level logits/valid to token-level."""
    B, seq_len = target_shape
    L = logits_word.shape[-1]
    device = logits_word.device

    logits_tok = torch.zeros(B, seq_len, L, device=device)
    valid_tok = torch.zeros(B, seq_len, dtype=torch.bool, device=device)

    for b in range(B):
        for j in range(seq_len):
            if word_ids is None:
                wid = j
            else:
                wid = word_ids[b, j].item()
            if wid < 0 or wid >= logits_word.shape[1]:
                continue
            logits_tok[b, j] = logits_word[b, wid]
            if valid_word is not None:
                valid_tok[b, j] = valid_word[b, wid]
            else:
                valid_tok[b, j] = True

    return logits_tok, valid_tok


def _sym_kl(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Symmetric KL on valid (non-PAD) positions."""
    valid = labels.view(-1) != PAD_LABEL_ID
    if valid.sum() == 0:
        return logits_a.new_tensor(0.0)
    p = F.log_softmax(logits_a.view(-1, logits_a.size(-1))[valid], dim=-1)
    q = F.log_softmax(logits_b.view(-1, logits_b.size(-1))[valid], dim=-1)
    kl_pq = F.kl_div(q, p.exp(), reduction="batchmean")
    kl_qp = F.kl_div(p, q.exp(), reduction="batchmean")
    return 0.5 * (kl_pq + kl_qp)


def _masked_ce(
    logits: torch.Tensor,
    labels: torch.Tensor,
    valid_mask: torch.Tensor | None,
    class_weights: torch.Tensor,
) -> torch.Tensor:
    """CE loss; also masks positions where valid_mask is False."""
    effective_labels = labels.clone()
    if valid_mask is not None:
        effective_labels[~valid_mask] = PAD_LABEL_ID
    return F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        effective_labels.view(-1),
        weight=class_weights,
        ignore_index=PAD_LABEL_ID,
    )
