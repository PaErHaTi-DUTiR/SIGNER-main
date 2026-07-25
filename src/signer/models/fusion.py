"""Cross-modal fusion layers for SIGNER."""

from __future__ import annotations

import torch
import torch.nn as nn


class GatedCrossModalFusion(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.gate = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text_states, speech_states, speech_mask, text_mask=None):
        attended = torch.zeros_like(text_states)
        if speech_states.size(1) > 0:
            has_speech = speech_mask.any(dim=1)
            if has_speech.any():
                selected = has_speech.nonzero(as_tuple=False).squeeze(-1)
                attention_output, _ = self.cross_attention(
                    query=text_states[selected],
                    key=speech_states[selected],
                    value=speech_states[selected],
                    key_padding_mask=~speech_mask[selected],
                )
                attended[selected] = attention_output
        gate = torch.sigmoid(self.gate(torch.cat([text_states, attended], dim=-1)))
        fused = gate * text_states + (1.0 - gate) * attended
        return self.dropout(fused)


class TextAnchoredResidualFusion(nn.Module):
    """Speech-aware residual fusion anchored to text token states.

    NER boundaries are decoded on text tokens, so speech should act as
    supporting evidence instead of replacing the text representation.
    """

    def __init__(self, hidden_size: int, num_heads: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.speech_gate = nn.Linear(hidden_size * 3, hidden_size)
        self.residual_norm = nn.LayerNorm(hidden_size)
        self.output_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        nn.init.constant_(self.speech_gate.bias, -2.0)

    def forward(self, text_states, speech_states, speech_mask, text_mask=None):
        attended = torch.zeros_like(text_states)
        if speech_states.size(1) > 0:
            has_speech = speech_mask.any(dim=1)
            if has_speech.any():
                selected = has_speech.nonzero(as_tuple=False).squeeze(-1)
                attention_output, _ = self.cross_attention(
                    query=text_states[selected],
                    key=speech_states[selected],
                    value=speech_states[selected],
                    key_padding_mask=~speech_mask[selected],
                )
                attended[selected] = attention_output
        residual = self.residual_norm(attended - text_states)
        gate = torch.sigmoid(self.speech_gate(torch.cat([text_states, attended, residual], dim=-1)))
        fused = text_states + self.dropout(gate * residual)
        valid_text = text_mask.bool() if text_mask is not None else torch.ones(
            text_states.shape[:2], dtype=torch.bool, device=text_states.device
        )
        denom = valid_text.sum(dim=1).clamp_min(1).float()
        self.last_diagnostics = {
            "gate_mean": (gate.mean(dim=-1) * valid_text.float()).sum(dim=1).detach() / denom,
            "gate_max": gate.mean(dim=-1).masked_fill(~valid_text, 0.0).max(dim=1).values.detach(),
            "residual_norm_mean": (residual.norm(dim=-1) * valid_text.float()).sum(dim=1).detach() / denom,
            "residual_norm_max": residual.norm(dim=-1).masked_fill(~valid_text, 0.0).max(dim=1).values.detach(),
        }
        return self.output_norm(fused)


class ReliabilityGatedTextAnchoredFusion(nn.Module):
    """Text-anchored fusion with an explicit speech reliability gate."""

    def __init__(self, hidden_size: int, num_heads: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.token_gate = nn.Linear(hidden_size * 3, hidden_size)
        self.reliability_gate = nn.Sequential(
            nn.Linear(hidden_size * 4 + 1, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )
        self.residual_norm = nn.LayerNorm(hidden_size)
        self.output_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        nn.init.constant_(self.token_gate.bias, -2.0)
        nn.init.constant_(self.reliability_gate[-1].bias, -1.5)

    def forward(self, text_states, speech_states, speech_mask, text_mask=None):
        attended = torch.zeros_like(text_states)
        if speech_states.size(1) > 0:
            has_speech = speech_mask.any(dim=1)
            if has_speech.any():
                selected = has_speech.nonzero(as_tuple=False).squeeze(-1)
                attention_output, _ = self.cross_attention(
                    query=text_states[selected],
                    key=speech_states[selected],
                    value=speech_states[selected],
                    key_padding_mask=~speech_mask[selected],
                )
                attended[selected] = attention_output

        residual = self.residual_norm(attended - text_states)
        token_gate = torch.sigmoid(self.token_gate(torch.cat([text_states, attended, residual], dim=-1)))

        valid_text = text_mask.bool() if text_mask is not None else torch.ones(
            text_states.shape[:2], dtype=torch.bool, device=text_states.device
        )
        text_den = valid_text.sum(dim=1, keepdim=True).clamp_min(1).float()
        text_pool = (text_states * valid_text.unsqueeze(-1)).sum(dim=1) / text_den
        attended_pool = (attended * valid_text.unsqueeze(-1)).sum(dim=1) / text_den
        residual_pool = (residual * valid_text.unsqueeze(-1)).sum(dim=1) / text_den
        residual_abs_pool = (residual.abs() * valid_text.unsqueeze(-1)).sum(dim=1) / text_den
        speech_ratio = speech_mask.float().mean(dim=1, keepdim=True)
        reliability = torch.sigmoid(
            self.reliability_gate(
                torch.cat([text_pool, attended_pool, residual_pool, residual_abs_pool, speech_ratio], dim=-1)
            )
        ).unsqueeze(1)

        fused = text_states + self.dropout(reliability * token_gate * residual)
        return self.output_norm(fused)


class PrecisionControlledTextAnchoredFusion(nn.Module):
    """Text-anchored residual fusion with conservative speech reliability.

    Speech is only allowed to enter through a low-prior token reliability gate.
    The gate receives both learned residual evidence and attention concentration:
    diffuse speech attention usually means the acoustic evidence is unreliable
    for a specific NER token, so the residual should be suppressed.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        *,
        use_attention_reliability: bool = True,
        gate_bias: float = -2.5,
    ) -> None:
        super().__init__()
        self.use_attention_reliability = use_attention_reliability
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.token_gate = nn.Linear(hidden_size * 3, hidden_size)
        self.reliability_gate = nn.Sequential(
            nn.Linear(hidden_size * 4 + 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )
        self.residual_norm = nn.LayerNorm(hidden_size)
        self.output_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        nn.init.constant_(self.token_gate.bias, gate_bias)
        nn.init.constant_(self.reliability_gate[-1].bias, gate_bias)

    def forward(self, text_states, speech_states, speech_mask, text_mask=None):
        batch_size, text_len, _ = text_states.shape
        attended = torch.zeros_like(text_states)
        peak = torch.zeros(batch_size, text_len, 1, dtype=text_states.dtype, device=text_states.device)
        concentration = torch.zeros_like(peak)
        if speech_states.size(1) > 0:
            has_speech = speech_mask.any(dim=1)
            if has_speech.any():
                selected = has_speech.nonzero(as_tuple=False).squeeze(-1)
                attention_output, attention_weights = self.cross_attention(
                    query=text_states[selected],
                    key=speech_states[selected],
                    value=speech_states[selected],
                    key_padding_mask=~speech_mask[selected],
                    need_weights=True,
                    average_attn_weights=True,
                )
                attended[selected] = attention_output
                selected_weights = attention_weights.clamp_min(1e-8)
                selected_peak = selected_weights.max(dim=-1, keepdim=True).values
                entropy = -(selected_weights * selected_weights.log()).sum(dim=-1, keepdim=True)
                valid_speech = speech_mask[selected].sum(dim=1).clamp_min(2).float().log().view(-1, 1, 1)
                selected_concentration = (1.0 - entropy / valid_speech).clamp(0.0, 1.0)
                peak[selected] = selected_peak
                concentration[selected] = selected_concentration
        if not self.use_attention_reliability:
            peak = torch.zeros_like(peak)
            concentration = torch.zeros_like(concentration)

        residual = self.residual_norm(attended - text_states)
        token_gate = torch.sigmoid(self.token_gate(torch.cat([text_states, attended, residual], dim=-1)))
        reliability = torch.sigmoid(
            self.reliability_gate(
                torch.cat([text_states, attended, residual, residual.abs(), peak, concentration], dim=-1)
            )
        )
        fused = text_states + self.dropout(reliability * token_gate * residual)
        valid_text = text_mask.bool() if text_mask is not None else torch.ones(
            text_states.shape[:2], dtype=torch.bool, device=text_states.device
        )
        denom = valid_text.sum(dim=1).clamp_min(1).float()
        gate_strength = reliability.squeeze(-1) * token_gate.mean(dim=-1)
        self.last_diagnostics = {
            "gate_mean": (gate_strength * valid_text.float()).sum(dim=1).detach() / denom,
            "gate_max": gate_strength.masked_fill(~valid_text, 0.0).max(dim=1).values.detach(),
            "reliability_mean": (reliability.squeeze(-1) * valid_text.float()).sum(dim=1).detach() / denom,
            "attention_peak_mean": (peak.squeeze(-1) * valid_text.float()).sum(dim=1).detach() / denom,
            "attention_concentration_mean": (concentration.squeeze(-1) * valid_text.float()).sum(dim=1).detach() / denom,
            "residual_norm_mean": (residual.norm(dim=-1) * valid_text.float()).sum(dim=1).detach() / denom,
            "residual_norm_max": residual.norm(dim=-1).masked_fill(~valid_text, 0.0).max(dim=1).values.detach(),
        }
        return self.output_norm(fused)


class TextUncertaintyGuidedSpeechInterventionFusion(nn.Module):
    """SIGNER V3: text uncertainty decides when speech may intervene.

    The text stream remains the anchor. Speech is injected as a residual only
    when the text-only branch is uncertain about a token/span-like position.
    """

    uses_text_uncertainty = True

    def __init__(self, hidden_size: int, num_heads: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.token_gate = nn.Linear(hidden_size * 3, hidden_size)
        self.reliability_gate = nn.Sequential(
            nn.Linear(hidden_size * 4 + 4, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )
        self.residual_norm = nn.LayerNorm(hidden_size)
        self.output_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        nn.init.constant_(self.token_gate.bias, -2.5)
        nn.init.constant_(self.reliability_gate[-1].bias, -2.5)

    def forward(self, text_states, speech_states, speech_mask, text_mask=None, text_uncertainty=None):
        batch_size, text_len, _ = text_states.shape
        attended = torch.zeros_like(text_states)
        if speech_states.size(1) > 0:
            has_speech = speech_mask.any(dim=1)
            if has_speech.any():
                selected = has_speech.nonzero(as_tuple=False).squeeze(-1)
                attention_output, _ = self.cross_attention(
                    query=text_states[selected],
                    key=speech_states[selected],
                    value=speech_states[selected],
                    key_padding_mask=~speech_mask[selected],
                )
                attended[selected] = attention_output

        if text_uncertainty is None:
            text_uncertainty = torch.zeros(batch_size, text_len, 4, dtype=text_states.dtype, device=text_states.device)
        residual = self.residual_norm(attended - text_states)
        token_gate = torch.sigmoid(self.token_gate(torch.cat([text_states, attended, residual], dim=-1)))
        reliability = torch.sigmoid(
            self.reliability_gate(
                torch.cat([text_states, attended, residual, residual.abs(), text_uncertainty], dim=-1)
            )
        )
        entropy = text_uncertainty[..., 0:1].clamp(0.0, 1.0)
        inv_margin = text_uncertainty[..., 1:2].clamp(0.0, 1.0)
        uncertainty_prior = torch.sigmoid(4.0 * (((entropy + inv_margin) * 0.5) - 0.5))
        intervention = reliability * (0.25 + 0.75 * uncertainty_prior)
        fused = text_states + self.dropout(intervention * token_gate * residual)

        valid_text = text_mask.bool() if text_mask is not None else torch.ones(
            text_states.shape[:2], dtype=torch.bool, device=text_states.device
        )
        denom = valid_text.sum(dim=1).clamp_min(1).float()
        gate_strength = intervention.squeeze(-1) * token_gate.mean(dim=-1)
        self.last_diagnostics = {
            "gate_mean": (gate_strength * valid_text.float()).sum(dim=1).detach() / denom,
            "gate_max": gate_strength.masked_fill(~valid_text, 0.0).max(dim=1).values.detach(),
            "reliability_mean": (reliability.squeeze(-1) * valid_text.float()).sum(dim=1).detach() / denom,
            "text_entropy_mean": (entropy.squeeze(-1) * valid_text.float()).sum(dim=1).detach() / denom,
            "text_inv_margin_mean": (inv_margin.squeeze(-1) * valid_text.float()).sum(dim=1).detach() / denom,
            "intervention_prior_mean": (uncertainty_prior.squeeze(-1) * valid_text.float()).sum(dim=1).detach() / denom,
            "residual_norm_mean": (residual.norm(dim=-1) * valid_text.float()).sum(dim=1).detach() / denom,
            "residual_norm_max": residual.norm(dim=-1).masked_fill(~valid_text, 0.0).max(dim=1).values.detach(),
        }
        return self.output_norm(fused)


class SparseTopKTextAnchoredFusion(nn.Module):
    """ACT-S inspired sparse cross-channel fusion.

    Each text token attends only to its top-k speech frames, then speech is
    added as a conservative residual to preserve text-token NER stability.
    """

    def __init__(
        self,
        hidden_size: int,
        topk: int = 32,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.topk = max(int(topk), 1)
        self.query = nn.Linear(hidden_size, hidden_size, bias=False)
        self.key = nn.Linear(hidden_size, hidden_size, bias=False)
        self.value = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out = nn.Linear(hidden_size, hidden_size)
        self.speech_gate = nn.Linear(hidden_size * 3, hidden_size)
        self.residual_norm = nn.LayerNorm(hidden_size)
        self.output_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        nn.init.constant_(self.speech_gate.bias, -2.0)

    def _split_heads(self, states):
        batch_size, seq_len, hidden_size = states.shape
        states = states.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return states.permute(0, 2, 1, 3)

    def _merge_heads(self, states):
        batch_size, _, seq_len, _ = states.shape
        states = states.permute(0, 2, 1, 3).contiguous()
        return states.view(batch_size, seq_len, self.num_heads * self.head_dim)

    def forward(self, text_states, speech_states, speech_mask, text_mask=None):
        if speech_states.size(1) == 0 or not speech_mask.any():
            return text_states

        q = self._split_heads(self.query(text_states))
        k = self._split_heads(self.key(speech_states))
        v = self._split_heads(self.value(speech_states))
        scores = torch.matmul(q, k.transpose(-1, -2)) / (self.head_dim ** 0.5)
        scores = scores.masked_fill(~speech_mask[:, None, None, :], -10000.0)

        valid_speech_len = int(speech_mask.sum(dim=1).max().item())
        topk = min(self.topk, max(valid_speech_len, 1), scores.size(-1))
        if topk < scores.size(-1):
            threshold = torch.topk(scores, k=topk, dim=-1).values[..., -1:]
            scores = scores.masked_fill(scores < threshold, -10000.0)

        probs = torch.softmax(scores, dim=-1)
        probs = self.dropout(probs)
        attended = self.out(self._merge_heads(torch.matmul(probs, v)))
        residual = self.residual_norm(attended - text_states)
        gate = torch.sigmoid(self.speech_gate(torch.cat([text_states, attended, residual], dim=-1)))
        fused = text_states + self.dropout(gate * residual)
        return self.output_norm(fused)


class TokenAlignedCrossModalFusion(nn.Module):
    """Proportional token-to-speech local attention.

    This is a deterministic alignment prior: token index positions are mapped
    proportionally onto the speech frame sequence, then each token attends only
    to a local speech window. It is not forced alignment, but it is stricter
    than global attention and gives SIGNER v3 a token-level fusion baseline.
    """

    def __init__(self, hidden_size: int, window_size: int = 16, num_heads: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        self.window_size = max(int(window_size), 1)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.gate = nn.Linear(hidden_size * 3, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text_states, speech_states, speech_mask, text_mask=None):
        batch_size, text_len, hidden = text_states.shape
        attended = torch.zeros_like(text_states)
        if speech_states.size(1) == 0:
            return text_states

        valid_text = text_mask.bool() if text_mask is not None else torch.ones(
            batch_size, text_len, dtype=torch.bool, device=text_states.device
        )
        half = max(self.window_size // 2, 1)
        for batch_index in range(batch_size):
            speech_valid_len = int(speech_mask[batch_index].sum().item())
            text_valid_len = int(valid_text[batch_index].sum().item())
            if speech_valid_len <= 0 or text_valid_len <= 0:
                continue
            for token_index in range(text_valid_len):
                center = int(round(token_index * max(speech_valid_len - 1, 1) / max(text_valid_len - 1, 1)))
                left = max(center - half, 0)
                right = min(center + half + 1, speech_valid_len)
                query = text_states[batch_index : batch_index + 1, token_index : token_index + 1]
                key_value = speech_states[batch_index : batch_index + 1, left:right]
                context, _ = self.cross_attention(query=query, key=key_value, value=key_value)
                attended[batch_index, token_index] = context[0, 0]

        gate = torch.sigmoid(self.gate(torch.cat([text_states, attended, text_states - attended], dim=-1)))
        fused = gate * text_states + (1.0 - gate) * attended
        return self.dropout(self.norm(fused))


class TokenAlignedTextAnchoredFusion(nn.Module):
    """Local token-to-speech attention with text-preserving residual output."""

    def __init__(self, hidden_size: int, window_size: int = 16, num_heads: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        self.window_size = max(int(window_size), 1)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.speech_gate = nn.Linear(hidden_size * 3, hidden_size)
        self.residual_norm = nn.LayerNorm(hidden_size)
        self.output_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        nn.init.constant_(self.speech_gate.bias, -2.0)

    def forward(self, text_states, speech_states, speech_mask, text_mask=None):
        batch_size, text_len, _ = text_states.shape
        attended = torch.zeros_like(text_states)
        if speech_states.size(1) == 0:
            return text_states

        valid_text = text_mask.bool() if text_mask is not None else torch.ones(
            batch_size, text_len, dtype=torch.bool, device=text_states.device
        )
        half = max(self.window_size // 2, 1)
        for batch_index in range(batch_size):
            speech_valid_len = int(speech_mask[batch_index].sum().item())
            text_valid_len = int(valid_text[batch_index].sum().item())
            if speech_valid_len <= 0 or text_valid_len <= 0:
                continue
            for token_index in range(text_valid_len):
                center = int(round(token_index * max(speech_valid_len - 1, 1) / max(text_valid_len - 1, 1)))
                left = max(center - half, 0)
                right = min(center + half + 1, speech_valid_len)
                query = text_states[batch_index : batch_index + 1, token_index : token_index + 1]
                key_value = speech_states[batch_index : batch_index + 1, left:right]
                context, _ = self.cross_attention(query=query, key=key_value, value=key_value)
                attended[batch_index, token_index] = context[0, 0]

        residual = self.residual_norm(attended - text_states)
        gate = torch.sigmoid(self.speech_gate(torch.cat([text_states, attended, residual], dim=-1)))
        fused = text_states + self.dropout(gate * residual)
        return self.output_norm(fused)


class MeanPoolSpeechFusion(nn.Module):
    """Naive mean-pool speech baseline: no cross-attention, all frames weighted equally.

    Aggregates speech frames via masked mean-pooling into a single utterance
    vector, then injects it at every text token position through a learned
    interpolation gate.  Serves as the simplest non-trivial speech fusion
    ablation, requiring no frame-level attention mechanism.
    """

    def __init__(self, hidden_size: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.gate = nn.Linear(hidden_size * 2, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text_states, speech_states, speech_mask, text_mask=None):
        if speech_states.size(1) == 0 or not speech_mask.any():
            return text_states
        mask_f = speech_mask.unsqueeze(-1).float()
        speech_mean = (speech_states * mask_f).sum(1) / mask_f.sum(1).clamp_min(1.0)
        speech_mean = speech_mean.unsqueeze(1).expand_as(text_states)
        gate = torch.sigmoid(self.gate(torch.cat([text_states, speech_mean], dim=-1)))
        fused = gate * text_states + (1.0 - gate) * speech_mean
        return self.dropout(self.norm(fused))


def build_fusion(*, fusion_type: str, hidden_size: int, dropout: float, alignment_window_size: int):
    if fusion_type in {"mean_pool", "mean_pool_speech"}:
        return MeanPoolSpeechFusion(hidden_size=hidden_size, dropout=dropout)
    if fusion_type == "global":
        return GatedCrossModalFusion(hidden_size=hidden_size, dropout=dropout)
    if fusion_type in {"text_anchored", "text_anchored_residual"}:
        return TextAnchoredResidualFusion(hidden_size=hidden_size, dropout=dropout)
    if fusion_type in {"reliability_gated", "reliability_gated_text_anchored"}:
        return ReliabilityGatedTextAnchoredFusion(hidden_size=hidden_size, dropout=dropout)
    if fusion_type in {"precision_controlled", "precision_controlled_text_anchored"}:
        return PrecisionControlledTextAnchoredFusion(hidden_size=hidden_size, dropout=dropout)
    if fusion_type in {"precision_controlled_no_attention", "precision_controlled_noattn"}:
        return PrecisionControlledTextAnchoredFusion(
            hidden_size=hidden_size,
            dropout=dropout,
            use_attention_reliability=False,
        )
    if fusion_type in {"precision_controlled_strict", "precision_controlled_strict_gate"}:
        return PrecisionControlledTextAnchoredFusion(
            hidden_size=hidden_size,
            dropout=dropout,
            gate_bias=-3.0,
        )
    if fusion_type in {"precision_controlled_loose", "precision_controlled_loose_gate"}:
        return PrecisionControlledTextAnchoredFusion(
            hidden_size=hidden_size,
            dropout=dropout,
            gate_bias=-2.0,
        )
    if fusion_type in {"text_uncertainty_guided", "uncertainty_guided", "signer_v3"}:
        return TextUncertaintyGuidedSpeechInterventionFusion(hidden_size=hidden_size, dropout=dropout)
    if fusion_type in {"sparse_topk_text_anchored", "act_sparse_text_anchored"}:
        return SparseTopKTextAnchoredFusion(
            hidden_size=hidden_size,
            dropout=dropout,
            topk=alignment_window_size,
        )
    if fusion_type in {"token_aligned_text_anchored", "local_text_anchored"}:
        return TokenAlignedTextAnchoredFusion(
            hidden_size=hidden_size,
            dropout=dropout,
            window_size=alignment_window_size,
        )
    if fusion_type in {"token_aligned", "local_aligned"}:
        return TokenAlignedCrossModalFusion(
            hidden_size=hidden_size,
            dropout=dropout,
            window_size=alignment_window_size,
        )
    raise ValueError(f"unsupported fusion_type: {fusion_type}")
