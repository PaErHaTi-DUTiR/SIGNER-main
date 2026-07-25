"""Loss functions used by SIGNER."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from signer.constants import IGNORE_INDEX


def masked_cross_entropy_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    label_mask: torch.Tensor,
    class_weights: torch.Tensor | None = None,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    flat_logits = logits.reshape(-1, logits.size(-1))
    flat_labels = labels.reshape(-1)
    loss = F.cross_entropy(
        flat_logits,
        flat_labels,
        ignore_index=IGNORE_INDEX,
        reduction="none",
        weight=class_weights,
        label_smoothing=label_smoothing,
    )
    masked_loss = loss * label_mask.reshape(-1).float()
    normalizer = label_mask.sum().clamp_min(1).float()
    return masked_loss.sum() / normalizer


def masked_focal_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    label_mask: torch.Tensor,
    class_weights: torch.Tensor | None = None,
    gamma: float = 2.0,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    flat_logits = logits.reshape(-1, logits.size(-1))
    flat_labels = labels.reshape(-1)
    ce = F.cross_entropy(
        flat_logits,
        flat_labels,
        ignore_index=IGNORE_INDEX,
        reduction="none",
        weight=class_weights,
        label_smoothing=label_smoothing,
    )
    probs = F.softmax(flat_logits, dim=-1)
    safe_labels = flat_labels.clamp_min(0)
    pt = probs.gather(1, safe_labels.unsqueeze(1)).squeeze(1)
    focal = ((1.0 - pt).clamp_min(1e-6) ** gamma) * ce
    masked_loss = focal * label_mask.reshape(-1).float()
    normalizer = label_mask.sum().clamp_min(1).float()
    return masked_loss.sum() / normalizer


def symmetric_kl_divergence(logits_a: torch.Tensor, logits_b: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    log_prob_a = F.log_softmax(logits_a, dim=-1)
    log_prob_b = F.log_softmax(logits_b, dim=-1)
    prob_a = log_prob_a.exp()
    prob_b = log_prob_b.exp()
    kl_ab = F.kl_div(log_prob_a, prob_b, reduction="none").sum(dim=-1)
    kl_ba = F.kl_div(log_prob_b, prob_a, reduction="none").sum(dim=-1)
    symmetric = 0.5 * (kl_ab + kl_ba)
    symmetric = symmetric * mask.float()
    return symmetric.sum() / mask.sum().clamp_min(1).float()
