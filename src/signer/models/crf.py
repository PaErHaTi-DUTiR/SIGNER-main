"""Small linear-chain CRF for BIO token decoding."""

from __future__ import annotations

import torch
import torch.nn as nn


class LinearChainCRF(nn.Module):
    """Batch-first linear-chain CRF.

    This implementation is intentionally local to SIGNER to avoid making the
    experiment depend on an external CRF package.
    """

    def __init__(self, num_tags: int) -> None:
        super().__init__()
        self.num_tags = num_tags
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
        nn.init.uniform_(self.transitions, -0.1, 0.1)

    def forward(self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Return log-likelihood for each sequence."""
        mask = mask.bool()
        emissions = emissions.float()
        tags = tags.long()
        numerator = self._compute_score(emissions, tags, mask)
        denominator = self._compute_normalizer(emissions, mask)
        return numerator - denominator

    def neg_log_likelihood(self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return -self.forward(emissions, tags, mask).mean()

    def _compute_score(self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_length, _ = emissions.shape
        first_tags = tags[:, 0]
        score = self.start_transitions[first_tags] + emissions[:, 0].gather(1, first_tags.unsqueeze(1)).squeeze(1)

        for index in range(1, seq_length):
            prev_tags = tags[:, index - 1]
            curr_tags = tags[:, index]
            transition_score = self.transitions[prev_tags, curr_tags]
            emission_score = emissions[:, index].gather(1, curr_tags.unsqueeze(1)).squeeze(1)
            score = score + (transition_score + emission_score) * mask[:, index]

        lengths = mask.long().sum(dim=1).clamp_min(1)
        last_tags = tags.gather(1, (lengths - 1).unsqueeze(1)).squeeze(1)
        score = score + self.end_transitions[last_tags]
        return score

    def _compute_normalizer(self, emissions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        score = self.start_transitions + emissions[:, 0]
        for index in range(1, emissions.size(1)):
            broadcast_score = score.unsqueeze(2)
            broadcast_emission = emissions[:, index].unsqueeze(1)
            next_score = torch.logsumexp(broadcast_score + self.transitions + broadcast_emission, dim=1)
            score = torch.where(mask[:, index].unsqueeze(1), next_score, score)
        score = score + self.end_transitions
        return torch.logsumexp(score, dim=1)

    def decode(self, emissions: torch.Tensor, mask: torch.Tensor) -> list[list[int]]:
        mask = mask.bool()
        emissions = emissions.float()
        score = self.start_transitions + emissions[:, 0]
        history: list[torch.Tensor] = []

        for index in range(1, emissions.size(1)):
            next_score = score.unsqueeze(2) + self.transitions + emissions[:, index].unsqueeze(1)
            best_score, best_path = next_score.max(dim=1)
            score = torch.where(mask[:, index].unsqueeze(1), best_score, score)
            history.append(best_path)

        score = score + self.end_transitions
        best_last_score, best_last_tag = score.max(dim=1)
        del best_last_score

        lengths = mask.long().sum(dim=1).clamp_min(1)
        paths: list[list[int]] = []
        for batch_index in range(emissions.size(0)):
            seq_len = int(lengths[batch_index].item())
            best_tag = int(best_last_tag[batch_index].item())
            path = [best_tag]
            for hist in reversed(history[: seq_len - 1]):
                best_tag = int(hist[batch_index][best_tag].item())
                path.append(best_tag)
            path.reverse()
            if seq_len < emissions.size(1):
                path.extend([0] * (emissions.size(1) - seq_len))
            paths.append(path)
        return paths
