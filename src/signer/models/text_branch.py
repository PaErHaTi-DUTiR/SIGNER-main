"""Text encoder branch for SIGNER."""

from __future__ import annotations

import torch.nn as nn
from transformers import AutoModel


class XLMRTextBranch(nn.Module):
    def __init__(self, model_name_or_path: str, output_dim: int, freeze_encoder: bool = False) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name_or_path)
        hidden_size = self.encoder.config.hidden_size
        self.projection = nn.Identity() if hidden_size == output_dim else nn.Linear(hidden_size, output_dim)
        if freeze_encoder:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return self.projection(outputs.last_hidden_state)
