"""Checkpoint loading helpers with explicit compatibility checks."""

from __future__ import annotations

from pathlib import Path

import torch


def load_model_checkpoint(model: torch.nn.Module, checkpoint_path: str | Path) -> dict:
    """Load a checkpoint while tolerating only legacy position-id buffers."""
    state = torch.load(checkpoint_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state["model_state_dict"], strict=False)
    speech_wrapper = getattr(model, "speech_encoder", None)
    speech_model = getattr(speech_wrapper, "encoder", None)
    speech_config = getattr(speech_model, "config", None)
    adapters_disabled = speech_config is not None and not getattr(speech_config, "add_adapter", False)

    def allowed_missing(key: str) -> bool:
        if key.endswith("position_ids"):
            return True
        return adapters_disabled and key.startswith("speech_encoder.") and ".adapter_layer." in key

    bad_missing = [key for key in missing if not allowed_missing(key)]
    bad_unexpected = [key for key in unexpected if not key.endswith("position_ids")]
    if bad_missing or bad_unexpected:
        raise RuntimeError(
            f"Incompatible checkpoint {checkpoint_path}: "
            f"missing={bad_missing}, unexpected={bad_unexpected}"
        )
    return state
