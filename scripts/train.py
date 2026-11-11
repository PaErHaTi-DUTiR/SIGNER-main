#!/usr/bin/env python3
"""Minimal training entrypoint for SIGNER."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, WeightedRandomSampler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from signer.config import dump_config, load_config  # noqa: E402
from signer.constants import BIO_TAGS, TAG_TO_ID, VIEW_NAME_TO_ID  # noqa: E402
from signer.data.collator import SignerBatchCollator  # noqa: E402
from signer.data.dataset import CASpeechNERDataset  # noqa: E402
from signer.models.model import SignerTokenClassifier  # noqa: E402
from signer.training.losses import symmetric_kl_divergence  # noqa: E402
from signer.utils.io import ensure_dir  # noqa: E402
from signer.utils.ner import compute_span_prf, decode_bio_tags  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_output_dir(output_dir: str | Path | None) -> Path:
    if output_dir is None:
        raise ValueError("output_dir must be provided")
    path = Path(output_dir)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return ensure_dir(path)


def build_sampler(dataset: CASpeechNERDataset, positive_multiplier: float):
    if positive_multiplier <= 1.0:
        return None
    weights = []
    for sample in dataset.samples:
        weights.append(float(positive_multiplier if sample.entities else 1.0))
    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)


def build_class_weights(non_o_loss_weight: float, device: torch.device) -> torch.Tensor | None:
    if non_o_loss_weight <= 1.0:
        return None
    weights = torch.ones(len(BIO_TAGS), dtype=torch.float32, device=device)
    for tag, idx in TAG_TO_ID.items():
        if tag != 'O':
            weights[idx] = non_o_loss_weight
    return weights


def build_text_only_batch(batch: dict) -> dict:
    text_batch = dict(batch)
    text_batch.pop("speech_states", None)
    text_batch["audio_features"] = torch.zeros_like(batch["audio_features"])
    text_batch["audio_mask"] = torch.zeros_like(batch["audio_mask"], dtype=torch.bool)
    text_batch["view_type"] = torch.full_like(batch["view_type"], VIEW_NAME_TO_ID["text-only"])
    return text_batch


def build_speech_conditioned_batch(batch: dict) -> dict:
    speech_batch = dict(batch)
    speech_batch["text_input_ids"] = batch["placeholder_input_ids"]
    speech_batch["view_type"] = torch.full_like(batch["view_type"], VIEW_NAME_TO_ID["speech-only"])
    return speech_batch


def apply_modality_dropout(batch: dict, *, audio_dropout: float, text_dropout: float) -> dict:
    if audio_dropout <= 0.0 and text_dropout <= 0.0:
        return batch
    dropped = dict(batch)
    if audio_dropout > 0.0 and random.random() < audio_dropout:
        dropped.pop("speech_states", None)
        dropped["audio_features"] = torch.zeros_like(batch["audio_features"])
        dropped["audio_mask"] = torch.zeros_like(batch["audio_mask"], dtype=torch.bool)
        dropped["view_type"] = torch.full_like(batch["view_type"], VIEW_NAME_TO_ID["text-only"])
    elif text_dropout > 0.0 and random.random() < text_dropout:
        dropped = build_speech_conditioned_batch(dropped)
    return dropped


def load_compatible_checkpoint(model: SignerTokenClassifier, checkpoint_path: str | Path) -> dict:
    init_state = torch.load(str(checkpoint_path), map_location='cpu', weights_only=True)
    source_state = init_state['model_state_dict']
    target_state = model.state_dict()
    expected = {k for k in target_state if k.startswith(('text_branch.', 'classifier.', 'modality_embedding.'))}
    if set(source_state) != expected or any(source_state[k].shape != target_state[k].shape for k in expected):
        raise ValueError('Warm-start must restore every text/classifier/view tensor, and no speech/fusion tensor')
    compatible_state = {}
    skipped_shape = []
    for key, value in source_state.items():
        if key in target_state and target_state[key].shape == value.shape:
            compatible_state[key] = value
        elif key in target_state:
            skipped_shape.append({
                "key": key,
                "checkpoint_shape": list(value.shape),
                "model_shape": list(target_state[key].shape),
            })
    missing, unexpected = model.load_state_dict(compatible_state, strict=False)
    return {
        "init_checkpoint": str(checkpoint_path),
        "loaded_keys": len(compatible_state),
        "skipped_shape_mismatch": skipped_shape,
        "missing_keys": missing,
        "unexpected_keys": unexpected,
    }


def evaluate_text_view_micro_f1(
    model: SignerTokenClassifier,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """Evaluate exact-span micro-F1 through the deployable text-only view."""
    was_training = model.training
    model.eval()
    predicted_spans = []
    reference_spans = []
    with torch.no_grad():
        for batch in loader:
            tensor_batch = {
                key: value.to(device) if hasattr(value, "to") else value
                for key, value in batch.items()
            }
            predictions = model.decode(tensor_batch)
            for pred_ids, entities, offsets, mask in zip(
                predictions,
                batch["entity_spans"],
                batch["offset_mappings"],
                batch["label_mask"],
            ):
                valid_pred_ids = torch.tensor(pred_ids, dtype=torch.long)[mask].tolist()
                valid_offsets = [o for o, valid in zip(offsets, mask.tolist()) if valid]
                predicted_spans.append({(min(a for a, b in valid_offsets[first:last]),
                                         max(b for a, b in valid_offsets[first:last]), label)
                                        for first, last, label in decode_bio_tags(valid_pred_ids)})
                reference_spans.append({(e["start"], e["end"], e["label"]) for e in entities})
    if was_training:
        model.train()
    return float(compute_span_prf(predicted_spans, reference_spans)["micro_f1"])


def main() -> None:
    args = parse_args()
    config = load_config(args.config, project_root=PROJECT_ROOT)
    if args.max_steps is not None:
        config.training.max_steps = args.max_steps
    if args.init_checkpoint is not None:
        config.training.init_checkpoint = str(args.init_checkpoint.resolve())
    output_dir = resolve_output_dir(args.output_dir or config.training.output_dir)
    device = torch.device(args.device or config.training.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    set_seed(config.training.seed)

    dump_config(config, output_dir / "config.resolved.yaml")

    train_dataset = CASpeechNERDataset(
        jsonl_path=Path(config.data.dataset_root) / config.data.train_file,
        dataset_root=config.data.dataset_root,
        tokenizer_name_or_path=config.model.text_model_name_or_path,
        max_text_length=config.data.max_text_length,
        default_view=config.data.train_default_view,
        sample_views=config.data.sample_training_views,
        view_probabilities=config.data.train_view_probs,
        speech_feature_cache_dir=config.data.speech_feature_cache_dir,
    )
    collator = SignerBatchCollator(
        whisper_name_or_path=config.model.whisper_name_or_path,
        speech_encoder_type=config.model.speech_encoder_type,
        sample_rate=config.data.sample_rate,
        max_audio_seconds=config.data.max_audio_seconds,
    )
    sampler = build_sampler(train_dataset, config.training.positive_example_oversampling)
    loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=config.training.num_workers,
        collate_fn=collator,
    )
    dev_dataset = CASpeechNERDataset(
        jsonl_path=Path(config.data.dataset_root) / config.data.dev_file,
        dataset_root=config.data.dataset_root,
        tokenizer_name_or_path=config.model.text_model_name_or_path,
        max_text_length=config.data.max_text_length,
        default_view="text-only",
        sample_views=False,
        speech_feature_cache_dir=config.data.speech_feature_cache_dir,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=config.training.eval_batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        collate_fn=collator,
    )

    model = SignerTokenClassifier(config.model).to(device)
    init_checkpoint = config.training.init_checkpoint
    if init_checkpoint:
        print(json.dumps(load_compatible_checkpoint(model, init_checkpoint), indent=2))

    speech_enc_lr_mult = config.training.speech_encoder_lr_multiplier
    if (
        speech_enc_lr_mult != 1.0
        and model.speech_encoder is not None
        and hasattr(model.speech_encoder, "encoder")
    ):
        speech_enc_params = set(id(p) for p in model.speech_encoder.encoder.parameters() if p.requires_grad)
        other_params = [p for p in model.parameters() if p.requires_grad and id(p) not in speech_enc_params]
        speech_params = [p for p in model.speech_encoder.encoder.parameters() if p.requires_grad]
        optimizer = AdamW(
            [
                {"params": other_params, "lr": config.training.learning_rate},
                {"params": speech_params, "lr": config.training.learning_rate * speech_enc_lr_mult},
            ],
            weight_decay=config.training.weight_decay,
        )
    else:
        optimizer = AdamW(model.parameters(), lr=config.training.learning_rate, weight_decay=config.training.weight_decay)
    class_weights = build_class_weights(config.training.non_o_loss_weight, device)

    model.train()
    global_step = 0
    optimizer_step = 0
    log_window_steps = 0
    running_loss = 0.0
    final_loss = None
    best_dev_micro_f1 = -1.0
    best_epoch = None
    best_checkpoint_path = output_dir / "signer_best.pt"
    accumulation_steps = max(int(config.training.gradient_accumulation_steps), 1)
    optimizer.zero_grad()
    for epoch in range(config.training.num_epochs):
        model.train()
        for batch in loader:
            global_step += 1
            batch = {key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()}
            batch = apply_modality_dropout(
                batch,
                audio_dropout=config.training.audio_dropout,
                text_dropout=config.training.text_dropout,
            )
            outputs = model(batch, class_weights=class_weights)
            loss = outputs["loss"]

            use_text_aux = config.training.aux_text_loss_weight > 0.0
            use_consistency = config.training.consistency_weight > 0.0
            if config.model.use_speech_branch and (use_text_aux or use_consistency):
                text_batch = build_text_only_batch(batch)
                text_outputs = model(text_batch, class_weights=class_weights)
                if use_text_aux:
                    loss = loss + config.training.aux_text_loss_weight * text_outputs["loss"]
                if use_consistency:
                    loss = loss + config.training.consistency_weight * symmetric_kl_divergence(
                        logits_a=outputs["logits"],
                        logits_b=text_outputs["logits"],
                        mask=batch["label_mask"],
                    )
            (loss / accumulation_steps).backward()

            final_loss = float(loss.item())
            running_loss += final_loss
            log_window_steps += 1

            should_step = global_step % accumulation_steps == 0
            should_stop = bool(config.training.max_steps and global_step >= config.training.max_steps)
            if should_step or should_stop:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip_norm)
                optimizer.step()
                optimizer.zero_grad()
                optimizer_step += 1

            if global_step % config.training.log_every_steps == 0:
                avg_loss = running_loss / max(log_window_steps, 1)
                print(f"epoch={epoch + 1} step={global_step} loss={avg_loss:.4f}")
                running_loss = 0.0
                log_window_steps = 0

            if config.training.max_steps and global_step >= config.training.max_steps:
                break

        dev_micro_f1 = evaluate_text_view_micro_f1(model, dev_loader, device)
        print(f"epoch={epoch + 1} dev_text_micro_f1={dev_micro_f1:.6f}")
        if dev_micro_f1 > best_dev_micro_f1:
            best_dev_micro_f1 = dev_micro_f1
            best_epoch = epoch + 1
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": best_epoch,
                    "dev_text_micro_f1": best_dev_micro_f1,
                    "selection_split": "dev",
                    "selection_view": "text-only",
                    "selection_metric": "exact_original_character_span_and_type_micro_f1",
                },
                best_checkpoint_path,
            )
        if config.training.max_steps and global_step >= config.training.max_steps:
            break

    last_checkpoint_path = output_dir / "signer_last.pt"
    torch.save({"model_state_dict": model.state_dict()}, last_checkpoint_path)

    train_metrics = {
        "checkpoint_path": str(best_checkpoint_path),
        "best_checkpoint_path": str(best_checkpoint_path),
        "last_checkpoint_path": str(last_checkpoint_path),
        "best_epoch": best_epoch,
        "best_dev_text_micro_f1": best_dev_micro_f1,
        "device": str(device),
        "num_steps": global_step,
        "optimizer_steps": optimizer_step,
        "final_loss": final_loss,
        "train_default_view": config.data.train_default_view,
        "sample_training_views": config.data.sample_training_views,
        "use_speech_branch": config.model.use_speech_branch,
        "positive_example_oversampling": config.training.positive_example_oversampling,
        "non_o_loss_weight": config.training.non_o_loss_weight,
        "fusion_type": config.model.fusion_type,
        "share_modality_embedding": config.model.share_modality_embedding,
        "decoder_type": config.model.decoder_type,
        "loss_type": config.model.loss_type,
        "label_smoothing": config.model.label_smoothing,
        "focal_gamma": config.model.focal_gamma,
        "use_entity_prior": config.model.use_entity_prior,
        "entity_prior_loss_weight": config.model.entity_prior_loss_weight,
        "gradient_accumulation_steps": config.training.gradient_accumulation_steps,
        "audio_dropout": config.training.audio_dropout,
        "text_dropout": config.training.text_dropout,
        "init_checkpoint": config.training.init_checkpoint,
    }
    (output_dir / "train_metrics.json").write_text(json.dumps(train_metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(train_metrics, indent=2))
    print(f"saved best checkpoint to {best_checkpoint_path}")
    print(f"saved last checkpoint to {last_checkpoint_path}")


if __name__ == "__main__":
    main()
