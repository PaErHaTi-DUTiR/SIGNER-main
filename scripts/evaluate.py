#!/usr/bin/env python3
"""Evaluation entrypoint for SIGNER."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from signer.config import load_config  # noqa: E402
from signer.constants import ID_TO_TAG  # noqa: E402
from signer.data.collator import SignerBatchCollator  # noqa: E402
from signer.data.dataset import CASpeechNERDataset  # noqa: E402
from signer.models.model import SignerTokenClassifier  # noqa: E402
from signer.utils.checkpoint import load_model_checkpoint  # noqa: E402
from signer.utils.ner import compute_span_prf, decode_bio_tags  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--view", choices=["text-only", "speech-only", "speech-text"], default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--language", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config, project_root=PROJECT_ROOT)
    device = torch.device(args.device or config.training.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    split_file = config.data.dev_file if args.split == "dev" else config.data.test_file
    view_name = args.view or config.data.view_for_eval

    dataset = CASpeechNERDataset(
        jsonl_path=Path(config.data.dataset_root) / split_file,
        dataset_root=config.data.dataset_root,
        tokenizer_name_or_path=config.model.text_model_name_or_path,
        max_text_length=config.data.max_text_length,
        default_view=view_name,
        sample_views=False,
        speech_feature_cache_dir=config.data.speech_feature_cache_dir,
    )
    if args.language is not None:
        language_indices = [
            index for index, sample in enumerate(dataset.samples) if sample.language == args.language
        ]
        dataset = Subset(dataset, language_indices)
    if args.max_examples is not None:
        dataset = Subset(dataset, range(min(args.max_examples, len(dataset))))
    collator = SignerBatchCollator(
        whisper_name_or_path=config.model.whisper_name_or_path,
        speech_encoder_type=config.model.speech_encoder_type,
        sample_rate=config.data.sample_rate,
        max_audio_seconds=config.data.max_audio_seconds,
    )
    loader = DataLoader(dataset, batch_size=config.training.eval_batch_size, shuffle=False, collate_fn=collator)

    model = SignerTokenClassifier(config.model).to(device)
    load_model_checkpoint(model, args.checkpoint)
    model.eval()

    predicted_spans = []
    reference_spans = []
    predicted_tag_distribution = {}
    reference_tag_distribution = {}
    # Accumulate per-label counts dynamically from the data (supports any label space)
    per_label_counts: dict[str, dict] = {}
    non_empty_pred_examples = 0
    non_empty_ref_examples = 0
    with torch.no_grad():
        for batch in loader:
            tensor_batch = {key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()}
            predictions = torch.tensor(model.decode(tensor_batch), dtype=torch.long)
            labels = batch["token_labels"]
            masks = batch["label_mask"]
            for pred_ids, label_ids, mask in zip(predictions, labels, masks):
                valid_pred = pred_ids[mask].tolist()
                valid_label = label_ids[mask].tolist()
                decoded_pred = decode_bio_tags(valid_pred)
                pred_spans = set(decoded_pred)
                ref_spans = set(decode_bio_tags(valid_label))
                predicted_spans.append(pred_spans)
                reference_spans.append(ref_spans)
                # Collect all labels seen in this batch
                seen_labels = {span[2] for span in pred_spans} | {span[2] for span in ref_spans}
                for label in seen_labels:
                    if label not in per_label_counts:
                        per_label_counts[label] = {"predicted": 0, "reference": 0, "matched": 0}
                for label, counts in per_label_counts.items():
                    label_predictions = {span for span in pred_spans if span[2] == label}
                    label_references = {span for span in ref_spans if span[2] == label}
                    counts["predicted"] += len(label_predictions)
                    counts["reference"] += len(label_references)
                    counts["matched"] += len(label_predictions & label_references)
                if pred_spans:
                    non_empty_pred_examples += 1
                if ref_spans:
                    non_empty_ref_examples += 1
                for tag_id in valid_pred:
                    tag = ID_TO_TAG[int(tag_id)]
                    predicted_tag_distribution[tag] = predicted_tag_distribution.get(tag, 0) + 1
                for tag_id in valid_label:
                    tag = ID_TO_TAG[int(tag_id)]
                    reference_tag_distribution[tag] = reference_tag_distribution.get(tag, 0) + 1

    metrics = compute_span_prf(predicted_spans, reference_spans)
    per_label_metrics = {}
    for label, counts in per_label_counts.items():
        precision = counts["matched"] / counts["predicted"] if counts["predicted"] else 0.0
        recall = counts["matched"] / counts["reference"] if counts["reference"] else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_label_metrics[label] = {**counts, "precision": precision, "recall": recall, "f1": f1}
    payload = {
        "split": args.split,
        "language": args.language or "all",
        "view": view_name,
        "checkpoint": str(args.checkpoint),
        "num_examples": len(dataset),
        "non_empty_pred_examples": non_empty_pred_examples,
        "non_empty_ref_examples": non_empty_ref_examples,
        "predicted_tag_distribution": predicted_tag_distribution,
        "reference_tag_distribution": reference_tag_distribution,
        "per_label": per_label_metrics,
        **metrics,
    }
    if args.output_json:
        output_path = args.output_json if args.output_json.is_absolute() else (PROJECT_ROOT / args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
