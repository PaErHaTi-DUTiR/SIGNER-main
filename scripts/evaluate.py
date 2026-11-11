#!/usr/bin/env python3
"""Gold-free text inference; optionally score against unmodified character gold."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "model_checkpoint/SIGNER.pt")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/predictions.jsonl")
    parser.add_argument("--text-model", default="xlm-roberta-base", help="Tokenizer name or local folder; no pretrained weights loaded")
    parser.add_argument("--encoder-config", type=Path, default=ROOT / "configs/xlm_roberta_base.json")
    parser.add_argument("--vocabulary", type=Path, default=ROOT / "configs/character_vocabulary.json")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-examples", type=int, help="Partial smoke test, not a benchmark evaluation")
    args = parser.parse_args()
    import torch
    from signer.inference import TextInput, load_bundle, predict
    from signer.metrics import transcript, score
    torch.set_num_threads(4)
    rows = [json.loads(s) for s in args.input.read_text(encoding="utf-8").splitlines() if s.strip()]
    if args.max_examples is not None:
        if args.max_examples <= 0:
            parser.error("--max-examples must be positive")
        rows = rows[:args.max_examples]
    vocabulary = json.loads(args.vocabulary.read_text(encoding="utf-8"))
    model, tokenizer, head = load_bundle(args.checkpoint, args.encoder_config, args.text_model, vocabulary, args.device)
    inputs = [TextInput(r["id"], r["language"], transcript(r)) for r in rows]
    predictions = predict(model, tokenizer, head, vocabulary, inputs, args.device, args.batch_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in predictions), encoding="utf-8")
    report = dict(view="text-only", samples=len(rows), partial_smoke=args.max_examples is not None,
                  input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
                  source_split_values=sorted({r.get("split", "unspecified") for r in rows}))
    if all("entities" in r for r in rows):
        report.update(score(predictions, rows))
    args.output.with_suffix(".metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
