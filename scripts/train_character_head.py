#!/usr/bin/env python3
"""Separate TRAIN-supervised end refinement of a frozen, DEV-selected NER."""
import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ner-checkpoint", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--text-model", default="xlm-roberta-base")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    import torch
    from torch.nn import functional as F
    from transformers import AutoConfig, AutoTokenizer
    from transformers.modeling_utils import no_init_weights
    from signer.bio import legacy_owner
    from signer.inference import TextDeployment, TextInput, predict
    from signer.metrics import transcript, score, span_set
    from signer.models.character_end import CharacterEndHead, make_vocabulary, training_endpoints, pack_features
    random.seed(args.seed); torch.manual_seed(args.seed); torch.set_num_threads(4)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    def read(path):
        return [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]
    train, dev = read(args.train), read(args.dev)
    if not train or not dev or {r["id"] for r in train} & {r["id"] for r in dev}:
        raise ValueError("Nonempty, disjoint TRAIN/DEV required")
    if any(r.get("split") not in (None, "train") for r in train) or any(r.get("split") not in (None, "dev") for r in dev):
        raise ValueError("Only TRAIN supervision and DEV selection allowed")
    for row in train + dev:
        span_set(row)
    vocabulary = {"characters": make_vocabulary([transcript(r) for r in train]),
                  "languages": {"<UNK>": 0, **{s: i+1 for i, s in enumerate(sorted({r["language"] for r in train}))}}}
    saved = torch.load(str(args.ner_checkpoint), map_location="cpu", mmap=True, weights_only=True)
    state = saved["model_state_dict"]
    labels, hidden_size = state["classifier.weight"].shape
    with no_init_weights():
        model = TextDeployment(AutoConfig.from_pretrained(args.text_model), hidden_size, labels)
    model.load_state_dict({k: v for k, v in state.items() if not k.startswith(("speech_encoder.", "fusion."))}, strict=True, assign=True)
    model.requires_grad_(False).to(args.device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.text_model, use_fast=True)
    examples = []
    for first in range(0, len(train), 16):
        part = train[first:first+16]
        enc = tokenizer([transcript(r) for r in part], padding=True, truncation=True, max_length=256,
                        return_offsets_mapping=True, return_tensors="pt")
        with torch.no_grad():
            hidden, _ = model(enc["input_ids"].to(args.device), enc["attention_mask"].to(args.device))
        for i, row in enumerate(part):
            raw = [(e["start"], e["end"], e["label"]) for e in row["entities"]]
            specs, _ = training_endpoints(dict(text=transcript(row), language=row["language"], raw=raw,
                                               offsets=enc["offset_mapping"][i].tolist()), legacy_owner,
                                          vocabulary["characters"], vocabulary["languages"])
            examples.append([(s["feature"], hidden[i, s["token_index"]].detach().cpu(), s["target"]) for s in specs])
    head = CharacterEndHead(len(vocabulary["characters"]), len(vocabulary["languages"]), hidden_size).to(args.device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=.01)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "character_vocabulary.json").write_text(json.dumps(vocabulary, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    inputs = [TextInput(r["id"], r["language"], transcript(r)) for r in dev]
    best, history = -1., []
    for epoch in range(1, 11):
        head.train(); order = list(range(len(examples))); random.shuffle(order)
        for first in range(0, len(order), 16):
            sentences = [examples[i] for i in order[first:first+16] if examples[i]]
            if not sentences:
                continue
            flat = [e for sentence in sentences for e in sentence]
            packed = pack_features([e[0] for e in flat], torch.stack([e[1] for e in flat]), args.device)
            logits = head(**packed)
            targets = torch.tensor([e[2] for e in flat], device=args.device)
            losses = F.cross_entropy(logits, targets, reduction="none")
            # Accepted objective: mean endpoint CE over entities in this logical batch.
            loss = losses.mean()
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.)
            optimizer.step()
        head.eval()
        metrics = score(predict(model, tokenizer, head, vocabulary, inputs, args.device), dev)
        f1 = metrics["overall"]["micro_f1"]
        history.append(dict(epoch=epoch, dev_micro_f1=f1)); print(json.dumps(history[-1]), flush=True)
        if f1 > best:
            best = f1
            torch.save({"model_state_dict": head.state_dict(), "epoch": epoch, "dev_micro_f1": f1}, args.output_dir / "head_best.pt")
    selected = torch.load(str(args.output_dir / "head_best.pt"), map_location="cpu", weights_only=True)
    torch.save({"model_state_dict": state, "character_head_state_dict": selected["model_state_dict"]}, args.output_dir / "SIGNER.pt")
    (args.output_dir / "head_history.json").write_text(json.dumps(history, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
