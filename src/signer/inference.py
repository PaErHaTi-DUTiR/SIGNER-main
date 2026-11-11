"""Strictly text-only deployment; no gold or speech inputs."""
from dataclasses import dataclass
import torch
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.modeling_utils import no_init_weights
from signer.models.character_end import CharacterEndHead, prediction_endpoints, refine_spans
from signer.utils.ner import decode_bio_tags


@dataclass(frozen=True)
class TextInput:
    id: str
    language: str
    text: str


class TextDeployment(nn.Module):
    def __init__(self, config, hidden_size, num_labels):
        super().__init__()
        self.text_branch = nn.Module()
        self.text_branch.encoder = AutoModel.from_config(config)
        self.text_branch.projection = nn.Identity() if config.hidden_size == hidden_size else nn.Linear(config.hidden_size, hidden_size)
        self.modality_embedding = nn.Embedding(3, hidden_size)
        self.dropout = nn.Dropout(.1)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, ids, mask):
        hidden = self.text_branch.encoder(input_ids=ids, attention_mask=mask.long()).last_hidden_state
        hidden = self.text_branch.projection(hidden) + self.modality_embedding.weight[0]
        return hidden, self.classifier(self.dropout(hidden))


def load_bundle(checkpoint, encoder_config, tokenizer_path, vocabulary, device="cpu"):
    """Load trusted bundle with strict text/head tensor coverage; never load Whisper."""
    saved = torch.load(str(checkpoint), map_location="cpu", mmap=True, weights_only=True)
    if set(saved) != {"model_state_dict", "character_head_state_dict"}:
        raise ValueError("Expected bundled NER + character head")
    source = saved["model_state_dict"]
    num_labels, hidden_size = source["classifier.weight"].shape
    if num_labels not in (7, 9):
        raise ValueError("Unsupported BIO classifier")
    with no_init_weights():
        model = TextDeployment(AutoConfig.from_pretrained(str(encoder_config)), hidden_size, num_labels)
    state = {k: v for k, v in source.items() if not k.startswith(("speech_encoder.", "fusion."))}
    if set(state) != set(model.state_dict()):
        raise ValueError("Checkpoint text-state coverage mismatch")
    model.load_state_dict(state, strict=True, assign=True)
    model.requires_grad_(False).to(device).eval()
    head = CharacterEndHead(len(vocabulary["characters"]), len(vocabulary["languages"]), hidden_size)
    head.load_state_dict(saved["character_head_state_dict"], strict=True)
    head.requires_grad_(False).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), use_fast=True)
    if not tokenizer.is_fast:
        raise ValueError("A fast offset-aware tokenizer is required")
    return model, tokenizer, head


def predict(model, tokenizer, head, vocabulary, rows, device="cpu", batch_size=16, max_length=256):
    if batch_size <= 0 or any(not isinstance(r, TextInput) for r in rows):
        raise ValueError("Expected positive batch size and gold-free TextInput rows")
    result = []
    for first in range(0, len(rows), batch_size):
        part = rows[first:first + batch_size]
        encoded = tokenizer([r.text for r in part], padding=True, truncation=True,
                            max_length=max_length, return_offsets_mapping=True, return_tensors="pt")
        offsets = encoded.pop("offset_mapping")
        with torch.inference_mode():
            hidden, logits = model(encoded["input_ids"].to(device), encoded["attention_mask"].to(device))
            tags = logits.argmax(-1).cpu().tolist()
            for i, row in enumerate(part):
                length = int(encoded["attention_mask"][i].sum())
                specs = prediction_endpoints(row.text, row.language, offsets[i, :length].tolist(), tags[i][:length],
                                             decode_bio_tags, vocabulary["characters"], vocabulary["languages"])
                states = [hidden[i, s["token_index"]] for s in specs]
                spans = refine_spans(head, specs, torch.stack(states) if states else states, device)
                result.append({"id": row.id, "language": row.language, "text": row.text,
                               "entities": [{"start": a, "end": b, "label": label, "text": row.text[a:b]}
                                            for a, b, label in spans]})
    return result
