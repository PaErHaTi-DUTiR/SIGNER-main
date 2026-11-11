# Single-checkpoint bundle

The release uses one file: **`SIGNER.pt`** (CA-STNER, XLM-R-base, seed42), containing `model_state_dict` and `character_head_state_dict`. The head vocabulary is `configs/character_vocabulary.json`. No second model checkpoint is required.

The binary is approximately1.4GB and intentionally excluded from ordinary Git. **A hosted download is not yet available from this repository.** If you already received the author-provided checkpoint, place it here; otherwise contact the authors. No nonfunctional download link or Git LFS pointer is provided.

Validate the local file against `manifest.json`:

```bash
python scripts/verify_release.py --checkpoint model_checkpoint/SIGNER.pt
```

Use `scripts/evaluate.py` for strictly text-only deployment. It restores only the non-acoustic NER state and the character head, preserving all classifier slots. XLM-R tokenizer files are obtained separately from their original distribution. The MIT code license does not override upstream backbone/checkpoint terms. Load only trusted checkpoint files.
