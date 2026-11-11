# SIGNER model card

## Scope

The supplied single `SIGNER.pt` is the retained CA-STNER **base / seed42** model, with a NER state and separately trained character-end head in one bundle. Its tensors were checked against the retained DEV-selected NER and head checkpoints. It is not a large-backbone model, an ASR-adapted variant, or a three-seed ensemble.

Input: transcript and language code. Output: original-character PER/LOC/ORG entity spans; the state also retains unused-in-gold legacy MISC output slots. The loader preserves these slots rather than changing the checkpoint to force a seven-slot shape. The language-conditioned character head uses a TRAIN-only vocabulary provided in `configs/character_vocabulary.json`.

## Architecture and training

XLM-R-base is trained with paired frozen Whisper-small features through text-anchored residual fusion and shared-head dual-view learning. The total NER objective is speech-text CE + 0.3 × text CE + 0.5 × symmetric KL. Additive view embeddings distinguish the views. The character head is trained separately on frozen NER states with gold character-end targets and cross-entropy over valid endpoints. It preserves the predicted start/type and cannot add entities.

The released `scripts/evaluate.py` instantiates only the text encoder, text-view embedding, classifier, and character head. Stored acoustic/fusion tensors are excluded with strict key checking. No speech model, audio decoder, gold entity, entity gazetteer, or test-selected parameter enters prediction.

## Interpreting scores

README tables are manuscript three-seed TEST results. They are not new measurements of this one checkpoint on the public examples. The author-designated test-example file actually retains DEV provenance; it must not be used to claim reproduction of held-out TEST performance. A short `--max-examples` run is a smoke test only.

## Limitations

- Exact-character boundaries are sensitive to tokenization, spelling, and morphology.
- The endpoint head can shorten the final token span but cannot fix a wrong start/type or a missed entity.
- Speech-to-token attention is soft attention, not forced alignment.
- Training audio dropout is drawn once per batch, not independently per sample.
- Performance and annotation coverage vary by language and domain; aggregate gains are not uniform gains for every language/type.
- Pretrained-model and source-data biases may remain. This model is for research, not unvalidated high-stakes decisions.

Load only trusted PyTorch checkpoint files. Respect upstream model terms and dataset/source rights. See `model_checkpoint/README.md` for distribution status and SHA-256 verification.
