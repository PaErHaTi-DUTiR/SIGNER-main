# Training and evaluation

Full retraining requires authorized paired TRAIN/DEV data and audio obtained separately. The public example file is not a complete training dataset. Do not use TEST for development or checkpoint selection.

## 1. Text-only warm-start

Set dataset and model paths in `configs/ca_stner_text_xlmr.yaml`. Keep TRAIN and DEV disjoint, preserve all raw entity spans, and use the same tokenizer and classifier schema for the subsequent SIGNER stage.

```bash
python scripts/train.py --config configs/ca_stner_text_xlmr.yaml
```

The base model uses the original XLM-R-base pretrained initialization. The matched warm-start restores the text encoder/projection, classifier, and view embedding, while speech/fusion parameters remain newly initialized.

## 2. Paired dual-view NER

Set the same data paths in `configs/ca_stner_signer.yaml`, and pass the DEV-selected text checkpoint:

```bash
python scripts/train.py --config configs/ca_stner_signer.yaml \
  --init-checkpoint outputs/ca_stner/text_xlmr/signer_best.pt
```

The main objective is `CE_speech-text + 0.3 CE_text + 0.5 symmetric_KL`, using AdamW, LR 1e-5, weight decay .01, 10 epochs, gradient clipping 1, effective batch16 and eval batch16. CA uses positive-example oversampling2 and non-O weight3. Frozen Whisper-small supplies speech states; XLM-R and fusion are trainable. The main NER checkpoint is chosen by full-DEV exact original-character micro-F1 in the text-only view. BIO supervision includes the retained duplicate-B repair; reference character spans are never projected to tokenizer boundaries for scoring.

Audio dropout is a batch-level draw at .1. On a dropped batch, cached speech is removed, audio features/masks are zeroed, and the view changes to text-only. Both supervised passes and the consistency term are still computed; absence of audio does not automatically disable these losses.

The retained public bundle uses a nine-slot classifier (`O`, B/I-PER, B/I-LOC, B/I-ORG, B/I-MISC). CA-STNER annotations contain only the first three entity types; legacy MISC slots are retained for tensor compatibility, not additional CA gold labels. Do not reshape this supplied checkpoint. MSNER's gold includes MISC.

## 3. Separate character-end head

```bash
python scripts/train_character_head.py \
  --ner-checkpoint outputs/ca_stner/signer/signer_best.pt \
  --train data/ca_stner_signer/train.jsonl \
  --dev data/ca_stner_signer/dev.jsonl \
  --text-model xlm-roberta-base \
  --output-dir outputs/ca_stner/character_head --seed 42 --device cuda
```

The NER is frozen in eval mode. TRAIN-only characters and language IDs define the vocabulary. Each covered gold entity end is a categorical target among nonempty character endpoints inside its covering terminal token. The endpoint head uses cross-entropy, AdamW LR1e-3, weight decay.01, batch16, clipping1, 10epochs, and DEV exact-span selection. It is **not** jointly trained with NER and receives no TEST targets. Predictions refine only the end of each BIO-decoded span; starts and types are unchanged. The script saves a single deployment bundle `SIGNER.pt` and its matching vocabulary.

The first two stages are public training entry points, not a packaged full-data experiment queue or a guarantee of bitwise reproduction across hardware. Preserve resolved configs, seeds, tokenizers, and stage-specific DEV selection when conducting independent retraining.

## 4. Frozen evaluation

```bash
python scripts/evaluate.py --checkpoint model_checkpoint/SIGNER.pt \
  --input data/ca_stner_test/test.jsonl --device cpu
```

The input filename does not override its source split. The supplied 998 author-designated test examples retain `split: dev` and do not reproduce the manuscript's held-out TEST partition. Predictions depend only on text/language; gold is read by the separate scorer afterwards. Precision/recall/F1 require exact original-character start, end, and type, including empty-entity sentences and truncation failures in the reference denominator.

For repeated experiments, use seeds42/43/44, select each run independently on DEV, freeze the method, then evaluate TEST. Report mean ± sample SD, not a min–max range. The released one-checkpoint example is not a substitute for all three runs.
