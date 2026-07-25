# Reproducibility protocol

## Task

The task is transcript-space named entity recognition with `PER`, `LOC`, and
`ORG` labels. Entity spans use exact character boundaries. Unless explicitly
stated otherwise, SIGNER uses speech only during training and is evaluated
through its text-only view.

## Main training sequence

1. Convert the public dataset with `scripts/prepare_ca_stner.py`.
2. Train the matched XLM-R text-only model.
3. Initialize SIGNER from that text-only checkpoint.
4. Train SIGNER with paired speech and transcripts.
5. Select hyperparameters on the development split.
6. Evaluate the frozen selected configuration once on the test split.

The two stages must use the same label schema, data splits, tokenizer family,
and exact-span evaluator.

## Base configuration

| Setting | Value |
|---|---:|
| Text encoder | XLM-R-base |
| Speech encoder | Whisper-small, frozen |
| Fusion | text-anchored residual |
| Hidden size | 768 |
| Optimizer | AdamW |
| Learning rate | `1e-5` |
| Weight decay | `0.01` |
| Batch size | `12` |
| Epochs | 10 |
| Consistency weight | `0.5` |
| Audio dropout | `0.1` |
| Gradient clipping | `1.0` |
| Seed | `42` |

The matched text-only baseline uses the values in
`configs/ca_stner_text_xlmr.yaml`. The speech-informed configuration is stored
in `configs/ca_stner_signer.yaml`.

## Views and losses

The speech-text view uses text states as queries over speech frames and injects
the acoustic context through a gated residual. The text-only view removes the
speech states and uses an all-zero audio mask. Both views share the token
classifier.

Training combines:

- supervised NER loss on the primary view;
- auxiliary supervised NER loss on the text-only view;
- symmetric KL consistency over valid labeled token positions.

The current reference implementation applies the audio-dropout draw once per
batch. When triggered, speech states are removed, audio features and masks are
zeroed, and the view identifier is changed to text-only.

## Evaluation

`scripts/evaluate.py` decodes BIO tags into spans and computes micro
precision/recall/F1. A true positive requires an exact match of:

- sample;
- start boundary;
- end boundary;
- entity type.

The default SIGNER evaluation view is `text-only`; no audio is required by the
deployed prediction path.

## Low-resource curves

Low-resource subsets are stratified by language. The labels `50`, `100`,
`200`, `500`, and `1k` denote the number of training examples **per language**.
For a fixed seed, subsets are nested. The reported seeds are `0`, `1`, and `2`.
`full` denotes the complete training split and is distinct from `1k`.

## Environment capture

For archival runs, record:

```bash
python --version
python -c "import torch, transformers; print(torch.__version__, transformers.__version__)"
nvidia-smi
pip freeze > outputs/environment.txt
```

Store the resolved YAML emitted by `scripts/train.py` beside every checkpoint.
