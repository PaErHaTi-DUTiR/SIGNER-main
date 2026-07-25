# CA-STNER data

CA-STNER is a speech-text named entity recognition benchmark for Kazakh
(`kk`), Kyrgyz (`ky`), Tajik (`tg`), Turkmen (`tk`), and Uzbek (`uz`). Its
annotation schema contains `PER`, `LOC`, and `ORG`, with exact character
offsets and BIO tags over whitespace-tokenized transcripts.

## Initial public sample

`data/ca_stner_sample/sample.jsonl` contains one human-verified annotation
example per language. It is intended for:

- inspecting the schema;
- testing loaders and offset validation;
- understanding multilingual surface forms;
- preparing downstream adapters.

Audio is not redistributed in this lightweight repository snapshot. Each
record retains the expected relative path in the complete dataset. The sample
must not be used to report benchmark performance.

## Public annotation format

Each JSONL record has the following structure:

```json
{
  "id": "example_id",
  "split": "test",
  "language": "kk",
  "audio": {
    "path": "kk/audio/example.wav",
    "sample_rate": 16000,
    "duration_sec": 5.0
  },
  "text": "transcript",
  "tokens": ["transcript"],
  "ner_tags": ["O"],
  "entities": [
    {"text": "entity", "label": "LOC", "start": 0, "end": 6}
  ]
}
```

Offsets follow Python's half-open convention `[start, end)`. For every entity,
`text[start:end]` must exactly equal `entity.text`.

## Complete release layout

```text
data/ca_stner/
├── kk/
│   ├── train.jsonl
│   ├── dev.jsonl
│   ├── test.jsonl
│   └── audio/
├── ky/
├── tg/
├── tk/
└── uz/
```

Use `scripts/prepare_ca_stner.py` to convert this public representation into
the canonical model input format.

## Dataset integrity

The preparation script checks:

1. valid JSON records;
2. unique sample identifiers;
3. disjoint train/dev/test identifiers;
4. equal token and BIO-tag sequence lengths;
5. valid BIO transitions;
6. exact entity-character offsets;
7. supported languages and labels.

## Annotation

The final labels are human-verified. Audio may be consulted to resolve
ambiguity, while entity spans are always defined over the provided transcript.
See [ANNOTATION_GUIDELINES.md](ANNOTATION_GUIDELINES.md) for the complete
protocol.

## Licensing and provenance

The SIGNER code license does not automatically apply to dataset content.
CA-STNER annotations and audio are governed by the terms accompanying the
archival dataset release and by the licenses of the underlying public speech
sources. Users are responsible for complying with those terms. This GitHub
repository does not redistribute third-party model weights or the complete
speech corpus.

