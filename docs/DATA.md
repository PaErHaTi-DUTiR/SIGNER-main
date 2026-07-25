# CA-STNER data

CA-STNER is a speech-text named entity recognition benchmark for Kazakh
(`kk`), Kyrgyz (`ky`), Tajik (`tg`), Turkmen (`tk`), and Uzbek (`uz`). Its
annotation schema contains `PER`, `LOC`, and `ORG`, with exact character
offsets and BIO tags over whitespace-tokenized transcripts.

## Dataset statistics

| Language | Train/dev/test | Hours | Utterances | Entity utterances | Zero-entity utterances | PER | LOC | ORG | Entities |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Kazakh (`kk`) | 1600/200/200 | 7.05 | 2,000 | 641 | 1,359 (68.0%) | 294 | 758 | 218 | 1,270 |
| Kyrgyz (`ky`) | 1600/200/200 | 6.17 | 2,000 | 1,050 | 950 (47.5%) | 575 | 1,033 | 312 | 1,920 |
| Tajik (`tg`) | 1581/198/197 | 7.16 | 1,976 | 779 | 1,197 (60.6%) | 364 | 959 | 228 | 1,551 |
| Turkmen (`tk`) | 1600/200/200 | 6.48 | 2,000 | 946 | 1,054 (52.7%) | 245 | 1,109 | 399 | 1,753 |
| Uzbek (`uz`) | 1600/200/200 | 4.32 | 2,000 | 1,425 | 575 (28.8%) | 878 | 1,022 | 386 | 2,286 |
| **Total** | **7981/998/997** | **31.17** | **9,976** | **4,841** | **5,135 (51.5%)** | **2,356** | **4,881** | **1,543** | **8,780** |

These values are copied from the final submitted manuscript. No experimental
performance results are included in this repository snapshot.

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
