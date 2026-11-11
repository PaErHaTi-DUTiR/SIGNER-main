# CA-STNER: data format and public scope

## Included files

| File | Records | Purpose |
|:--|--:|:--|
| `data/ca_stner_test/test.jsonl` | 998 | Author-designated test examples; original source split is DEV |
| `data/ca_stner_sample/sample.jsonl` | 5 | Compact token/BIO format examples, one per language |

The supplied test-example file is preserved byte for byte. All 998 records have `split: dev` (kk 200, ky 200, tg 198, tk 200, uz 200). It is not the manuscript's 997-example held-out TEST partition. Its original filename is preserved for the public example workflow. Do not relabel, merge, or use these DEV examples to report the manuscript's held-out TEST numbers.

No full TRAIN split, private audio, API output cache, annotation-review history, manuscript source, or internal experimental output is redistributed here. Audio paths in records are source references, not bundled files or promised download locations. They are not read during text-only inference.

## Transcript-space schema

```json
{
  "id": "example-001",
  "language": "uz",
  "split": "dev",
  "text": {"transcript": "Toshkent", "normalized_transcript": "Toshkent"},
  "entities": [{"start": 0, "end": 8, "text": "Toshkent", "label": "LOC"}]
}
```

This abbreviated synthetic example illustrates the fields used by the text evaluator, not a released corpus record. Source records additionally contain audio and metadata fields.

Offsets are zero-based, end-exclusive Unicode character indices into `text.normalized_transcript`, not UTF-8 byte indices. Entity surfaces must satisfy `text[start:end] == entity.text`. The labels are PER, LOC, and ORG. Empty entity lists are valid and must not be filtered. Preserve original text, identifiers, boundaries, and language codes.

Evaluation requires exact boundaries and type on each sample. A partial overlap does not count as a true positive. Language macro-F1 is the unweighted average of language-specific entity micro-F1, not the average of entity-type scores. Public examples and benchmark scores have different scopes.

## Manuscript corpus

The manuscript reports 9,976 utterances, 31.17 hours, and 8,780 entity mentions, split 7,981/998/997. The five languages are Kazakh, Kyrgyz, Tajik, Turkmen, and Uzbek. These corpus statistics describe the full benchmark, not just the files included here.

## Rights and privacy

The revised manuscript identifies Mozilla Common Voice v25 as the speech source for all five languages. Underlying source data-use terms and consent requirements continue to apply. This repository's MIT license covers original software; it is not a blanket license for speech, personal information, or third-party datasets. No audio is redistributed in this release. A separate dataset-specific reuse license has not been included; contact the authors before redistribution beyond the supplied research examples.
