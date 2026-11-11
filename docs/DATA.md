# CA-STNER: data format and public scope

## Included files

| File | Records | Purpose |
|:--|--:|:--|
| `data/ca_stner_test/test.jsonl` | 997 | CA-STNER TEST set |
| `data/ca_stner_sample/sample.jsonl` | 5 | Compact token/BIO format examples, one per language |

The author-confirmed TEST file is preserved byte for byte. All 997 records have `split: test`: kk 200, ky 200, tg 197, tk 200, and uz 200. These counts match the manuscript's TEST partition. Keep this split held out from training, hyperparameter selection, and checkpoint selection; use separately obtained TRAIN/DEV data for development.

No TRAIN/DEV split, private audio, API output cache, annotation-review history, manuscript source, or internal experimental output is redistributed here. Audio paths in records are source references, not bundled files or promised download locations. They are not read during text-only inference.

## Transcript-space schema

```json
{
  "id": "example-001",
  "language": "uz",
  "split": "test",
  "text": {"transcript": "Toshkent", "normalized_transcript": "Toshkent"},
  "entities": [{"start": 0, "end": 8, "text": "Toshkent", "label": "LOC"}]
}
```

This abbreviated synthetic example illustrates the fields used by the text evaluator, not a released corpus record. Source records additionally contain audio and metadata fields.

Offsets are zero-based, end-exclusive Unicode character indices into `text.normalized_transcript`, not UTF-8 byte indices. Entity surfaces must satisfy `text[start:end] == entity.text`. The labels are PER, LOC, and ORG. Empty entity lists are valid and must not be filtered. Preserve original text, identifiers, boundaries, and language codes.

Evaluation requires exact boundaries and type on each sample. A partial overlap does not count as a true positive. Language macro-F1 is the unweighted average of language-specific entity micro-F1, not the average of entity-type scores. README benchmark scores are manuscript three-seed results; evaluating the single released checkpoint produces a single-run score, not that three-seed mean.

## Manuscript corpus

The manuscript reports 9,976 utterances, 31.17 hours, and 8,780 entity mentions, split 7,981/998/997. The five languages are Kazakh, Kyrgyz, Tajik, Turkmen, and Uzbek. These corpus statistics describe the full benchmark, not just the files included here.

## Rights and privacy

The revised manuscript identifies Mozilla Common Voice v25 as the speech source for all five languages. Underlying source data-use terms and consent requirements continue to apply. This repository's MIT license covers original software; it is not a blanket license for speech, personal information, or third-party datasets. No audio is redistributed in this release. A separate dataset-specific reuse license has not been included; contact the authors before redistribution beyond the supplied research examples.
