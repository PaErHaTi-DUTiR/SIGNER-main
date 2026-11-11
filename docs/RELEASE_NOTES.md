# Release 0.2.0 — 2026-11-11

This update adds the retained character-end head and its TRAIN-only vocabulary,
strict text-only loading of the single checkpoint, original-character scoring,
separate head training, updated figures, and manuscript-based result tables.
The manuscript TeX remains a local reference and is excluded from Git.

The author-provided model bundle was verified tensor by tensor against the
retained seed42 NER (399 tensors) and character head (20 tensors). Its SHA-256
and byte count are recorded in `model_checkpoint/manifest.json`.

The public data file has been replaced with the author-confirmed CA-STNER
TEST set: 997 records, all with `split: test`, matching the manuscript's
language-level TEST counts. Its new checksum is recorded in the data README
and enforced by the release validator. Earlier validation of a different
998-record file does not constitute evaluation of this TEST release; that
obsolete score is not reported as a current TEST result.

Seven lightweight release/scoring tests passed, together with Python syntax
compilation and the secret scan. The author-confirmed TEST file and both PNG
figures are preserved unchanged. No GPU was used for this release validation.

The1.48GB checkpoint remains a local author-provided file; no hosted download
is claimed until a separate binary upload is completed.
