# Release 0.2.0 — 2026-11-11

This update adds the retained character-end head and its TRAIN-only vocabulary,
strict text-only loading of the single checkpoint, original-character scoring,
separate head training, updated figures, and manuscript-based result tables.
The manuscript TeX remains a local reference and is excluded from Git.

The author-provided model bundle was verified tensor by tensor against the
retained seed42 NER (399 tensors) and character head (20 tensors). Its SHA-256
and byte count are recorded in `model_checkpoint/manifest.json`.

A CPU-only reload and full prediction pass on all998 supplied public examples
produced866 predictions,866 reference spans, and649 exact matches:
micro-F1 **74.942263%**. This reproduces the retained seed42 **DEV** result,
not the manuscript's three-seed held-out TEST mean. The public filename is
`test.jsonl`, but all source records preserve `split: dev`.

Seven lightweight release/scoring tests passed, together with Python syntax
compilation and the secret scan. The supplied example file and both PNG
figures are preserved unchanged. No GPU was used for this release validation.

The1.48GB checkpoint remains a local author-provided file; no hosted download
is claimed until a separate binary upload is completed.
