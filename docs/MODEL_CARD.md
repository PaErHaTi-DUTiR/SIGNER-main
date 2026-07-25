# SIGNER model card

## Intended use

SIGNER is intended for research on multilingual named entity recognition when
paired speech and transcripts are available during training but only text is
available at deployment. The reference label space is `PER`, `LOC`, and `ORG`.

## Inputs and outputs

- Training input: transcript, paired speech, and transcript-space entity labels.
- Inference input: transcript only.
- Output: BIO tags and exact transcript-space entity spans.

## Architecture

The reference model combines XLM-R-base with a frozen Whisper-small encoder.
Text tokens attend to speech frames, and a gated residual injects the resulting
acoustic context into the text representation. The speech-text and text-only
views share the NER head and are regularized with symmetric KL consistency.

## Limitations

- Performance depends on transcript quality and language coverage in the text
  encoder.
- Soft token-to-frame attention is not forced alignment.
- The current audio-dropout draw is batch-level.
- The reference schema does not cover nested or overlapping entities.
- The model should not be used for high-stakes decisions without
  domain-specific validation.
- Biases in the underlying text/speech encoders and training corpora may
  propagate to predictions.

## Privacy

Do not use private or personally identifying speech without an appropriate
legal basis and consent. Users are responsible for the licensing, privacy, and
retention requirements of their own training audio.

