"""Shared constants for the SIGNER codebase."""

from __future__ import annotations

LANGUAGES = ("kk", "ky", "tg", "uz", "tk", "zh", "de", "es", "fr", "nl")
LANGUAGE_TO_ID = {language: index for index, language in enumerate(LANGUAGES)}
ID_TO_LANGUAGE = {index: language for language, index in LANGUAGE_TO_ID.items()}

LABELS = ("PER", "LOC", "ORG", "MISC")
VIEW_NAMES = ("text-only", "speech-only", "speech-text")
VIEW_NAME_TO_ID = {name: index for index, name in enumerate(VIEW_NAMES)}
VIEW_ID_TO_NAME = {index: name for name, index in VIEW_NAME_TO_ID.items()}

BIO_TAGS = ["O"]
for label in LABELS:
    BIO_TAGS.extend((f"B-{label}", f"I-{label}"))
TAG_TO_ID = {tag: index for index, tag in enumerate(BIO_TAGS)}
ID_TO_TAG = {index: tag for tag, index in TAG_TO_ID.items()}

IGNORE_INDEX = -100
