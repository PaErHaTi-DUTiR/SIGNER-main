"""Core package for the SIGNER implementation."""

from signer.config import SignerConfig, load_config

__all__ = ["SignerConfig", "SignerTokenClassifier", "load_config"]


def __getattr__(name: str):
    """Load the model lazily so data/config utilities remain lightweight."""
    if name == "SignerTokenClassifier":
        from signer.models.model import SignerTokenClassifier

        return SignerTokenClassifier
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
