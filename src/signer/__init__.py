"""Core package for the SIGNER implementation."""

__all__ = ["SignerConfig", "SignerTokenClassifier", "load_config"]


def __getattr__(name: str):
    """Resolve optional configuration/model dependencies only when requested."""
    if name in {"SignerConfig", "load_config"}:
        from signer.config import SignerConfig, load_config
        return {"SignerConfig": SignerConfig, "load_config": load_config}[name]
    if name == "SignerTokenClassifier":
        from signer.models.model import SignerTokenClassifier

        return SignerTokenClassifier
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
