"""Embedding providers for Engram.

This module provides embedding generation for semantic search.
Supports OpenAI (cloud) and FastEmbed (local) providers.

Example:
    ```python
    from engram.embeddings import get_embedder
    from engram.config import Settings

    # Use default provider from settings
    embedder = get_embedder()

    # Or specify provider
    embedder = get_embedder(Settings(embedding_provider="fastembed"))

    # Generate embeddings
    vector = await embedder.embed("Hello world")
    vectors = await embedder.embed_batch(["Hello", "World"])
    ```
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import Embedder
from .fastembed import FastEmbedEmbedder
from .openai import OpenAIEmbedder

if TYPE_CHECKING:
    from engram.config import Settings


def get_embedder(settings: Settings | None = None) -> Embedder:
    """Create an embedder based on settings.

    Factory function that returns the appropriate embedder
    based on the configured provider.

    Args:
        settings: Optional settings. Uses default Settings() if None.

    Returns:
        Configured Embedder instance.

    Raises:
        ValueError: If embedding provider is unknown.

    Example:
        ```python
        # Use default (OpenAI)
        embedder = get_embedder()

        # Use FastEmbed for local/free embeddings
        from engram.config import Settings
        embedder = get_embedder(Settings(embedding_provider="fastembed"))
        ```
    """
    if settings is None:
        from engram.config import Settings

        settings = Settings()

    provider = settings.embedding_provider

    if provider == "openai":
        return OpenAIEmbedder(
            model=settings.embedding_model,
            api_key=settings.openai_api_key,
        )
    elif provider == "fastembed":
        return FastEmbedEmbedder(
            model=settings.embedding_model,
        )
    else:
        raise ValueError(f"Unknown embedding provider: {provider}")


__all__ = [
    "Embedder",
    "OpenAIEmbedder",
    "FastEmbedEmbedder",
    "get_embedder",
]
