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
from .cached import CachedEmbedder
from .fastembed import FastEmbedEmbedder
from .openai import OpenAIEmbedder

if TYPE_CHECKING:
    from engram.config import Settings


def get_embedder(settings: Settings | None = None) -> Embedder:
    """Create an embedder based on settings.

    Factory function that returns the appropriate embedder
    based on the configured provider. Automatically wraps
    the embedder with a cache if enabled in settings.

    Args:
        settings: Optional settings. Uses default Settings() if None.

    Returns:
        Configured Embedder instance, optionally wrapped with cache.

    Raises:
        ValueError: If embedding provider is unknown.

    Example:
        ```python
        # Use default (OpenAI) with caching enabled
        embedder = get_embedder()

        # Use FastEmbed for local/free embeddings
        from engram.config import Settings
        embedder = get_embedder(Settings(embedding_provider="fastembed"))

        # Disable cache
        embedder = get_embedder(Settings(embedding_cache_enabled=False))
        ```
    """
    if settings is None:
        from engram.config import Settings

        settings = Settings()

    provider = settings.embedding_provider

    # Create base embedder
    base_embedder: Embedder
    if provider == "openai":
        base_embedder = OpenAIEmbedder(
            model=settings.embedding_model,
            api_key=settings.openai_api_key,
        )
    elif provider == "fastembed":
        base_embedder = FastEmbedEmbedder(
            model=settings.embedding_model,
        )
    else:
        msg = f"Unknown embedding provider: {provider}"
        raise ValueError(msg)

    # Wrap with cache if enabled
    if settings.embedding_cache_enabled and settings.embedding_cache_size > 0:
        return CachedEmbedder(
            embedder=base_embedder,
            cache_size=settings.embedding_cache_size,
        )

    return base_embedder


__all__ = [
    "Embedder",
    "OpenAIEmbedder",
    "FastEmbedEmbedder",
    "CachedEmbedder",
    "get_embedder",
]
