"""Cached embedder wrapper with LRU eviction.

Wraps any Embedder implementation to cache embedding results and prevent
redundant computation. Uses content hashing for efficient key storage.
"""

from __future__ import annotations

import hashlib
import logging
from collections import OrderedDict

from .base import Embedder

logger = logging.getLogger(__name__)


def _content_hash(text: str) -> str:
    """Generate SHA256 hash of text for cache key.

    Args:
        text: Text to hash.

    Returns:
        Hex digest of SHA256 hash.
    """
    return hashlib.sha256(text.encode()).hexdigest()


class CachedEmbedder(Embedder):
    """LRU-cached wrapper for any Embedder implementation.

    Caches embedding results using content hashing to prevent redundant
    computation. Thread-safe with OrderedDict-based LRU eviction.

    Example:
        ```python
        base_embedder = OpenAIEmbedder()
        cached = CachedEmbedder(base_embedder, cache_size=1000)

        # First call hits API
        vector1 = await cached.embed("Hello world")

        # Second call returns cached result
        vector2 = await cached.embed("Hello world")
        assert vector1 == vector2  # Same result, no API call
        ```
    """

    def __init__(
        self,
        embedder: Embedder,
        cache_size: int = 1000,
    ) -> None:
        """Initialize cached embedder wrapper.

        Args:
            embedder: Base embedder to wrap.
            cache_size: Maximum number of embeddings to cache.
                Set to 0 to disable caching.
        """
        self._embedder = embedder
        self._cache_size = cache_size
        self._cache: OrderedDict[str, list[float]] = OrderedDict()
        self._hits = 0
        self._misses = 0

        if cache_size > 0:
            logger.debug(f"Initialized embedding cache with size {cache_size}")

    async def embed(self, text: str) -> list[float]:
        """Generate embedding with caching.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector. Returns cached result if available,
            otherwise calls underlying embedder and caches result.
        """
        # Cache disabled
        if self._cache_size == 0:
            return await self._embedder.embed(text)

        # Check cache
        key = _content_hash(text)
        if key in self._cache:
            self._hits += 1
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            logger.debug(
                f"Embedding cache hit (hit_rate={self.hit_rate:.2%}, "
                f"size={len(self._cache)}/{self._cache_size})"
            )
            return self._cache[key]

        # Cache miss - compute embedding
        self._misses += 1
        embedding = await self._embedder.embed(text)

        # Store in cache with LRU eviction
        if len(self._cache) >= self._cache_size:
            # Remove oldest (first) entry
            evicted_key = next(iter(self._cache))
            del self._cache[evicted_key]
            logger.debug(f"Evicted LRU cache entry (size={len(self._cache)})")

        self._cache[key] = embedding
        logger.debug(
            f"Embedding cache miss (hit_rate={self.hit_rate:.2%}, "
            f"size={len(self._cache)}/{self._cache_size})"
        )

        return embedding

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts with caching.

        Checks cache for each text individually. Only uncached texts
        are sent to the underlying embedder for batch processing.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors in same order as input.
        """
        if not texts:
            return []

        # Cache disabled - pass through
        if self._cache_size == 0:
            return await self._embedder.embed_batch(texts)

        # Check which texts are cached
        results: list[list[float] | None] = [None] * len(texts)
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        for i, text in enumerate(texts):
            key = _content_hash(text)
            if key in self._cache:
                self._hits += 1
                self._cache.move_to_end(key)
                results[i] = self._cache[key]
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        # Fetch uncached embeddings
        if uncached_texts:
            self._misses += len(uncached_texts)
            uncached_embeddings = await self._embedder.embed_batch(uncached_texts)

            # Store in cache and fill results
            for idx, text, embedding in zip(
                uncached_indices,
                uncached_texts,
                uncached_embeddings,
                strict=True,
            ):
                # LRU eviction
                if len(self._cache) >= self._cache_size:
                    evicted_key = next(iter(self._cache))
                    del self._cache[evicted_key]

                key = _content_hash(text)
                self._cache[key] = embedding
                results[idx] = embedding

        logger.debug(
            f"Batch embed: {len(texts)} texts, "
            f"{len(uncached_texts)} cache misses, "
            f"hit_rate={self.hit_rate:.2%}"
        )

        # All results should be filled
        return [r for r in results if r is not None]

    @property
    def dimensions(self) -> int:
        """Get embedding dimensions from underlying embedder.

        Returns:
            Number of dimensions in embedding vectors.
        """
        return self._embedder.dimensions

    @property
    def wrapped_embedder(self) -> Embedder:
        """Get the underlying wrapped embedder.

        Returns:
            The base embedder being cached.
        """
        return self._embedder

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate.

        Returns:
            Fraction of requests served from cache (0.0 to 1.0).
            Returns 0.0 if no requests have been made.
        """
        total = self._hits + self._misses
        if total == 0:
            return 0.0
        return self._hits / total

    @property
    def cache_stats(self) -> dict[str, int | float]:
        """Get cache statistics.

        Returns:
            Dictionary with hits, misses, size, and hit_rate.
        """
        return {
            "hits": self._hits,
            "misses": self._misses,
            "size": len(self._cache),
            "max_size": self._cache_size,
            "hit_rate": self.hit_rate,
        }

    def clear_cache(self) -> None:
        """Clear all cached embeddings.

        Resets cache but preserves hit/miss statistics.
        """
        self._cache.clear()
        logger.debug("Cleared embedding cache")
