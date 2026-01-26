"""Tests for cached embedder wrapper."""

from __future__ import annotations

import pytest

from engram.embeddings.base import Embedder
from engram.embeddings.cached import CachedEmbedder, _content_hash


class MockEmbedder(Embedder):
    """Mock embedder for testing cache behavior."""

    def __init__(self, dimensions: int = 384) -> None:
        """Initialize mock embedder.

        Args:
            dimensions: Number of dimensions in vectors.
        """
        self._dimensions = dimensions
        self.embed_calls = 0
        self.embed_batch_calls = 0

    async def embed(self, text: str) -> list[float]:
        """Generate mock embedding.

        Args:
            text: Text to embed.

        Returns:
            Mock vector based on text length.
        """
        self.embed_calls += 1
        # Simple deterministic vector based on text
        return [float(len(text) % 256) / 255.0] * self._dimensions

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate mock embeddings for batch.

        Args:
            texts: List of texts to embed.

        Returns:
            List of mock vectors.
        """
        self.embed_batch_calls += 1
        # Don't call embed() to avoid double-counting
        return [[float(len(text) % 256) / 255.0] * self._dimensions for text in texts]

    @property
    def dimensions(self) -> int:
        """Get vector dimensions.

        Returns:
            Number of dimensions.
        """
        return self._dimensions


class TestContentHash:
    """Tests for content hashing function."""

    def test_deterministic(self) -> None:
        """Hash should be deterministic for same input."""
        text = "Hello world"
        hash1 = _content_hash(text)
        hash2 = _content_hash(text)
        assert hash1 == hash2

    def test_different_inputs(self) -> None:
        """Different inputs should produce different hashes."""
        hash1 = _content_hash("Hello")
        hash2 = _content_hash("World")
        assert hash1 != hash2

    def test_case_sensitive(self) -> None:
        """Hash should be case-sensitive."""
        hash1 = _content_hash("Hello")
        hash2 = _content_hash("hello")
        assert hash1 != hash2

    def test_hex_format(self) -> None:
        """Hash should be a hex string."""
        result = _content_hash("test")
        assert isinstance(result, str)
        assert len(result) == 64  # SHA256 produces 64 hex chars
        assert all(c in "0123456789abcdef" for c in result)


class TestCachedEmbedder:
    """Tests for CachedEmbedder wrapper."""

    def test_dimensions_passthrough(self) -> None:
        """Dimensions should pass through from base embedder."""
        base = MockEmbedder(dimensions=768)
        cached = CachedEmbedder(base, cache_size=10)
        assert cached.dimensions == 768

    @pytest.mark.asyncio
    async def test_first_call_misses_cache(self) -> None:
        """First call should miss cache and call base embedder."""
        base = MockEmbedder()
        cached = CachedEmbedder(base, cache_size=10)

        result = await cached.embed("Hello")

        assert base.embed_calls == 1
        assert cached._hits == 0
        assert cached._misses == 1
        assert len(result) == 384

    @pytest.mark.asyncio
    async def test_second_call_hits_cache(self) -> None:
        """Second call with same text should hit cache."""
        base = MockEmbedder()
        cached = CachedEmbedder(base, cache_size=10)

        result1 = await cached.embed("Hello")
        result2 = await cached.embed("Hello")

        # Only one call to base embedder
        assert base.embed_calls == 1
        assert cached._hits == 1
        assert cached._misses == 1
        assert result1 == result2

    @pytest.mark.asyncio
    async def test_different_texts_miss_cache(self) -> None:
        """Different texts should each miss cache initially."""
        base = MockEmbedder()
        cached = CachedEmbedder(base, cache_size=10)

        await cached.embed("Hello")
        await cached.embed("World")

        assert base.embed_calls == 2
        assert cached._hits == 0
        assert cached._misses == 2

    @pytest.mark.asyncio
    async def test_lru_eviction(self) -> None:
        """Oldest entry should be evicted when cache is full."""
        base = MockEmbedder()
        cached = CachedEmbedder(base, cache_size=2)

        # Fill cache
        await cached.embed("First")
        await cached.embed("Second")

        # This should evict "First" (oldest)
        await cached.embed("Third")

        # "First" should miss cache (evicted)
        await cached.embed("First")
        assert base.embed_calls == 4  # First, Second, Third, First (again)

        # "Second" should be evicted now (by Third and First)
        await cached.embed("Second")
        assert base.embed_calls == 5  # Second evicted, needs refetch

    @pytest.mark.asyncio
    async def test_cache_disabled_with_zero_size(self) -> None:
        """Cache should be disabled when size is 0."""
        base = MockEmbedder()
        cached = CachedEmbedder(base, cache_size=0)

        await cached.embed("Hello")
        await cached.embed("Hello")

        # Both calls should go to base embedder
        assert base.embed_calls == 2
        assert len(cached._cache) == 0

    @pytest.mark.asyncio
    async def test_hit_rate_calculation(self) -> None:
        """Hit rate should be calculated correctly."""
        base = MockEmbedder()
        cached = CachedEmbedder(base, cache_size=10)

        # No requests yet
        assert cached.hit_rate == 0.0

        # 1 miss
        await cached.embed("Hello")
        assert cached.hit_rate == 0.0

        # 1 hit
        await cached.embed("Hello")
        assert cached.hit_rate == 0.5

        # 2 hits
        await cached.embed("Hello")
        assert cached.hit_rate == pytest.approx(2 / 3)

    @pytest.mark.asyncio
    async def test_cache_stats(self) -> None:
        """Cache stats should be accurate."""
        base = MockEmbedder()
        cached = CachedEmbedder(base, cache_size=10)

        await cached.embed("Hello")
        await cached.embed("Hello")
        await cached.embed("World")

        stats = cached.cache_stats
        assert stats["hits"] == 1
        assert stats["misses"] == 2
        assert stats["size"] == 2
        assert stats["max_size"] == 10
        assert stats["hit_rate"] == pytest.approx(1 / 3)

    @pytest.mark.asyncio
    async def test_clear_cache(self) -> None:
        """Clear cache should remove all entries but preserve stats."""
        base = MockEmbedder()
        cached = CachedEmbedder(base, cache_size=10)

        await cached.embed("Hello")
        await cached.embed("Hello")
        assert cached._hits == 1

        cached.clear_cache()

        assert len(cached._cache) == 0
        assert cached._hits == 1  # Stats preserved
        assert cached._misses == 1

        # Next call should miss
        await cached.embed("Hello")
        assert cached._misses == 2

    @pytest.mark.asyncio
    async def test_embed_batch_all_cached(self) -> None:
        """Batch embed should use cache for all texts if available."""
        base = MockEmbedder()
        cached = CachedEmbedder(base, cache_size=10)

        texts = ["Hello", "World"]

        # Prime cache
        for text in texts:
            await cached.embed(text)

        # Batch should hit cache for all
        results = await cached.embed_batch(texts)

        assert len(results) == 2
        assert base.embed_calls == 2  # Only from priming
        assert base.embed_batch_calls == 0

    @pytest.mark.asyncio
    async def test_embed_batch_partial_cache(self) -> None:
        """Batch embed should only fetch uncached texts."""
        base = MockEmbedder()
        cached = CachedEmbedder(base, cache_size=10)

        # Prime cache with one text
        await cached.embed("Hello")

        # Batch with one cached, one uncached
        results = await cached.embed_batch(["Hello", "World"])

        assert len(results) == 2
        assert base.embed_calls == 1  # From priming
        assert base.embed_batch_calls == 1  # Only for ["World"]

    @pytest.mark.asyncio
    async def test_embed_batch_none_cached(self) -> None:
        """Batch embed should fetch all if none cached."""
        base = MockEmbedder()
        cached = CachedEmbedder(base, cache_size=10)

        results = await cached.embed_batch(["Hello", "World"])

        assert len(results) == 2
        assert base.embed_batch_calls == 1

    @pytest.mark.asyncio
    async def test_embed_batch_empty(self) -> None:
        """Batch embed should handle empty list."""
        base = MockEmbedder()
        cached = CachedEmbedder(base, cache_size=10)

        results = await cached.embed_batch([])

        assert results == []
        assert base.embed_batch_calls == 0

    @pytest.mark.asyncio
    async def test_embed_batch_cache_disabled(self) -> None:
        """Batch embed should pass through when cache disabled."""
        base = MockEmbedder()
        cached = CachedEmbedder(base, cache_size=0)

        await cached.embed_batch(["Hello", "World"])
        await cached.embed_batch(["Hello", "World"])

        assert base.embed_batch_calls == 2  # Both pass through

    @pytest.mark.asyncio
    async def test_embed_batch_lru_eviction(self) -> None:
        """Batch embed should apply LRU eviction."""
        base = MockEmbedder()
        cached = CachedEmbedder(base, cache_size=2)

        # Fill cache
        await cached.embed_batch(["First", "Second"])
        assert len(cached._cache) == 2
        assert base.embed_batch_calls == 1

        # This should evict "First" (oldest)
        await cached.embed_batch(["Third"])
        assert len(cached._cache) == 2
        assert base.embed_batch_calls == 2

        # "First" should be evicted, "Second" and "Third" are cached
        results = await cached.embed_batch(["First", "Second"])
        assert len(results) == 2
        # Only "First" needs to be fetched
        assert base.embed_batch_calls == 3

    @pytest.mark.asyncio
    async def test_case_sensitivity(self) -> None:
        """Cache should be case-sensitive."""
        base = MockEmbedder()
        cached = CachedEmbedder(base, cache_size=10)

        await cached.embed("Hello")
        await cached.embed("hello")

        assert base.embed_calls == 2  # Different texts, both miss
        assert len(cached._cache) == 2

    @pytest.mark.asyncio
    async def test_whitespace_sensitivity(self) -> None:
        """Cache should be sensitive to whitespace."""
        base = MockEmbedder()
        cached = CachedEmbedder(base, cache_size=10)

        await cached.embed("Hello World")
        await cached.embed("Hello  World")  # Extra space

        assert base.embed_calls == 2
        assert len(cached._cache) == 2


@pytest.mark.asyncio
async def test_get_embedder_with_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """get_embedder should return cached embedder when enabled."""
    from engram.config import Settings
    from engram.embeddings import get_embedder

    # Mock API key
    monkeypatch.setenv("ENGRAM_OPENAI_API_KEY", "sk-test")

    settings = Settings(
        embedding_provider="openai",
        embedding_cache_enabled=True,
        embedding_cache_size=100,
    )

    embedder = get_embedder(settings)

    assert isinstance(embedder, CachedEmbedder)
    assert embedder._cache_size == 100


@pytest.mark.asyncio
async def test_get_embedder_without_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """get_embedder should return base embedder when cache disabled."""
    from engram.config import Settings
    from engram.embeddings import get_embedder
    from engram.embeddings.openai import OpenAIEmbedder

    # Mock API key
    monkeypatch.setenv("ENGRAM_OPENAI_API_KEY", "sk-test")

    settings = Settings(
        embedding_provider="openai",
        embedding_cache_enabled=False,
    )

    embedder = get_embedder(settings)

    assert isinstance(embedder, OpenAIEmbedder)
    assert not isinstance(embedder, CachedEmbedder)


@pytest.mark.asyncio
async def test_get_embedder_cache_size_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """get_embedder should return base embedder when cache size is 0."""
    from engram.config import Settings
    from engram.embeddings import get_embedder
    from engram.embeddings.openai import OpenAIEmbedder

    # Mock API key
    monkeypatch.setenv("ENGRAM_OPENAI_API_KEY", "sk-test")

    settings = Settings(
        embedding_provider="openai",
        embedding_cache_enabled=True,
        embedding_cache_size=0,  # Disabled via size
    )

    embedder = get_embedder(settings)

    assert isinstance(embedder, OpenAIEmbedder)
    assert not isinstance(embedder, CachedEmbedder)
