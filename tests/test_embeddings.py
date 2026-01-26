"""Unit tests for Engram embeddings layer."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from engram.config import Settings
from engram.embeddings import (
    Embedder,
    FastEmbedEmbedder,
    OpenAIEmbedder,
    get_embedder,
)
from engram.embeddings.cached import CachedEmbedder


class TestEmbedderBase:
    """Tests for Embedder abstract base class."""

    def test_embedder_is_abstract(self):
        """Embedder cannot be instantiated directly."""
        with pytest.raises(TypeError):
            Embedder()  # type: ignore[abstract]


class TestOpenAIEmbedder:
    """Tests for OpenAIEmbedder."""

    def test_init_default_model(self):
        """Should use text-embedding-3-small by default."""
        # Use dummy key since OpenAI SDK requires it
        embedder = OpenAIEmbedder(api_key="sk-test-dummy-key")
        assert embedder.model == "text-embedding-3-small"
        assert embedder.dimensions == 1536

    def test_init_large_model(self):
        """Should support text-embedding-3-large."""
        embedder = OpenAIEmbedder(model="text-embedding-3-large", api_key="sk-test-dummy-key")
        assert embedder.model == "text-embedding-3-large"
        assert embedder.dimensions == 3072

    def test_init_ada_model(self):
        """Should support legacy ada-002 model."""
        embedder = OpenAIEmbedder(model="text-embedding-ada-002", api_key="sk-test-dummy-key")
        assert embedder.model == "text-embedding-ada-002"
        assert embedder.dimensions == 1536

    def test_init_with_api_key(self):
        """Should accept API key."""
        embedder = OpenAIEmbedder(api_key="sk-test-key")
        assert embedder._client.api_key == "sk-test-key"

    @pytest.mark.asyncio
    async def test_embed_single(self):
        """Should embed single text."""
        embedder = OpenAIEmbedder(api_key="sk-test-dummy-key")

        # Mock the API response
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]

        with patch.object(
            embedder._client.embeddings, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = mock_response
            result = await embedder.embed("Hello world")

        assert result == [0.1, 0.2, 0.3]
        mock_create.assert_called_once_with(
            model="text-embedding-3-small",
            input="Hello world",
        )

    @pytest.mark.asyncio
    async def test_embed_batch(self):
        """Should embed multiple texts."""
        embedder = OpenAIEmbedder(api_key="sk-test-dummy-key")

        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=[0.1, 0.2]),
            MagicMock(embedding=[0.3, 0.4]),
        ]

        with patch.object(
            embedder._client.embeddings, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = mock_response
            result = await embedder.embed_batch(["Hello", "World"])

        assert result == [[0.1, 0.2], [0.3, 0.4]]
        mock_create.assert_called_once_with(
            model="text-embedding-3-small",
            input=["Hello", "World"],
        )

    @pytest.mark.asyncio
    async def test_embed_batch_empty(self):
        """Should return empty list for empty input."""
        embedder = OpenAIEmbedder(api_key="sk-test-dummy-key")
        result = await embedder.embed_batch([])
        assert result == []


class TestFastEmbedEmbedder:
    """Tests for FastEmbedEmbedder."""

    def test_init_default_model(self):
        """Should use bge-small by default."""
        embedder = FastEmbedEmbedder()
        assert embedder._model_name == "BAAI/bge-small-en-v1.5"
        assert embedder.dimensions == 384

    def test_init_base_model(self):
        """Should support bge-base model."""
        embedder = FastEmbedEmbedder(model="BAAI/bge-base-en-v1.5")
        assert embedder._model_name == "BAAI/bge-base-en-v1.5"
        assert embedder.dimensions == 768

    def test_init_large_model(self):
        """Should support bge-large model."""
        embedder = FastEmbedEmbedder(model="BAAI/bge-large-en-v1.5")
        assert embedder.dimensions == 1024

    def test_lazy_model_loading(self):
        """Model should not load until first use."""
        embedder = FastEmbedEmbedder()
        assert embedder._model is None

    @pytest.mark.asyncio
    async def test_embed_single(self):
        """Should embed single text."""
        embedder = FastEmbedEmbedder()

        # Mock the model
        import numpy as np

        mock_model = MagicMock()
        mock_model.embed.return_value = iter([np.array([0.1, 0.2, 0.3])])
        embedder._model = mock_model

        result = await embedder.embed("Hello world")

        assert result == [0.1, 0.2, 0.3]
        mock_model.embed.assert_called_once_with(["Hello world"])

    @pytest.mark.asyncio
    async def test_embed_batch(self):
        """Should embed multiple texts."""
        embedder = FastEmbedEmbedder()

        import numpy as np

        mock_model = MagicMock()
        mock_model.embed.return_value = iter(
            [
                np.array([0.1, 0.2]),
                np.array([0.3, 0.4]),
            ]
        )
        embedder._model = mock_model

        result = await embedder.embed_batch(["Hello", "World"])

        assert result == [[0.1, 0.2], [0.3, 0.4]]
        mock_model.embed.assert_called_once_with(["Hello", "World"])

    @pytest.mark.asyncio
    async def test_embed_batch_empty(self):
        """Should return empty list for empty input."""
        embedder = FastEmbedEmbedder()
        result = await embedder.embed_batch([])
        assert result == []


class TestGetEmbedder:
    """Tests for get_embedder factory function."""

    def test_default_returns_openai(self):
        """Default settings should return OpenAI embedder (wrapped in cache)."""
        # Explicitly set provider to openai to override any .env settings
        settings = Settings(openai_api_key="sk-test-dummy-key", embedding_provider="openai")
        embedder = get_embedder(settings)
        # With caching enabled (default), get_embedder returns CachedEmbedder
        if isinstance(embedder, CachedEmbedder):
            assert isinstance(embedder.wrapped_embedder, OpenAIEmbedder)
        else:
            assert isinstance(embedder, OpenAIEmbedder)

    def test_openai_provider(self):
        """Should create OpenAI embedder for openai provider."""
        settings = Settings(embedding_provider="openai", openai_api_key="sk-test-dummy-key")
        embedder = get_embedder(settings)
        if isinstance(embedder, CachedEmbedder):
            assert isinstance(embedder.wrapped_embedder, OpenAIEmbedder)
        else:
            assert isinstance(embedder, OpenAIEmbedder)

    def test_fastembed_provider(self):
        """Should create FastEmbed embedder for fastembed provider."""
        settings = Settings(embedding_provider="fastembed")
        embedder = get_embedder(settings)
        if isinstance(embedder, CachedEmbedder):
            assert isinstance(embedder.wrapped_embedder, FastEmbedEmbedder)
        else:
            assert isinstance(embedder, FastEmbedEmbedder)

    def test_openai_with_custom_model(self):
        """Should pass model from settings to OpenAI embedder."""
        settings = Settings(
            embedding_provider="openai",
            embedding_model="text-embedding-3-large",
            openai_api_key="sk-test-dummy-key",
        )
        embedder = get_embedder(settings)
        # Unwrap if cached
        base_embedder = (
            embedder.wrapped_embedder if isinstance(embedder, CachedEmbedder) else embedder
        )
        assert isinstance(base_embedder, OpenAIEmbedder)
        assert base_embedder.model == "text-embedding-3-large"
        assert base_embedder.dimensions == 3072

    def test_fastembed_with_custom_model(self):
        """Should pass model from settings to FastEmbed embedder."""
        settings = Settings(
            embedding_provider="fastembed",
            embedding_model="BAAI/bge-base-en-v1.5",
        )
        embedder = get_embedder(settings)
        # Unwrap if cached
        base_embedder = (
            embedder.wrapped_embedder if isinstance(embedder, CachedEmbedder) else embedder
        )
        assert isinstance(base_embedder, FastEmbedEmbedder)
        assert base_embedder._model_name == "BAAI/bge-base-en-v1.5"
        assert base_embedder.dimensions == 768

    def test_unknown_provider_raises(self):
        """Should raise ValueError for unknown provider."""
        # Create settings with valid provider first, then modify
        settings = Settings()
        # Hack to bypass validation
        object.__setattr__(settings, "embedding_provider", "unknown")

        with pytest.raises(ValueError, match="Unknown embedding provider"):
            get_embedder(settings)


class TestEmbedderDimensions:
    """Tests for embedder dimension consistency."""

    def test_openai_small_dimensions(self):
        """OpenAI small model should have 1536 dimensions."""
        embedder = OpenAIEmbedder(model="text-embedding-3-small", api_key="sk-test-dummy-key")
        assert embedder.dimensions == 1536

    def test_openai_large_dimensions(self):
        """OpenAI large model should have 3072 dimensions."""
        embedder = OpenAIEmbedder(model="text-embedding-3-large", api_key="sk-test-dummy-key")
        assert embedder.dimensions == 3072

    def test_fastembed_small_dimensions(self):
        """FastEmbed small model should have 384 dimensions."""
        embedder = FastEmbedEmbedder(model="BAAI/bge-small-en-v1.5")
        assert embedder.dimensions == 384

    def test_fastembed_base_dimensions(self):
        """FastEmbed base model should have 768 dimensions."""
        embedder = FastEmbedEmbedder(model="BAAI/bge-base-en-v1.5")
        assert embedder.dimensions == 768
