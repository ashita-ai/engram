"""FastEmbed local embedding provider.

Uses FastEmbed for local, free embeddings without API keys.
Ideal for demos, development, and offline usage.
"""

from __future__ import annotations

import asyncio

from fastembed import TextEmbedding

from .base import Embedder

# Model dimensions for known models
MODEL_DIMENSIONS = {
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-large-en-v1.5": 1024,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
}


class FastEmbedEmbedder(Embedder):
    """FastEmbed local embedding provider.

    Uses FastEmbed for local embeddings without API dependencies.
    Models are downloaded on first use and cached locally.

    Example:
        ```python
        embedder = FastEmbedEmbedder()
        vector = await embedder.embed("Hello world")
        # vector has 384 dimensions (bge-small)
        ```
    """

    def __init__(
        self,
        model: str = "BAAI/bge-small-en-v1.5",
    ) -> None:
        """Initialize FastEmbed embedder.

        Args:
            model: FastEmbed model name. Model will be downloaded on first use.
        """
        self._model_name = model
        self._model: TextEmbedding | None = None
        self._dimensions = MODEL_DIMENSIONS.get(model, 384)

    def _get_model(self) -> TextEmbedding:
        """Lazy load the model on first use."""
        if self._model is None:
            self._model = TextEmbedding(self._model_name)
        return self._model

    async def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector (dimensions depend on model).
        """
        # FastEmbed is sync, run in executor for async compatibility
        loop = asyncio.get_event_loop()
        model = self._get_model()

        def _embed() -> list[float]:
            embeddings = list(model.embed([text]))
            result: list[float] = embeddings[0].tolist()
            return result

        result: list[float] = await loop.run_in_executor(None, _embed)
        return result

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors in same order as input.
        """
        if not texts:
            return []

        loop = asyncio.get_event_loop()
        model = self._get_model()

        def _embed_batch() -> list[list[float]]:
            embeddings = list(model.embed(texts))
            return [e.tolist() for e in embeddings]

        return await loop.run_in_executor(None, _embed_batch)

    @property
    def dimensions(self) -> int:
        """Get embedding dimensions.

        Returns:
            384 for bge-small/MiniLM, 768 for bge-base, 1024 for bge-large.
        """
        return self._dimensions
