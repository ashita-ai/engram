"""OpenAI embedding provider.

Uses OpenAI's text-embedding models via the official SDK.
"""

from __future__ import annotations

from openai import AsyncOpenAI

from .base import Embedder

# Model dimensions for known models
MODEL_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class OpenAIEmbedder(Embedder):
    """OpenAI embedding provider.

    Uses AsyncOpenAI client for non-blocking API calls.
    Supports text-embedding-3-small (recommended), text-embedding-3-large,
    and text-embedding-ada-002 (legacy).

    Example:
        ```python
        embedder = OpenAIEmbedder()
        vector = await embedder.embed("Hello world")
        # vector has 1536 dimensions
        ```
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
    ) -> None:
        """Initialize OpenAI embedder.

        Args:
            model: OpenAI embedding model name.
            api_key: OpenAI API key. If None, uses OPENAI_API_KEY env var.
        """
        self.model = model
        self._client = AsyncOpenAI(api_key=api_key)
        self._dimensions = MODEL_DIMENSIONS.get(model, 1536)

    async def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector (1536 or 3072 dimensions depending on model).
        """
        response = await self._client.embeddings.create(
            model=self.model,
            input=text,
        )
        return list(response.data[0].embedding)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts.

        OpenAI supports batching up to 2048 texts per request.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors in same order as input.
        """
        if not texts:
            return []

        response = await self._client.embeddings.create(
            model=self.model,
            input=texts,
        )
        return [list(d.embedding) for d in response.data]

    @property
    def dimensions(self) -> int:
        """Get embedding dimensions.

        Returns:
            1536 for text-embedding-3-small/ada-002, 3072 for text-embedding-3-large.
        """
        return self._dimensions
