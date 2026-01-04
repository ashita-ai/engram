"""Base classes for embedding providers.

Embedders convert text to vector representations for semantic search.
All embedders implement async interfaces for non-blocking I/O.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


class Embedder(ABC):
    """Abstract base class for embedding providers.

    All embedders must implement async embed methods and expose
    their vector dimensions for storage configuration.

    Example:
        ```python
        embedder = OpenAIEmbedder()
        vector = await embedder.embed("Hello world")
        vectors = await embedder.embed_batch(["Hello", "World"])
        print(f"Dimensions: {embedder.dimensions}")
        ```
    """

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """Generate embedding vector for a single text.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector as list of floats.
        """
        ...

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embedding vectors for multiple texts.

        More efficient than calling embed() multiple times.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """
        ...

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Get the dimensionality of embedding vectors.

        Returns:
            Number of dimensions in the embedding vector.
        """
        ...
