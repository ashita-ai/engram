"""Base classes for content extraction from episodes.

Extractors use deterministic pattern matching (regex, validators) to identify
content in episode text. Results are returned as simple strings.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from engram.models import Episode


class Extractor(ABC):
    """Abstract base class for content extractors.

    Extractors identify specific patterns in episode content and return
    matched strings. Each extractor focuses on one category of content
    (emails, phones, urls).

    Example:
        ```python
        class EmailExtractor(Extractor):
            name = "email"

            def extract(self, episode: Episode) -> list[str]:
                # Find emails in episode.content
                ...
        ```
    """

    name: str = "base"

    @abstractmethod
    def extract(self, episode: Episode) -> list[str]:
        """Extract content from an episode.

        Args:
            episode: The episode to extract content from.

        Returns:
            List of extracted strings. Empty list if no matches found.
        """
        ...
