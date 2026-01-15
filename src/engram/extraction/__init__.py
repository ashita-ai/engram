"""Content extraction from episodes.

This module provides deterministic pattern matching to extract
content from episode text. Each extractor returns a list of strings.

Example:
    ```python
    from engram.extraction import EmailExtractor, PhoneExtractor, URLExtractor
    from engram.models import Episode

    episode = Episode(content="Email me at user@example.com", ...)

    emails = EmailExtractor().extract(episode)  # ["user@example.com"]
    phones = PhoneExtractor().extract(episode)  # []
    urls = URLExtractor().extract(episode)  # []
    ```
"""

from .base import Extractor
from .email import EmailExtractor
from .phone import PhoneExtractor
from .url import URLExtractor

__all__ = [
    # Base class
    "Extractor",
    # Extractors
    "EmailExtractor",
    "PhoneExtractor",
    "URLExtractor",
]
