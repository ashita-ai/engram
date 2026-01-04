"""URL extractor.

Uses validators library for URL validation combined with regex for extraction.
"""

from __future__ import annotations

import re
from urllib.parse import urlparse

import validators

from engram.models import Episode, Fact

from .base import Extractor

# Pattern to find URL candidates (liberal matching)
# Actual validation is done by validators library
URL_CANDIDATE_PATTERN = re.compile(
    r"(?:https?://|www\.)[^\s<>\"')\]]+",
    re.IGNORECASE,
)


def normalize_url(url: str) -> str:
    """Normalize URL to consistent format.

    - Adds https:// if missing protocol
    - Lowercases scheme and domain
    - Removes trailing punctuation

    Args:
        url: Raw URL string.

    Returns:
        Normalized URL.
    """
    # Remove trailing punctuation
    url = url.rstrip(".,;:!?")

    # Add protocol if missing
    if url.lower().startswith("www."):
        url = "https://" + url

    # Parse and normalize
    parsed = urlparse(url)
    if parsed.scheme and parsed.netloc:
        # Lowercase scheme and netloc, keep path as-is
        normalized = f"{parsed.scheme.lower()}://{parsed.netloc.lower()}"
        if parsed.path:
            normalized += parsed.path
        if parsed.query:
            normalized += f"?{parsed.query}"
        if parsed.fragment:
            normalized += f"#{parsed.fragment}"
        return normalized

    return url


class URLExtractor(Extractor):
    """Extract URLs from episode content.

    Uses validators library for URL validation.
    Supports http, https, and www prefixes.

    Example:
        ```python
        extractor = URLExtractor()
        facts = extractor.extract(episode)
        # facts[0].content = "https://example.com/page"
        # facts[0].category = "url"
        ```
    """

    name: str = "url"
    category: str = "url"

    def extract(self, episode: Episode) -> list[Fact]:
        """Extract URLs from episode content.

        Args:
            episode: Episode containing text to search.

        Returns:
            List of Facts, one per unique valid URL found.
        """
        candidates = URL_CANDIDATE_PATTERN.findall(episode.content)
        valid_urls: list[str] = []

        for candidate in candidates:
            normalized = normalize_url(candidate)

            # Validate the URL
            if validators.url(normalized):
                valid_urls.append(normalized)

        # Deduplicate while preserving order
        unique_urls = list(dict.fromkeys(valid_urls))

        return [self._create_fact(url, episode) for url in unique_urls]
