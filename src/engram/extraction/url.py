"""URL extractor.

Uses regex pattern matching to extract URLs from episode content.
Supports http, https, and bare domain formats.
"""

from __future__ import annotations

import re

from engram.models import Episode, Fact

from .base import Extractor

# URL pattern - matches http://, https://, and www. prefixes
# Captures path, query string, and fragments
URL_PATTERN = re.compile(
    r"(?:https?://|www\.)"  # Protocol or www
    r"[A-Za-z0-9]"  # Must start with alphanumeric
    r"[A-Za-z0-9.-]*"  # Domain characters
    r"\.[A-Za-z]{2,}"  # TLD
    r"(?:[/?#][^\s<>\"')*\]]*)?",  # Optional path/query/fragment
    re.IGNORECASE,
)


def normalize_url(url: str) -> str:
    """Normalize URL to consistent format.

    - Adds https:// if missing protocol
    - Lowercases domain portion
    - Removes trailing punctuation that was incorrectly captured

    Args:
        url: Raw URL string.

    Returns:
        Normalized URL.
    """
    # Remove trailing punctuation that might have been captured
    url = url.rstrip(".,;:!?")

    # Add protocol if missing
    if url.lower().startswith("www."):
        url = "https://" + url

    # Lowercase the protocol and domain
    # Split at first / after protocol
    if "://" in url:
        protocol, rest = url.split("://", 1)
        if "/" in rest:
            domain, path = rest.split("/", 1)
            url = f"{protocol.lower()}://{domain.lower()}/{path}"
        else:
            url = f"{protocol.lower()}://{rest.lower()}"

    return url


class URLExtractor(Extractor):
    """Extract URLs from episode content.

    Supports various URL formats:
    - Full URLs: https://example.com/path
    - HTTP URLs: http://example.com
    - WWW URLs: www.example.com (normalized to https://)

    URLs are normalized for consistent storage.

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
            List of Facts, one per unique URL found.
        """
        matches = URL_PATTERN.findall(episode.content)

        # Normalize and deduplicate
        normalized = [normalize_url(url) for url in matches]
        unique_urls = list(dict.fromkeys(normalized))

        return [self._create_fact(url, episode) for url in unique_urls]
