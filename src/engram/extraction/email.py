"""Email address extractor.

Uses regex pattern matching with validation to extract email addresses
from episode content.
"""

from __future__ import annotations

import re

from engram.models import Episode, Fact

from .base import Extractor

# RFC 5322 simplified pattern - catches most valid emails
# Allows: local-part@domain.tld
# Local part: alphanumeric, dots, hyphens, underscores, plus signs
# Domain: alphanumeric, dots, hyphens
EMAIL_PATTERN = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    re.IGNORECASE,
)


class EmailExtractor(Extractor):
    """Extract email addresses from episode content.

    Uses RFC 5322 simplified regex pattern to find email addresses.
    Validates basic structure but does not verify deliverability.

    Example:
        ```python
        extractor = EmailExtractor()
        facts = extractor.extract(episode)
        # facts[0].content = "user@example.com"
        # facts[0].category = "email"
        ```
    """

    name: str = "email"
    category: str = "email"

    def extract(self, episode: Episode) -> list[Fact]:
        """Extract email addresses from episode content.

        Args:
            episode: Episode containing text to search.

        Returns:
            List of Facts, one per unique email found.
        """
        matches = EMAIL_PATTERN.findall(episode.content)

        # Deduplicate and normalize to lowercase
        unique_emails = list(dict.fromkeys(email.lower() for email in matches))

        return [self._create_fact(email, episode) for email in unique_emails]
