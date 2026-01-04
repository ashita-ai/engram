"""Email address extractor.

Uses email-validator library for RFC-compliant email extraction and validation.
"""

from __future__ import annotations

import re

from email_validator import EmailNotValidError, validate_email

from engram.models import Episode, Fact

from .base import Extractor

# Simple pattern to find email candidates (liberal matching)
# The actual validation is done by email-validator
EMAIL_CANDIDATE_PATTERN = re.compile(
    r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
    re.IGNORECASE,
)


class EmailExtractor(Extractor):
    """Extract email addresses from episode content.

    Uses email-validator library for RFC-compliant validation.
    Invalid emails are filtered out, valid ones are normalized.

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
            List of Facts, one per unique valid email found.
        """
        candidates = EMAIL_CANDIDATE_PATTERN.findall(episode.content)
        valid_emails: list[str] = []

        for candidate in candidates:
            try:
                # validate_email normalizes and validates
                result = validate_email(candidate, check_deliverability=False)
                valid_emails.append(result.normalized)
            except EmailNotValidError:
                # Skip invalid emails
                continue

        # Deduplicate while preserving order
        unique_emails = list(dict.fromkeys(valid_emails))

        return [self._create_fact(email, episode) for email in unique_emails]
