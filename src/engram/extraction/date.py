"""Date extractor.

Uses dateparser library for robust natural language date parsing.
Only extracts HIGH-CONFIDENCE absolute dates (with explicit year).

Relative dates like "tomorrow", "next week", "3:30 PM" are left for
LLM consolidation, which has full context to resolve them correctly.
"""

from __future__ import annotations

import re

from dateparser.search import search_dates  # type: ignore[import-untyped]

from engram.models import Episode, Fact

from .base import Extractor

# Patterns that indicate a relative/ambiguous date - skip these
_RELATIVE_PATTERNS = re.compile(
    r"\b(today|tomorrow|yesterday|"
    r"next\s+\w+|last\s+\w+|this\s+\w+|"
    r"ago|from\s+now|"
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    re.IGNORECASE,
)

# Pattern to detect if text contains an explicit year (1900-2099)
_HAS_YEAR = re.compile(r"\b(19|20)\d{2}\b")


class DateExtractor(Extractor):
    """Extract absolute dates from episode content.

    Only extracts HIGH-CONFIDENCE dates:
    - Dates with explicit years (2024-01-15, January 15, 2024)
    - ISO format dates
    - Unambiguous date formats

    SKIPS relative/ambiguous dates:
    - "tomorrow", "next week", "yesterday"
    - Bare times like "3:30 PM"
    - Day names without dates

    These are left for LLM consolidation which has full context.

    Example:
        ```python
        extractor = DateExtractor()
        facts = extractor.extract(episode)
        # facts[0].content = "2024-01-15"
        # facts[0].category = "date"
        ```
    """

    name: str = "date"
    category: str = "date"

    def __init__(
        self,
        languages: list[str] | None = None,
        prefer_dates_from: str = "past",
    ) -> None:
        """Initialize date extractor.

        Args:
            languages: List of language codes to try (e.g., ["en", "es"]).
                      Defaults to ["en"].
            prefer_dates_from: When ambiguous, prefer "past" or "future" dates.
        """
        self.languages = languages or ["en"]
        self.settings = {
            "PREFER_DATES_FROM": prefer_dates_from,
            "STRICT_PARSING": False,
            "RETURN_AS_TIMEZONE_AWARE": False,
        }

    def extract(self, episode: Episode) -> list[Fact]:
        """Extract HIGH-CONFIDENCE absolute dates from episode content.

        Only extracts dates with explicit years. Skips relative dates
        like "tomorrow" which require context to resolve correctly.

        Args:
            episode: Episode containing text to search.

        Returns:
            List of Facts, one per unique absolute date found.
        """
        extracted_dates: list[str] = []

        # search_dates finds all date-like strings in text
        results = search_dates(
            episode.content,
            languages=self.languages,
            settings=self.settings,
        )

        if results:
            for matched_text, dt in results:
                if dt is None:
                    continue

                # Skip relative dates - these need LLM context to resolve
                if _RELATIVE_PATTERNS.search(matched_text):
                    continue

                # Only extract if the matched text has an explicit year
                # This avoids "January 15" being resolved to current year
                if not _HAS_YEAR.search(matched_text):
                    continue

                # High-confidence absolute date - extract it
                # Include time if explicitly specified (not midnight)
                if dt.hour == 0 and dt.minute == 0 and dt.second == 0:
                    normalized = dt.strftime("%Y-%m-%d")
                else:
                    normalized = dt.strftime("%Y-%m-%d %H:%M")
                extracted_dates.append(normalized)

        # Deduplicate while preserving order
        unique_dates = list(dict.fromkeys(extracted_dates))

        return [self._create_fact(date, episode) for date in unique_dates]
