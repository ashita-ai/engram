"""Date extractor.

Uses dateparser library for robust natural language date parsing.
Supports 200+ languages and various date formats.
"""

from __future__ import annotations

from dateparser.search import search_dates  # type: ignore[import-untyped]

from engram.models import Episode, Fact

from .base import Extractor


class DateExtractor(Extractor):
    """Extract dates from episode content.

    Uses dateparser library for:
    - Multiple formats (ISO, US, European, named months)
    - Natural language ("next Tuesday", "2 weeks ago")
    - 200+ language support
    - Timezone awareness

    All dates are normalized to ISO 8601 format (YYYY-MM-DD).

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
        """Extract dates from episode content.

        Args:
            episode: Episode containing text to search.

        Returns:
            List of Facts, one per unique date found.
        """
        extracted_dates: list[str] = []

        # search_dates finds all date-like strings in text
        results = search_dates(
            episode.content,
            languages=self.languages,
            settings=self.settings,
        )

        if results:
            for _text, dt in results:
                if dt is not None:
                    # Normalize to ISO 8601 date format
                    normalized = dt.strftime("%Y-%m-%d")
                    extracted_dates.append(normalized)

        # Deduplicate while preserving order
        unique_dates = list(dict.fromkeys(extracted_dates))

        return [self._create_fact(date, episode) for date in unique_dates]
