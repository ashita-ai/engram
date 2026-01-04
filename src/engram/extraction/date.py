"""Date extractor.

Uses regex pattern matching to extract dates from episode content.
Supports various date formats and normalizes to ISO 8601.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from datetime import datetime

from engram.models import Episode, Fact

from .base import Extractor

# Type alias for date parser functions
DateParser = Callable[[re.Match[str]], tuple[int, int, int] | None]

# Month name mappings
MONTH_NAMES = {
    "january": 1,
    "jan": 1,
    "february": 2,
    "feb": 2,
    "march": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "may": 5,
    "june": 6,
    "jun": 6,
    "july": 7,
    "jul": 7,
    "august": 8,
    "aug": 8,
    "september": 9,
    "sep": 9,
    "sept": 9,
    "october": 10,
    "oct": 10,
    "november": 11,
    "nov": 11,
    "december": 12,
    "dec": 12,
}

# Date patterns: (compiled regex, parser function)
DATE_PATTERNS: list[tuple[re.Pattern[str], DateParser]] = [
    # ISO format: 2024-01-15
    (
        re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b"),
        lambda m: (int(m.group(1)), int(m.group(2)), int(m.group(3))),
    ),
    # US format: 01/15/2024 or 1/15/2024
    (
        re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b"),
        lambda m: (int(m.group(3)), int(m.group(1)), int(m.group(2))),
    ),
    # European format: 15/01/2024 or 15-01-2024 (day > 12 indicates European)
    (
        re.compile(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b"),
        lambda m: _parse_ambiguous_date(m.group(1), m.group(2), m.group(3)),
    ),
    # Month name formats: January 15, 2024 or Jan 15 2024
    (
        re.compile(
            r"\b(january|jan|february|feb|march|mar|april|apr|may|june|jun|"
            r"july|jul|august|aug|september|sept?|october|oct|november|nov|"
            r"december|dec)\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b",
            re.IGNORECASE,
        ),
        lambda m: (int(m.group(3)), MONTH_NAMES[m.group(1).lower()], int(m.group(2))),
    ),
    # Day Month Year: 15 January 2024 or 15th Jan, 2024
    (
        re.compile(
            r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(january|jan|february|feb|march|mar|"
            r"april|apr|may|june|jun|july|jul|august|aug|september|sept?|"
            r"october|oct|november|nov|december|dec),?\s+(\d{4})\b",
            re.IGNORECASE,
        ),
        lambda m: (int(m.group(3)), MONTH_NAMES[m.group(2).lower()], int(m.group(1))),
    ),
]


def _parse_ambiguous_date(
    first: str,
    second: str,
    year: str,
) -> tuple[int, int, int] | None:
    """Parse ambiguous date format (could be MM/DD or DD/MM).

    Uses heuristics:
    - If first > 12: must be DD/MM (European)
    - If second > 12: must be MM/DD (US)
    - Otherwise: assume US format (MM/DD)

    Args:
        first: First number in date.
        second: Second number in date.
        year: Year.

    Returns:
        Tuple of (year, month, day) or None if invalid.
    """
    f, s, y = int(first), int(second), int(year)

    if f > 12 and s <= 12:
        # Must be DD/MM
        return (y, s, f)
    elif s > 12 and f <= 12:
        # Must be MM/DD
        return (y, f, s)
    elif f <= 12 and s <= 12:
        # Ambiguous - assume US format (MM/DD)
        return (y, f, s)
    else:
        # Both > 12 - invalid
        return None


def _validate_date(year: int, month: int, day: int) -> bool:
    """Validate that a date is real.

    Args:
        year: Year (1900-2100 range).
        month: Month (1-12).
        day: Day (1-31, validated for month).

    Returns:
        True if valid date, False otherwise.
    """
    if not (1900 <= year <= 2100):
        return False
    if not (1 <= month <= 12):
        return False
    if not (1 <= day <= 31):
        return False

    try:
        datetime(year, month, day)
        return True
    except ValueError:
        return False


def normalize_date(year: int, month: int, day: int) -> str:
    """Normalize date to ISO 8601 format.

    Args:
        year: Year.
        month: Month.
        day: Day.

    Returns:
        ISO 8601 date string (YYYY-MM-DD).
    """
    return f"{year:04d}-{month:02d}-{day:02d}"


class DateExtractor(Extractor):
    """Extract dates from episode content.

    Supports multiple date formats:
    - ISO: 2024-01-15
    - US: 01/15/2024, 1/15/2024
    - European: 15/01/2024 (when day > 12)
    - Named: January 15, 2024, 15th Jan 2024

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

    def extract(self, episode: Episode) -> list[Fact]:
        """Extract dates from episode content.

        Args:
            episode: Episode containing text to search.

        Returns:
            List of Facts, one per unique date found.
        """
        extracted_dates: list[str] = []

        for pattern, parser in DATE_PATTERNS:
            for match in pattern.finditer(episode.content):
                result = parser(match)
                if result is None:
                    continue

                year, month, day = result
                if _validate_date(year, month, day):
                    normalized = normalize_date(year, month, day)
                    extracted_dates.append(normalized)

        # Deduplicate while preserving order
        unique_dates = list(dict.fromkeys(extracted_dates))

        return [self._create_fact(date, episode) for date in unique_dates]
