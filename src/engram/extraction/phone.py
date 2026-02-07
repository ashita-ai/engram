"""Phone number extractor.

Uses Google's phonenumbers library for robust international phone parsing.
"""

from __future__ import annotations

import re

import phonenumbers
from phonenumbers import Leniency, PhoneNumberFormat

from engram.config import settings
from engram.models import Episode

from .base import Extractor

# Matches raw strings that are date-like rather than phone-like.
# Covers YYYY-MM-DD, YYYY/MM/DD, YYYYMMDD, and ranges like YYYY-MM-DD/DD.
_DATE_PATTERN = re.compile(
    r"^\d{4}[-/]\d{2}[-/]\d{2}(?:[/\-]\d{2})?$"  # YYYY-MM-DD, YYYY/MM/DD, with optional /DD
    r"|^\d{8}$"  # YYYYMMDD
)


class PhoneExtractor(Extractor):
    """Extract phone numbers from episode content.

    Uses Google's phonenumbers library (libphonenumber port) for:
    - International format support (200+ countries)
    - Proper validation
    - E.164 normalization

    Example:
        ```python
        extractor = PhoneExtractor()
        phones = extractor.extract(episode)
        # phones = ["+15551234567"]
        ```
    """

    name: str = "phone"

    def __init__(self, default_region: str | None = None) -> None:
        """Initialize phone extractor.

        Args:
            default_region: Default region for parsing numbers without country code.
                           Uses ISO 3166-1 alpha-2 codes (e.g., "US", "GB", "DE").
                           If not provided, uses ENGRAM_PHONE_DEFAULT_REGION setting.
        """
        self.default_region = default_region or settings.phone_default_region

    def extract(self, episode: Episode) -> list[str]:
        """Extract phone numbers from episode content.

        Args:
            episode: Episode containing text to search.

        Returns:
            List of unique valid phone numbers in E.164 format.
        """
        valid_phones: list[str] = []

        for match in phonenumbers.PhoneNumberMatcher(
            episode.content,
            self.default_region,
            Leniency.VALID,
        ):
            if _DATE_PATTERN.match(match.raw_string):
                continue

            phone = match.number

            if phonenumbers.is_valid_number(phone):
                formatted = phonenumbers.format_number(phone, PhoneNumberFormat.E164)
                valid_phones.append(formatted)

        # Deduplicate while preserving order
        return list(dict.fromkeys(valid_phones))
