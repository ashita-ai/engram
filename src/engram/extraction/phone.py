"""Phone number extractor.

Uses Google's phonenumbers library for robust international phone parsing.
"""

from __future__ import annotations

import phonenumbers
from phonenumbers import PhoneNumberFormat

from engram.models import Episode, Fact

from .base import Extractor


class PhoneExtractor(Extractor):
    """Extract phone numbers from episode content.

    Uses Google's phonenumbers library (libphonenumber port) for:
    - International format support (200+ countries)
    - Proper validation
    - E.164 normalization

    Example:
        ```python
        extractor = PhoneExtractor()
        facts = extractor.extract(episode)
        # facts[0].content = "+15551234567"
        # facts[0].category = "phone"
        ```
    """

    name: str = "phone"
    category: str = "phone"

    def __init__(self, default_region: str = "US") -> None:
        """Initialize phone extractor.

        Args:
            default_region: Default region for parsing numbers without country code.
                           Uses ISO 3166-1 alpha-2 codes (e.g., "US", "GB", "DE").
        """
        self.default_region = default_region

    def extract(self, episode: Episode) -> list[Fact]:
        """Extract phone numbers from episode content.

        Args:
            episode: Episode containing text to search.

        Returns:
            List of Facts, one per unique valid phone number found.
        """
        valid_phones: list[str] = []

        # PhoneNumberMatcher finds all phone numbers in text
        for match in phonenumbers.PhoneNumberMatcher(
            episode.content,
            self.default_region,
        ):
            phone = match.number

            # Validate the number
            if phonenumbers.is_valid_number(phone):
                # Format as E.164 (+15551234567)
                formatted = phonenumbers.format_number(phone, PhoneNumberFormat.E164)
                valid_phones.append(formatted)

        # Deduplicate while preserving order
        unique_phones = list(dict.fromkeys(valid_phones))

        return [self._create_fact(phone, episode) for phone in unique_phones]
