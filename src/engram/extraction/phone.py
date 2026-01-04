"""Phone number extractor.

Uses regex pattern matching to extract phone numbers from episode content.
Supports various US and international formats.
"""

from __future__ import annotations

import re

from engram.models import Episode, Fact

from .base import Extractor

# Phone patterns for various formats
# US formats: (123) 456-7890, 123-456-7890, 123.456.7890, 1234567890
# International: +1-123-456-7890, +44 20 7946 0958
PHONE_PATTERNS = [
    # US with area code in parens: (123) 456-7890
    re.compile(r"\(\d{3}\)\s*\d{3}[-.\s]?\d{4}"),
    # US with dashes/dots: 123-456-7890 or 123.456.7890
    re.compile(r"\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b"),
    # International: +1-123-456-7890 or +44 20 7946 0958
    re.compile(r"\+\d{1,3}[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}"),
    # 10 digits together (but not in longer number sequences)
    re.compile(r"(?<!\d)\d{10}(?!\d)"),
]


def normalize_phone(phone: str) -> str:
    """Normalize phone number to digits only with optional + prefix.

    Args:
        phone: Raw phone string.

    Returns:
        Normalized phone with only digits and optional leading +.
    """
    # Keep + prefix for international
    if phone.startswith("+"):
        return "+" + re.sub(r"\D", "", phone[1:])
    return re.sub(r"\D", "", phone)


def _is_subset_phone(short: str, long: str) -> bool:
    """Check if short phone is a suffix of long phone (after stripping +).

    Args:
        short: Shorter normalized phone.
        long: Longer normalized phone.

    Returns:
        True if short is contained in long.
    """
    short_digits = short.lstrip("+")
    long_digits = long.lstrip("+")
    return long_digits.endswith(short_digits)


class PhoneExtractor(Extractor):
    """Extract phone numbers from episode content.

    Supports multiple phone formats including:
    - US: (123) 456-7890, 123-456-7890, 123.456.7890
    - International: +1-123-456-7890, +44 20 7946 0958
    - Plain digits: 1234567890

    Phone numbers are normalized to digits-only format for storage.

    Example:
        ```python
        extractor = PhoneExtractor()
        facts = extractor.extract(episode)
        # facts[0].content = "1234567890"
        # facts[0].category = "phone"
        ```
    """

    name: str = "phone"
    category: str = "phone"

    def extract(self, episode: Episode) -> list[Fact]:
        """Extract phone numbers from episode content.

        Args:
            episode: Episode containing text to search.

        Returns:
            List of Facts, one per unique phone number found.
        """
        all_matches: list[str] = []

        for pattern in PHONE_PATTERNS:
            matches = pattern.findall(episode.content)
            all_matches.extend(matches)

        # Normalize and deduplicate
        normalized = [normalize_phone(phone) for phone in all_matches]

        # Filter out numbers that are too short (less than 7 digits)
        valid = [p for p in normalized if len(p.lstrip("+")) >= 7]

        # Deduplicate while preserving order
        unique_phones = list(dict.fromkeys(valid))

        # Remove phones that are subsets of other phones
        # (e.g., 5551234567 is subset of +15551234567)
        final_phones: list[str] = []
        for phone in unique_phones:
            is_subset = False
            for other in unique_phones:
                if phone != other and _is_subset_phone(phone, other):
                    is_subset = True
                    break
            if not is_subset:
                final_phones.append(phone)

        return [self._create_fact(phone, episode) for phone in final_phones]
