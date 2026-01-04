"""Standard number extractor.

Uses python-stdnum library for validating standard identification numbers.
Supports 200+ formats including SSN, credit cards, ISBN, IBAN, VAT numbers, etc.
"""

from __future__ import annotations

import re

from engram.models import Episode, Fact

from .base import Extractor

# Import specific validators from stdnum
# Each module provides validate(), compact(), and format() functions
try:
    from stdnum import (
        iban,
        isbn,
        issn,
        luhn,  # For credit card validation
    )
    from stdnum.us import ssn as us_ssn
except ImportError as e:
    raise ImportError("python-stdnum is required for IDExtractor") from e


# Pattern for potential credit card numbers (13-19 digits, possibly with spaces/dashes)
CREDIT_CARD_PATTERN = re.compile(r"\b(\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{1,7})\b")

# Pattern for potential SSN (XXX-XX-XXXX or XXXXXXXXX)
SSN_PATTERN = re.compile(r"\b(\d{3}-\d{2}-\d{4}|\d{9})\b")

# Pattern for potential ISBN-10 or ISBN-13
ISBN_PATTERN = re.compile(
    r"\b((?:97[89][-\s]?)?\d{1,5}[-\s]?\d{1,7}[-\s]?\d{1,7}[-\s]?[\dX])\b",
    re.IGNORECASE,
)

# Pattern for potential IBAN (2 letters + 2 digits + up to 30 alphanumeric)
IBAN_PATTERN = re.compile(
    r"\b([A-Z]{2}\d{2}[A-Z0-9]{4,30})\b",
    re.IGNORECASE,
)

# Pattern for potential ISSN (XXXX-XXXX)
ISSN_PATTERN = re.compile(
    r"\b(\d{4}-\d{3}[\dX])\b",
    re.IGNORECASE,
)


class IDExtractor(Extractor):
    """Extract standard identification numbers from episode content.

    Uses python-stdnum library for:
    - Credit card validation (Luhn algorithm)
    - US Social Security Numbers
    - ISBN (International Standard Book Number)
    - IBAN (International Bank Account Number)
    - ISSN (International Standard Serial Number)

    Example:
        ```python
        extractor = IDExtractor()
        facts = extractor.extract(episode)
        # facts[0].content = "978-0-13-468599-1"
        # facts[0].category = "isbn"
        ```
    """

    name: str = "id"
    category: str = "identifier"

    def extract(self, episode: Episode) -> list[Fact]:
        """Extract standard IDs from episode content.

        Args:
            episode: Episode containing text to search.

        Returns:
            List of Facts, one per unique valid ID found.
        """
        facts: list[Fact] = []
        seen: set[str] = set()

        # Extract credit cards
        for match in CREDIT_CARD_PATTERN.finditer(episode.content):
            candidate = match.group(1)
            # Remove spaces and dashes for validation
            digits_only = re.sub(r"[\s-]", "", candidate)

            if len(digits_only) >= 13 and luhn.is_valid(digits_only):
                # Mask the middle digits for privacy
                masked = self._mask_credit_card(digits_only)
                if masked not in seen:
                    seen.add(masked)
                    facts.append(self._create_fact(masked, episode, category="credit_card"))

        # Extract SSNs
        for match in SSN_PATTERN.finditer(episode.content):
            candidate = match.group(1)
            try:
                # Validate and format
                formatted = us_ssn.format(us_ssn.validate(candidate))
                if formatted not in seen:
                    seen.add(formatted)
                    # Mask SSN for privacy
                    masked = self._mask_ssn(formatted)
                    facts.append(self._create_fact(masked, episode, category="ssn"))
            except Exception:
                continue

        # Extract ISBNs
        for match in ISBN_PATTERN.finditer(episode.content):
            candidate = match.group(1)
            try:
                formatted = isbn.format(isbn.validate(candidate))
                if formatted not in seen:
                    seen.add(formatted)
                    facts.append(self._create_fact(formatted, episode, category="isbn"))
            except Exception:
                continue

        # Extract IBANs
        for match in IBAN_PATTERN.finditer(episode.content):
            candidate = match.group(1).upper()
            try:
                formatted = iban.format(iban.validate(candidate))
                if formatted not in seen:
                    seen.add(formatted)
                    # Mask IBAN for privacy (show first 4, last 4)
                    masked = self._mask_iban(formatted)
                    facts.append(self._create_fact(masked, episode, category="iban"))
            except Exception:
                continue

        # Extract ISSNs
        for match in ISSN_PATTERN.finditer(episode.content):
            candidate = match.group(1)
            try:
                formatted = issn.format(issn.validate(candidate))
                if formatted not in seen:
                    seen.add(formatted)
                    facts.append(self._create_fact(formatted, episode, category="issn"))
            except Exception:
                continue

        return facts

    def _mask_credit_card(self, card: str) -> str:
        """Mask credit card number for privacy.

        Shows first 4 and last 4 digits.
        """
        if len(card) < 8:
            return "****"
        return f"{card[:4]}****{card[-4:]}"

    def _mask_ssn(self, ssn: str) -> str:
        """Mask SSN for privacy.

        Shows only last 4 digits.
        """
        return f"***-**-{ssn[-4:]}"

    def _mask_iban(self, iban_str: str) -> str:
        """Mask IBAN for privacy.

        Shows first 4 and last 4 characters.
        """
        clean = iban_str.replace(" ", "")
        if len(clean) < 8:
            return "****"
        return f"{clean[:4]}****{clean[-4:]}"
