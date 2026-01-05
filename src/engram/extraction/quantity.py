"""Quantity and unit extractor.

Uses Pint library for robust physical quantity parsing and normalization.
"""

from __future__ import annotations

import re
from typing import Any

from pint import UnitRegistry
from pint.errors import OffsetUnitCalculusError, UndefinedUnitError

from engram.models import Episode, Fact

from .base import Extractor

# Initialize unit registry once
_ureg: Any = UnitRegistry()


# Pattern to find potential quantity expressions
# Matches: "5 km", "72°F", "2.5 liters", "100 mph", etc.
# Uses negative lookbehind to avoid matching time patterns like "3:30 PM"
QUANTITY_PATTERN = re.compile(
    r"(?<!:)"  # Negative lookbehind: not preceded by colon (avoids 3:30 PM)
    r"\b(\d+(?:\.\d+)?)\s*"  # Number (integer or decimal)
    r"([a-zA-Z°][a-zA-Z°/²³]*(?:\s*/\s*[a-zA-Z°][a-zA-Z°/²³]*)?)"  # Unit
    r"\b",
    re.IGNORECASE,
)

# Units that are ambiguous with time/date notation - skip these
_AMBIGUOUS_UNITS = {"am", "pm", "st", "nd", "rd", "th"}


class QuantityExtractor(Extractor):
    """Extract physical quantities from episode content.

    Uses Pint library for:
    - Unit recognition and validation
    - Magnitude + unit parsing
    - Support for compound units (km/h, m/s²)

    Example:
        ```python
        extractor = QuantityExtractor()
        facts = extractor.extract(episode)
        # facts[0].content = "5.0 kilometer"
        # facts[0].category = "quantity"
        ```
    """

    name: str = "quantity"
    category: str = "quantity"

    def extract(self, episode: Episode) -> list[Fact]:
        """Extract quantities from episode content.

        Args:
            episode: Episode containing text to search.

        Returns:
            List of Facts, one per unique valid quantity found.
        """
        valid_quantities: list[str] = []

        for match in QUANTITY_PATTERN.finditer(episode.content):
            magnitude_str = match.group(1)
            unit_str = match.group(2)

            # Skip ambiguous units that conflict with time/date notation
            if unit_str.lower() in _AMBIGUOUS_UNITS:
                continue

            try:
                # Try to parse as a Pint quantity
                magnitude = float(magnitude_str)

                # Use Quantity constructor to handle all units including offset units
                quantity = _ureg.Quantity(magnitude, unit_str)

                # Format as "magnitude unit" (e.g., "5.0 kilometer")
                # Use compact format for cleaner output
                formatted = f"{quantity.magnitude} {quantity.units:~P}"
                valid_quantities.append(formatted)

            except (UndefinedUnitError, OffsetUnitCalculusError, ValueError):
                # Not a recognized unit or offset unit issue, skip
                continue

        # Deduplicate while preserving order
        unique_quantities = list(dict.fromkeys(valid_quantities))

        return [self._create_fact(qty, episode) for qty in unique_quantities]
