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

# Units that are ambiguous with time/date notation or common words - skip these
_AMBIGUOUS_UNITS = {"am", "pm", "st", "nd", "rd", "th", "unit", "units", "item", "items", "a", "an"}


def _pluralize_unit(unit_str: str, magnitude: float) -> str:
    """Pluralize a unit string based on magnitude.

    Args:
        unit_str: The unit string (e.g., "kilogram", "year")
        magnitude: The numeric magnitude

    Returns:
        Pluralized unit string if magnitude != 1
    """
    if magnitude == 1:
        return unit_str

    # Units that don't change in plural
    invariant = {"hertz", "hz", "siemens", "lux", "fps"}
    if unit_str.lower() in invariant:
        return unit_str

    # Units ending in 's', 'x', 'z', 'ch', 'sh' get 'es'
    if unit_str.endswith(("s", "x", "z", "ch", "sh")):
        return unit_str + "es"

    # Units ending in 'y' after consonant get 'ies'
    if unit_str.endswith("y") and len(unit_str) > 1 and unit_str[-2] not in "aeiou":
        return unit_str[:-1] + "ies"

    # Default: add 's'
    return unit_str + "s"


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

                # Format as "magnitude unit(s)" with proper pluralization
                # Strip trailing zeros for cleaner output (10.0 -> 10)
                magnitude_clean = (
                    int(quantity.magnitude)
                    if quantity.magnitude == int(quantity.magnitude)
                    else quantity.magnitude
                )
                unit_str_base = str(quantity.units)
                unit_str_final = _pluralize_unit(unit_str_base, magnitude_clean)
                formatted = f"{magnitude_clean} {unit_str_final}"
                valid_quantities.append(formatted)

            except (UndefinedUnitError, OffsetUnitCalculusError, ValueError):
                # Not a recognized unit or offset unit issue, skip
                continue

        # Deduplicate while preserving order
        unique_quantities = list(dict.fromkeys(valid_quantities))

        return [self._create_fact(qty, episode) for qty in unique_quantities]
