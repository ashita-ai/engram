"""Human name extractor.

Uses nameparser library for parsing human names into components.
"""

from __future__ import annotations

import re

from nameparser import HumanName  # type: ignore[import-untyped]

from engram.models import Episode, Fact

from .base import Extractor

# Pattern to find potential name candidates
# Matches capitalized word sequences (2-5 words)
NAME_CANDIDATE_PATTERN = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,4})\b")


class NameExtractor(Extractor):
    """Extract human names from episode content.

    Uses nameparser library for:
    - Title recognition (Dr., Mr., Mrs., etc.)
    - First/middle/last name parsing
    - Suffix handling (Jr., III, PhD, etc.)
    - Nickname extraction

    Example:
        ```python
        extractor = NameExtractor()
        facts = extractor.extract(episode)
        # facts[0].content = "John Smith"
        # facts[0].category = "person"
        ```
    """

    name: str = "name"
    category: str = "person"

    def extract(self, episode: Episode) -> list[Fact]:
        """Extract human names from episode content.

        Args:
            episode: Episode containing text to search.

        Returns:
            List of Facts, one per unique valid name found.
        """
        valid_names: list[str] = []

        # Find potential name candidates
        candidates = NAME_CANDIDATE_PATTERN.findall(episode.content)

        for candidate in candidates:
            # Parse with nameparser
            parsed = HumanName(candidate)

            # Validate: must have at least first and last name
            if parsed.first and parsed.last:
                # Normalize the name
                # Use the parsed components to reconstruct
                parts = []
                if parsed.title:
                    parts.append(parsed.title)
                if parsed.first:
                    parts.append(parsed.first)
                if parsed.middle:
                    parts.append(parsed.middle)
                if parsed.last:
                    parts.append(parsed.last)
                if parsed.suffix:
                    parts.append(parsed.suffix)

                normalized = " ".join(parts)
                valid_names.append(normalized)

        # Deduplicate while preserving order
        unique_names = list(dict.fromkeys(valid_names))

        return [self._create_fact(name, episode) for name in unique_names]
