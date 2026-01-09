"""Negation detection from episode content.

Detects explicit negations in text using pattern matching.
Creates NegationFact objects for filtering during recall.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field

from engram.models import Episode, NegationFact
from engram.models.base import ConfidenceScore


class NegationMatch(BaseModel):
    """A detected negation with its context.

    This is an intermediate representation used during pattern-based
    negation detection before creating NegationFact objects.
    """

    model_config = ConfigDict(extra="forbid")

    statement: str = Field(description="The full negation statement")
    negated_term: str = Field(description="What is being negated (pattern for filtering)")
    span: tuple[int, int] = Field(description="Character positions in original text")


# Patterns that indicate negation
# Format: (pattern, group_index_for_negated_term)
NEGATION_PATTERNS = [
    # "I don't/do not use X"
    (
        r"\b(?:I|i)\s+(?:don't|do\s+not|don't)\s+(?:use|like|want|need|have|prefer)\s+([A-Za-z0-9_-]+)",
        1,
    ),
    # "I'm not/am not a X user"
    (r"\b(?:I'm|I\s+am|i'm|i\s+am)\s+not\s+(?:a\s+)?([A-Za-z0-9_-]+)", 1),
    # "I never use/used X"
    (r"\b(?:I|i)\s+never\s+(?:use|used|like|liked|want|wanted|have|had)\s+([A-Za-z0-9_-]+)", 1),
    # "I no longer use X"
    (r"\b(?:I|i)\s+no\s+longer\s+(?:use|like|want|need|have|prefer)\s+([A-Za-z0-9_-]+)", 1),
    # "not X" in corrections: "Actually, not Python"
    (r"\b(?:actually|no|nope),?\s+not\s+([A-Za-z0-9_-]+)", 1),
    # "X is wrong/incorrect"
    (r"\b([A-Za-z0-9_-]+)\s+is\s+(?:wrong|incorrect|false|not\s+(?:right|correct|true))", 1),
    # "my X is not Y" (e.g., "my email is not jane@example.com")
    (r"\bmy\s+([A-Za-z0-9_-]+)\s+is\s+not\s+([A-Za-z0-9@._-]+)", 2),
    # "that's not my X"
    (r"\b(?:that's|that\s+is)\s+not\s+my\s+([A-Za-z0-9_-]+)", 1),
    # "I stopped using X"
    (r"\b(?:I|i)\s+stopped\s+(?:using|liking)\s+([A-Za-z0-9_-]+)", 1),
    # "not anymore" patterns: "I don't use X anymore"
    (r"\b(?:I|i)\s+(?:don't|do\s+not)\s+([A-Za-z0-9_-]+)\s+anymore", 1),
]

# Compiled patterns for efficiency
_COMPILED_PATTERNS = [(re.compile(p, re.IGNORECASE), g) for p, g in NEGATION_PATTERNS]


def detect_negations(text: str) -> list[NegationMatch]:
    """Detect explicit negations in text.

    Uses pattern matching to find statements like:
    - "I don't use MongoDB"
    - "I'm not a Java developer"
    - "My email is not jane@example.com"

    Args:
        text: Text to search for negations.

    Returns:
        List of NegationMatch objects with detected negations.
    """
    matches: list[NegationMatch] = []
    seen_spans: set[tuple[int, int]] = set()

    for pattern, group_idx in _COMPILED_PATTERNS:
        for match in pattern.finditer(text):
            span = match.span()

            # Skip overlapping matches
            if any(
                (span[0] >= s[0] and span[0] < s[1]) or (span[1] > s[0] and span[1] <= s[1])
                for s in seen_spans
            ):
                continue

            seen_spans.add(span)

            # Extract the negated term
            try:
                negated_term = match.group(group_idx).strip()
            except IndexError:
                negated_term = match.group(1).strip()

            # Clean up the full statement
            statement = match.group(0).strip()

            matches.append(
                NegationMatch(
                    statement=statement,
                    negated_term=negated_term.lower(),
                    span=span,
                )
            )

    return matches


def create_negation_facts(
    episode: Episode,
    matches: list[NegationMatch] | None = None,
) -> list[NegationFact]:
    """Create NegationFact objects from detected negations.

    If matches is None, runs detect_negations on the episode content.

    Args:
        episode: Source episode.
        matches: Optional pre-detected matches. If None, detection runs automatically.

    Returns:
        List of NegationFact objects.
    """
    if matches is None:
        matches = detect_negations(episode.content)

    facts: list[NegationFact] = []

    for match in matches:
        fact = NegationFact(
            content=match.statement,
            negates_pattern=match.negated_term,
            source_episode_ids=[episode.id],
            user_id=episode.user_id,
            org_id=episode.org_id,
            confidence=ConfidenceScore.for_extracted(),  # Pattern-matched = 0.9
        )
        facts.append(fact)

    return facts


class NegationDetector:
    """Detector for explicit negations in episode content.

    Unlike regular extractors that create Fact objects, this creates
    NegationFact objects which have different fields (negates_pattern,
    multiple source episodes).

    Example:
        ```python
        detector = NegationDetector()
        negations = detector.detect(episode)
        # negations[0].content = "I don't use MongoDB"
        # negations[0].negates_pattern = "mongodb"
        ```
    """

    name: str = "negation"

    def detect(self, episode: Episode) -> list[NegationFact]:
        """Detect negations in episode content.

        Args:
            episode: Episode to search for negations.

        Returns:
            List of NegationFact objects.
        """
        return create_negation_facts(episode)


__all__ = [
    "NegationDetector",
    "NegationMatch",
    "create_negation_facts",
    "detect_negations",
]
