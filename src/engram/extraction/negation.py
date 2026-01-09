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


# Core negation patterns - simplified for reliability
# Edge cases are handled by LLM during consolidation
# Format: (pattern, group_index_for_negated_term)
# Patterns capture term + optional version (e.g., "Python 2", "React 18")
NEGATION_PATTERNS = [
    # "I don't/do not use/like/want/need/prefer X [version]"
    (
        r"\b(?:I|i)\s+(?:don't|do\s+not|don't)\s+(?:use|like|want|need|prefer)\s+([A-Za-z][A-Za-z0-9_-]*(?:\s+\d+(?:\.\d+)?[a-z]?)?)",
        1,
    ),
    # "I'm not a/an X" - identity negation
    (r"\b(?:I'm|I\s+am|i'm|i\s+am)\s+not\s+(?:a|an)\s+([A-Za-z][A-Za-z0-9_-]*)", 1),
    # "I'm not interested in X"
    (r"\b(?:I'm|I\s+am|i'm|i\s+am)\s+not\s+interested\s+in\s+([A-Za-z][A-Za-z0-9_-]*)", 1),
    # "I never use/used X [version]"
    (r"\b(?:I|i)\s+never\s+(?:use|used)\s+([A-Za-z][A-Za-z0-9_-]*(?:\s+\d+(?:\.\d+)?[a-z]?)?)", 1),
    # "I/We no longer use/support X [version]"
    (
        r"\b(?:I|i|we|We)\s+no\s+longer\s+(?:use|support)\s+([A-Za-z][A-Za-z0-9_-]*(?:\s+\d+(?:\.\d+)?[a-z]?)?)",
        1,
    ),
]

# Compiled patterns for efficiency
_COMPILED_PATTERNS = [(re.compile(p, re.IGNORECASE), g) for p, g in NEGATION_PATTERNS]

# Short terms that are valid for negation (programming languages, etc.)
# These bypass the 2-character minimum check
# Note: These patterns require word-boundary matching during filtering to avoid
# false positives (e.g., "r" matching every word containing the letter)
SHORT_TERM_ALLOWLIST = {"r", "c", "d", "f", "j", "go"}  # R, C, D, F#, J, Go (languages)


def detect_negations(text: str) -> list[NegationMatch]:
    """Detect explicit negations in text.

    Uses simplified pattern matching for high-confidence cases:
    - "I don't use MongoDB"
    - "I'm not a Java developer"
    - "I never use Windows"
    - "We no longer support Python 2"

    Edge cases are handled by LLM during consolidation.

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

            # Skip patterns that are too short (single letters cause overly broad matches)
            # Minimum 2 characters unless it's a known short-term (R, C, Go, etc.)
            if len(negated_term) < 2 and negated_term.lower() not in SHORT_TERM_ALLOWLIST:
                continue

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
