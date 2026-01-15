"""Helper functions for the service layer.

Contains utility functions used across the service:
- _cosine_similarity: Vector similarity computation
- _calculate_importance: Episode importance scoring
"""

from __future__ import annotations

import math

from engram.models import StructuredMemory

# Keywords that indicate important content worth remembering
IMPORTANCE_KEYWORDS = frozenset(
    [
        "remember",
        "important",
        "don't forget",
        "always",
        "never",
        "critical",
        "key",
        "must",
        "essential",
        "priority",
        "urgent",
        "note that",
        "keep in mind",
        "fyi",
        "heads up",
    ]
)


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Compute cosine similarity between two vectors.

    Returns a value clamped to [0.0, 1.0] to handle floating point precision.

    Args:
        vec1: First vector.
        vec2: Second vector.

    Returns:
        Cosine similarity score between 0.0 and 1.0.
    """
    dot_product = sum(a * b for a, b in zip(vec1, vec2, strict=True))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    # Clamp to handle floating point precision (e.g., 1.0000000000000002)
    return max(0.0, min(1.0, dot_product / (norm1 * norm2)))


def calculate_importance(
    content: str,
    role: str,
    structured: StructuredMemory | None = None,
    base_importance: float = 0.5,
) -> float:
    """Calculate episode importance based on content and structured extraction results.

    Uses heuristics to determine how important an episode is:
    - Regex extracts (emails, phones, URLs) indicate concrete information
    - LLM extracts (people, prefs, negations) indicate rich content
    - Negations are corrections, which are very important
    - Certain keywords suggest the user wants something remembered
    - User messages are slightly more important than assistant responses

    Args:
        content: The episode content.
        role: Role of the speaker (user, assistant, system).
        structured: Optional StructuredMemory with extracts.
        base_importance: Starting importance value (default 0.5).

    Returns:
        Importance score clamped to [0.0, 1.0].
    """
    score = base_importance

    if structured is not None:
        # Regex extracts indicate concrete info worth remembering
        regex_count = len(structured.emails) + len(structured.phones) + len(structured.urls)
        score += min(0.15, regex_count * 0.05)

        # LLM extracts (if enriched)
        if structured.enriched:
            llm_count = (
                len(structured.people) + len(structured.preferences) + len(structured.negations)
            )
            score += min(0.15, llm_count * 0.05)

            # Negations are corrections - very important for accuracy
            score += min(0.2, len(structured.negations) * 0.1)

    # Check for importance keywords
    content_lower = content.lower()
    keyword_matches = sum(1 for kw in IMPORTANCE_KEYWORDS if kw in content_lower)
    # Each keyword match adds 0.05, capped at 0.1 (2+ matches)
    score += min(0.1, keyword_matches * 0.05)

    # User messages are slightly more important than assistant responses
    # (user is providing info, assistant is often just responding)
    if role == "user":
        score += 0.05

    # System messages are usually setup/instructions, less important for recall
    if role == "system":
        score -= 0.1

    return max(0.0, min(1.0, score))


__all__ = [
    "IMPORTANCE_KEYWORDS",
    "calculate_importance",
    "cosine_similarity",
]
