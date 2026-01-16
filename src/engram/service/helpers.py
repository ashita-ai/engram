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


def mmr_rerank(
    candidates: list[tuple[float, list[float], int]],
    limit: int,
    diversity: float = 0.3,
) -> list[int]:
    """Rerank candidates using Maximal Marginal Relevance (MMR).

    MMR balances relevance with diversity by penalizing candidates
    that are too similar to already-selected results.

    MMR(d) = (1-λ) * Sim(d, query) - λ * max(Sim(d, selected))

    Where:
    - First term: relevance to query (the original score)
    - Second term: maximum similarity to any already-selected result
    - λ (diversity): trade-off parameter (0.0 = pure relevance, 1.0 = max diversity)

    Args:
        candidates: List of tuples (relevance_score, embedding, original_index).
        limit: Maximum number of results to select.
        diversity: Trade-off parameter (0.0-1.0). Higher = more diverse.

    Returns:
        List of original indices in MMR-ranked order.

    Example:
        >>> candidates = [(0.9, [0.1, 0.2], 0), (0.85, [0.15, 0.25], 1), (0.8, [0.9, 0.1], 2)]
        >>> mmr_rerank(candidates, limit=2, diversity=0.3)
        [0, 2]  # Selects most relevant, then diversifies
    """
    if not candidates or limit <= 0:
        return []

    if diversity == 0.0:
        # No diversity needed, return by relevance (original indices)
        sorted_by_relevance = sorted(candidates, key=lambda c: c[0], reverse=True)
        return [c[2] for c in sorted_by_relevance[:limit]]

    selected_indices: list[int] = []
    selected_embeddings: list[list[float]] = []
    remaining = list(range(len(candidates)))

    while len(selected_indices) < limit and remaining:
        best_idx = -1
        best_mmr = float("-inf")

        for idx in remaining:
            score, embedding, _ = candidates[idx]

            # Calculate max similarity to already-selected
            max_sim_to_selected = 0.0
            if selected_embeddings:
                max_sim_to_selected = max(
                    cosine_similarity(embedding, sel_emb) for sel_emb in selected_embeddings
                )

            # MMR score: balance relevance and diversity
            mmr_score = (1 - diversity) * score - diversity * max_sim_to_selected

            if mmr_score > best_mmr:
                best_mmr = mmr_score
                best_idx = idx

        if best_idx >= 0:
            selected_indices.append(candidates[best_idx][2])  # original index
            selected_embeddings.append(candidates[best_idx][1])
            remaining.remove(best_idx)

    return selected_indices


__all__ = [
    "IMPORTANCE_KEYWORDS",
    "calculate_importance",
    "cosine_similarity",
    "mmr_rerank",
]
