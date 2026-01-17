"""Query expansion for improved recall.

Uses LLM to expand queries with semantically related terms.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field
from pydantic_ai import Agent

if TYPE_CHECKING:
    from engram.embeddings import Embedder

logger = logging.getLogger(__name__)


class ExpandedQuery(BaseModel):
    """Result of query expansion.

    Attributes:
        original: The original query.
        expanded_terms: List of related terms.
        reasoning: Brief explanation of why these terms are related.
    """

    original: str = Field(description="The original query")
    expanded_terms: list[str] = Field(
        default_factory=list,
        max_length=5,
        description="Related terms (max 5)",
    )
    reasoning: str = Field(
        default="",
        description="Brief reasoning for the expansion",
    )


# Query expansion agent using Pydantic AI
_expansion_agent: Agent[None, ExpandedQuery] | None = None


def get_expansion_agent(model: str = "openai:gpt-4o-mini") -> Agent[None, ExpandedQuery]:
    """Get or create the query expansion agent.

    Args:
        model: LLM model to use for expansion.

    Returns:
        The query expansion agent.
    """
    global _expansion_agent
    if _expansion_agent is None:
        _expansion_agent = Agent(
            model,
            output_type=ExpandedQuery,
            system_prompt="""You are a query expansion assistant for a memory system.

Given a search query, generate 2-5 semantically related terms that would help find relevant memories.

Guidelines:
- Focus on synonyms, related concepts, and alternative phrasings
- Include both formal and informal variations
- Consider context where the term might appear
- Be concise - only include highly relevant terms
- Don't include the original query term

Examples:
- "email" → "contact", "address", "mail", "reach"
- "phone number" → "telephone", "mobile", "cell", "contact"
- "meeting" → "appointment", "call", "discussion", "scheduled"
""",
        )
    return _expansion_agent


async def expand_query(
    query: str,
    model: str = "openai:gpt-4o-mini",
) -> ExpandedQuery:
    """Expand a query with semantically related terms.

    Uses LLM to generate related terms for the query.

    Args:
        query: The original search query.
        model: LLM model to use for expansion.

    Returns:
        ExpandedQuery with original and expanded terms.
    """
    agent = get_expansion_agent(model)
    try:
        result = await agent.run(f"Expand this query: {query}")
        expanded = result.output
        expanded.original = query  # Ensure original is set
        logger.debug(f"Expanded '{query}' → {expanded.expanded_terms} ({expanded.reasoning})")
        return expanded
    except Exception as e:
        logger.warning(f"Query expansion failed for '{query}': {e}")
        # Return original query without expansion on error
        return ExpandedQuery(original=query, expanded_terms=[], reasoning=f"Expansion failed: {e}")


async def get_combined_embedding(
    query: str,
    embedder: Embedder,
    expand: bool = True,
    expansion_model: str = "openai:gpt-4o-mini",
) -> list[float]:
    """Get embedding for query, optionally with expansion.

    When expansion is enabled:
    1. Expand query to get related terms
    2. Embed original + expanded terms
    3. Average embeddings

    Args:
        query: The search query.
        embedder: Embedder instance for generating embeddings.
        expand: Whether to expand the query.
        expansion_model: LLM model for expansion.

    Returns:
        Combined embedding vector.
    """
    if not expand:
        return await embedder.embed(query)

    # Expand query
    expanded = await expand_query(query, expansion_model)

    if not expanded.expanded_terms:
        # No expansion, just return original embedding
        return await embedder.embed(query)

    # Embed all terms
    all_terms = [query] + expanded.expanded_terms
    embeddings = await embedder.embed_batch(all_terms)

    # Average embeddings (weighted: original gets higher weight)
    # Weight: original=2, expanded=1 each
    weights = [2.0] + [1.0] * len(expanded.expanded_terms)
    total_weight = sum(weights)

    combined = [0.0] * len(embeddings[0])
    for emb, weight in zip(embeddings, weights, strict=True):
        for i, val in enumerate(emb):
            combined[i] += val * weight

    # Normalize
    combined = [v / total_weight for v in combined]

    logger.info(f"Combined embedding for '{query}' with {len(expanded.expanded_terms)} expansions")

    return combined


__all__ = [
    "ExpandedQuery",
    "expand_query",
    "get_combined_embedding",
    "get_expansion_agent",
]
