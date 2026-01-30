"""LLM-driven link discovery for A-MEM style memory linking.

Provides intelligent relationship discovery beyond embedding similarity,
enabling richer multi-hop reasoning through semantic analysis.

Example:
    ```python
    from engram.linking import discover_links, LinkType

    # Discover relationships between memories
    result = await discover_links(
        new_memory=semantic_memory,
        candidate_memories=similar_memories,
    )

    for link in result.links:
        new_memory.add_link(link.target_id, link.link_type)
    ```
"""

from .discovery import (
    LINK_TYPE_DESCRIPTIONS,
    DiscoveredLink,
    LinkDiscoveryResult,
    LinkType,
    MemoryEvolution,
    discover_and_apply_links,
    discover_links,
    evolve_memory,
)

__all__ = [
    "LINK_TYPE_DESCRIPTIONS",
    "DiscoveredLink",
    "LinkDiscoveryResult",
    "LinkType",
    "MemoryEvolution",
    "discover_and_apply_links",
    "discover_links",
    "evolve_memory",
]
