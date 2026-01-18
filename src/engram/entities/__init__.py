"""Entity extraction and resolution for memory linking.

Provides LLM-driven entity resolution to connect memories that reference
the same real-world entities using different names or aliases.

Example:
    ```python
    from engram.entities import resolve_entities, Entity

    # Resolve entities from memories
    result = await resolve_entities(
        memories=[
            {"id": "mem_1", "content": "John mentioned he likes Python"},
            {"id": "mem_2", "content": "John Smith is a software engineer"},
        ],
        user_id="user_123",
    )

    for entity in result.entities:
        print(f"{entity.canonical_name}: {entity.aliases}")
    ```
"""

from .models import (
    Entity,
    EntityCluster,
    EntityMention,
    EntityResolutionResult,
    EntityType,
)
from .resolution import (
    cluster_mentions,
    extract_entities,
    find_entity_links,
    resolve_entities,
)

__all__ = [
    # Models
    "Entity",
    "EntityCluster",
    "EntityMention",
    "EntityResolutionResult",
    "EntityType",
    # Functions
    "cluster_mentions",
    "extract_entities",
    "find_entity_links",
    "resolve_entities",
]
