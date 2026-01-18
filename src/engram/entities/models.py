"""Entity models for resolution and linking.

Implements entity extraction and resolution to connect memories
that reference the same real-world entities (people, organizations,
projects, etc.) using different names or aliases.

Example:
    "John mentioned he likes Python" and "John Smith is a software engineer"
    both reference the same person entity.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from engram.models.base import generate_id

# Entity types that can be extracted and resolved
EntityType = Literal[
    "person",
    "organization",
    "project",
    "technology",
    "location",
    "product",
    "concept",
]


class EntityMention(BaseModel):
    """A mention of an entity in a memory.

    Represents a single reference to an entity, which may be
    resolved to a canonical Entity later.

    Attributes:
        text: The exact text of the mention.
        entity_type: Type of entity (person, organization, etc.).
        memory_id: ID of the memory containing this mention.
        context: Surrounding text for disambiguation.
        confidence: Confidence in entity type classification.
    """

    model_config = ConfigDict(extra="forbid")

    text: str = Field(description="Exact text of the mention")
    entity_type: EntityType = Field(description="Type of entity")
    memory_id: str = Field(description="Memory containing this mention")
    context: str = Field(default="", description="Surrounding context")
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        default=0.8,
        description="Confidence in entity type",
    )


class Entity(BaseModel):
    """A resolved entity with canonical name and aliases.

    Represents a real-world entity (person, organization, etc.) that
    may be referenced by multiple names across different memories.

    Attributes:
        id: Unique entity identifier.
        canonical_name: Primary/official name of the entity.
        aliases: Alternative names and references.
        entity_type: Type of entity.
        memory_ids: IDs of memories referencing this entity.
        attributes: Key-value attributes about the entity.
        created_at: When the entity was first resolved.
        updated_at: When the entity was last modified.
        merge_count: Number of mentions merged into this entity.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: generate_id("ent"))
    canonical_name: str = Field(description="Primary name of the entity")
    aliases: list[str] = Field(
        default_factory=list,
        description="Alternative names/references",
    )
    entity_type: EntityType = Field(description="Type of entity")
    memory_ids: list[str] = Field(
        default_factory=list,
        description="Memories referencing this entity",
    )
    attributes: dict[str, str] = Field(
        default_factory=dict,
        description="Key-value attributes (role, title, etc.)",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
    )
    merge_count: int = Field(
        default=1,
        ge=1,
        description="Number of mentions merged into this entity",
    )
    user_id: str = Field(description="User who owns this entity")
    org_id: str | None = Field(default=None, description="Optional organization")

    def add_alias(self, alias: str) -> None:
        """Add an alias if not already present."""
        normalized = alias.strip()
        if normalized and normalized.lower() != self.canonical_name.lower():
            if normalized not in self.aliases:
                self.aliases.append(normalized)
                self.updated_at = datetime.now(UTC)

    def add_memory(self, memory_id: str) -> None:
        """Link a memory to this entity."""
        if memory_id not in self.memory_ids:
            self.memory_ids.append(memory_id)
            self.updated_at = datetime.now(UTC)

    def merge_from(self, other: Entity) -> None:
        """Merge another entity into this one.

        Combines aliases, memory_ids, and attributes from the other entity.
        The other entity should be deleted after merging.

        Args:
            other: Entity to merge from.
        """
        # Add other's canonical name as alias
        self.add_alias(other.canonical_name)

        # Merge aliases
        for alias in other.aliases:
            self.add_alias(alias)

        # Merge memory IDs
        for mem_id in other.memory_ids:
            self.add_memory(mem_id)

        # Merge attributes (other's values take precedence for conflicts)
        for key, value in other.attributes.items():
            if key not in self.attributes:
                self.attributes[key] = value

        self.merge_count += other.merge_count
        self.updated_at = datetime.now(UTC)

    def matches_name(self, name: str) -> bool:
        """Check if a name matches this entity.

        Args:
            name: Name to check.

        Returns:
            True if name matches canonical name or any alias.
        """
        normalized = name.strip().lower()
        if self.canonical_name.lower() == normalized:
            return True
        return any(alias.lower() == normalized for alias in self.aliases)


class EntityResolutionResult(BaseModel):
    """Result of entity resolution process.

    Attributes:
        entities: Resolved entities with aliases and memory links.
        new_entities: Count of newly created entities.
        merged_entities: Count of entities merged with existing ones.
        unresolved_mentions: Mentions that couldn't be resolved.
        reasoning: LLM reasoning for resolution decisions.
    """

    model_config = ConfigDict(extra="forbid")

    entities: list[Entity] = Field(default_factory=list)
    new_entities: int = Field(default=0, ge=0)
    merged_entities: int = Field(default=0, ge=0)
    unresolved_mentions: list[EntityMention] = Field(default_factory=list)
    reasoning: str = Field(default="")


class EntityCluster(BaseModel):
    """A cluster of entity mentions believed to be the same entity.

    Used during resolution to group mentions before creating/merging entities.

    Attributes:
        mentions: Entity mentions in this cluster.
        suggested_canonical: LLM-suggested canonical name.
        suggested_type: LLM-suggested entity type.
        confidence: Confidence that these mentions are the same entity.
        reasoning: Why these mentions were clustered together.
    """

    model_config = ConfigDict(extra="forbid")

    mentions: list[EntityMention] = Field(default_factory=list)
    suggested_canonical: str = Field(description="Suggested canonical name")
    suggested_type: EntityType = Field(description="Suggested entity type")
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Clustering confidence",
    )
    reasoning: str = Field(default="", description="Clustering rationale")
