"""Provenance models for memory derivation tracking.

Provenance enables auditing any derived memory back to its source episodes,
tracking the full chain of how knowledge was extracted and inferred.
"""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

ProvenanceEventType = Literal[
    "stored",  # Episode stored (ground truth)
    "extracted",  # StructuredMemory created (regex/LLM extraction)
    "inferred",  # SemanticMemory created (cross-episode synthesis)
    "synthesized",  # ProceduralMemory created (behavioral pattern)
    "linked",  # Memory linked to another
    "strengthened",  # Memory strengthened during consolidation
    "evolved",  # Memory metadata evolved
]


class ProvenanceEvent(BaseModel):
    """A single event in a memory's derivation history.

    Attributes:
        timestamp: When the event occurred.
        event_type: Type of derivation event.
        description: Human-readable description of what happened.
        memory_id: ID of the memory involved.
        metadata: Additional event-specific data.
    """

    model_config = ConfigDict(extra="forbid")

    timestamp: datetime = Field(description="When the event occurred")
    event_type: ProvenanceEventType = Field(description="Type of event")
    description: str = Field(description="Human-readable description")
    memory_id: str | None = Field(
        default=None,
        description="ID of the memory involved (if applicable)",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional event-specific data",
    )


class ProvenanceChain(BaseModel):
    """Complete derivation chain for a memory.

    Tracks the full provenance from source episodes through intermediate
    memories to the final derived memory, enabling full auditability.

    Attributes:
        memory_id: ID of the memory being traced.
        memory_type: Type of memory (episode, structured, semantic, procedural).
        derivation_method: How this memory was derived.
        derivation_reasoning: LLM's explanation (if applicable).
        source_episodes: Original episodes this was derived from.
        intermediate_memories: Intermediate derivations (StructuredMemory, SemanticMemory).
        timeline: Chronological list of derivation events.
    """

    model_config = ConfigDict(extra="forbid")

    memory_id: str = Field(description="ID of the memory being traced")
    memory_type: str = Field(description="Type of memory")
    derivation_method: str | None = Field(
        default=None,
        description="How this memory was derived",
    )
    derivation_reasoning: str | None = Field(
        default=None,
        description="LLM's explanation for the derivation",
    )
    derived_at: datetime | None = Field(
        default=None,
        description="When this memory was derived",
    )

    # Source chain
    source_episodes: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Original episodes this was derived from",
    )
    intermediate_memories: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Intermediate derivations (StructuredMemory, SemanticMemory)",
    )

    # Timeline
    timeline: list[ProvenanceEvent] = Field(
        default_factory=list,
        description="Chronological list of derivation events",
    )


__all__ = [
    "ProvenanceChain",
    "ProvenanceEvent",
    "ProvenanceEventType",
]
