"""SemanticMemory model - LLM-inferred knowledge."""

from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field

from .base import ConfidenceScore, MemoryBase, generate_id


class EvolutionEntry(BaseModel):
    """Record of a memory evolution event.

    Tracks when and how a memory was modified during consolidation.

    Attributes:
        timestamp: When the evolution occurred.
        trigger_memory_id: ID of the new memory that triggered this evolution.
        field_changed: Which field was modified.
        old_value: Previous value (as string for audit).
        new_value: New value (as string for audit).
        reason: Why this evolution was applied.
    """

    model_config = ConfigDict(extra="forbid")

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="When the evolution occurred",
    )
    trigger_memory_id: str = Field(
        description="ID of the new memory that triggered this evolution",
    )
    field_changed: str = Field(
        description="Which field was modified",
    )
    old_value: str = Field(
        default="",
        description="Previous value",
    )
    new_value: str = Field(
        description="New value",
    )
    reason: str = Field(
        default="",
        description="Why this evolution was applied",
    )


class SemanticMemory(MemoryBase):
    """Semantic memory inferred by LLM from conversations.

    Semantic memories are extracted during background consolidation.
    They represent inferred knowledge, preferences, and context that
    isn't explicitly stated but can be derived from conversations.

    Consolidation strength tracks how well-established a memory is,
    based on the Testing Effect research (Roediger & Karpicke 2006):
    memories that are repeatedly involved in retrieval/consolidation
    become stronger and more stable.

    - 0.0: Newly created, not yet reinforced
    - 1.0: Highly consolidated, repeatedly reinforced

    A-MEM inspired features:
    - keywords: Key terms for this memory (improves linking)
    - tags: Category labels (e.g., "preference", "technical", "personal")
    - context: Domain/theme classification
    - retrieval_count: How often accessed (activation tracking)
    - last_accessed: When last retrieved
    - evolution_history: Audit trail of memory modifications

    Attributes:
        content: The inferred semantic content.
        source_episode_ids: Episodes this was derived from.
        related_ids: Links to related memories (for multi-hop reasoning).
        event_at: When the underlying facts were true.
        derived_at: When we inferred this memory.
        confidence: Composite confidence score.
        consolidation_strength: How well-established (0=new, 1=strong).
        consolidation_passes: How many times this has been refined.
        keywords: Extracted key terms for this memory.
        tags: Category labels for classification.
        context: Domain or theme description.
        retrieval_count: Number of times this memory has been accessed.
        last_accessed: When this memory was last retrieved.
        evolution_history: Audit trail of modifications.
    """

    id: str = Field(default_factory=lambda: generate_id("sem"))
    content: str = Field(description="The inferred semantic content")
    source_episode_ids: list[str] = Field(
        default_factory=list,
        description="IDs of Episodes this was derived from",
    )
    # Provenance tracking
    derivation_method: str = Field(
        default="llm:unknown",
        description="How this memory was derived (e.g., 'llm:gpt-4o-mini')",
    )
    derivation_reasoning: str | None = Field(
        default=None,
        description="LLM's explanation for why this was extracted",
    )
    related_ids: list[str] = Field(
        default_factory=list,
        description="IDs of related memories for multi-hop reasoning",
    )
    link_types: dict[str, str] = Field(
        default_factory=dict,
        description="Maps memory_id to link type (related, supersedes, contradicts)",
    )
    event_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="When the underlying facts were true",
    )
    derived_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="When we inferred this memory",
    )
    confidence: ConfidenceScore = Field(
        default_factory=lambda: ConfidenceScore.for_inferred(0.6),
        description="Composite confidence score",
    )
    consolidation_strength: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="How well-established via repeated consolidation (0=new, 1=strong)",
    )
    consolidation_passes: int = Field(
        default=0,
        ge=0,
        description="Number of consolidation passes",
    )
    archived: bool = Field(
        default=False,
        description="Whether this memory is archived (low confidence)",
    )
    # A-MEM inspired fields
    keywords: list[str] = Field(
        default_factory=list,
        description="Extracted key terms for this memory",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Category labels (e.g., preference, technical, personal)",
    )
    context: str = Field(
        default="",
        description="Domain or theme description",
    )
    # Activation tracking
    retrieval_count: int = Field(
        default=0,
        ge=0,
        description="Number of times this memory has been accessed",
    )
    last_accessed: datetime | None = Field(
        default=None,
        description="When this memory was last retrieved",
    )
    # Evolution history
    evolution_history: list[EvolutionEntry] = Field(
        default_factory=list,
        description="Audit trail of memory modifications",
    )

    def add_link(self, memory_id: str, link_type: str = "related") -> None:
        """Add a link to a related memory.

        Args:
            memory_id: ID of the memory to link to.
            link_type: Type of link (related, supersedes, contradicts).
        """
        if memory_id not in self.related_ids:
            self.related_ids.append(memory_id)
        self.link_types[memory_id] = link_type

    def remove_link(self, memory_id: str) -> bool:
        """Remove a link to a related memory.

        Args:
            memory_id: ID of the memory to unlink.

        Returns:
            True if the link was removed, False if it didn't exist.
        """
        if memory_id in self.related_ids:
            self.related_ids.remove(memory_id)
            self.link_types.pop(memory_id, None)
            return True
        return False

    def get_link_type(self, memory_id: str) -> str | None:
        """Get the link type for a related memory.

        Args:
            memory_id: ID of the linked memory.

        Returns:
            Link type or None if not linked.
        """
        return self.link_types.get(memory_id)

    def strengthen(self, delta: float = 0.1) -> None:
        """Strengthen memory through consolidation involvement.

        Based on Testing Effect research: memories repeatedly involved
        in retrieval/consolidation become stronger and more stable.
        Tested subjects forgot 13% after 1 week vs 52% for study-only.
        See: Roediger & Karpicke (2006), PMID 26151629.

        Both consolidation_strength and consolidation_passes are incremented
        together to keep them in sync. Strength increases by delta (0.1),
        passes increments by 1.
        """
        self.consolidation_strength = min(1.0, self.consolidation_strength + delta)
        self.consolidation_passes += 1

    def weaken(self, delta: float = 0.1) -> None:
        """Weaken memory (pruned or contradicted during consolidation)."""
        self.consolidation_strength = max(0.0, self.consolidation_strength - delta)

    def record_access(self) -> None:
        """Record that this memory was accessed (activation tracking).

        Increments retrieval_count and updates last_accessed timestamp.
        Called by storage layer on search hits.
        """
        self.retrieval_count += 1
        self.last_accessed = datetime.now(UTC)

    def add_tag(self, tag: str) -> None:
        """Add a tag if not already present."""
        if tag not in self.tags:
            self.tags.append(tag)

    def add_keyword(self, keyword: str) -> None:
        """Add a keyword if not already present."""
        keyword_lower = keyword.lower()
        if keyword_lower not in [k.lower() for k in self.keywords]:
            self.keywords.append(keyword)

    def evolve(
        self,
        trigger_memory_id: str,
        field: str,
        new_value: str,
        reason: str = "",
    ) -> None:
        """Record an evolution event (A-MEM style memory update).

        Only metadata fields can be evolved (tags, keywords, context).
        Content is immutable.

        Args:
            trigger_memory_id: ID of the new memory that triggered this evolution.
            field: Which field was modified (tags, keywords, context).
            new_value: New value as string representation.
            reason: Why this evolution was applied.

        Raises:
            ValueError: If attempting to evolve content (immutable).
        """
        if field == "content":
            raise ValueError("Cannot evolve content - it is immutable")

        # Get old value
        old_value = ""
        if field == "tags":
            old_value = ",".join(self.tags)
        elif field == "keywords":
            old_value = ",".join(self.keywords)
        elif field == "context":
            old_value = self.context

        # Record the evolution
        entry = EvolutionEntry(
            trigger_memory_id=trigger_memory_id,
            field_changed=field,
            old_value=old_value,
            new_value=new_value,
            reason=reason,
        )
        self.evolution_history.append(entry)

        # Apply the change
        if field == "tags":
            new_tags = [t.strip() for t in new_value.split(",") if t.strip()]
            for tag in new_tags:
                self.add_tag(tag)
        elif field == "keywords":
            new_keywords = [k.strip() for k in new_value.split(",") if k.strip()]
            for keyword in new_keywords:
                self.add_keyword(keyword)
        elif field == "context":
            if not self.context:
                self.context = new_value
            elif new_value not in self.context:
                self.context = f"{self.context}; {new_value}"

    def __str__(self) -> str:
        """String representation showing content preview."""
        content_preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return f"SemanticMemory({content_preview!r})"
