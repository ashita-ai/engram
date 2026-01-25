"""Operations mixin for EngramService.

Provides consolidation, verification, and source tracing methods.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from engram.exceptions import NotFoundError, ValidationError
from engram.logging import get_logger
from engram.models import Episode, HistoryEntry, ProvenanceChain, ProvenanceEvent

from .models import VerificationResult

if TYPE_CHECKING:
    from engram.embeddings import Embedder
    from engram.storage import EngramStorage
    from engram.workflows.backend import WorkflowBackend
    from engram.workflows.consolidation import ConsolidationResult
    from engram.workflows.promotion import SynthesisResult

logger = get_logger(__name__)


class OperationsMixin:
    """Mixin providing consolidation and verification functionality.

    Expects these attributes from the base class:
    - storage: EngramStorage
    - embedder: Embedder
    - workflow_backend: WorkflowBackend
    - _working_memory: list[Episode]
    """

    storage: EngramStorage
    embedder: Embedder
    workflow_backend: WorkflowBackend | None  # Always set after __post_init__
    _working_memory: list[Episode]

    async def get_sources(
        self,
        memory_id: str,
        user_id: str,
    ) -> list[Episode]:
        """Get source episodes for a derived memory.

        Traces a derived memory (StructuredMemory, SemanticMemory, ProceduralMemory)
        back to its source Episode(s).

        Args:
            memory_id: ID of the derived memory.
            user_id: User ID for multi-tenancy isolation.

        Returns:
            List of source Episodes in chronological order.

        Raises:
            ValueError: If memory type cannot be determined from ID prefix.
            KeyError: If memory not found.

        Example:
            ```python
            # Get a structured memory and trace it back to source
            memories = await engram.recall("email", user_id="u1")
            struct = next(m for m in memories if m.memory_type == "structured")
            sources = await engram.get_sources(struct.memory_id, user_id="u1")
            for ep in sources:
                print(f"{ep.timestamp}: {ep.content}")
            ```
        """
        # Determine memory type from ID prefix
        source_episode_ids: list[str] = []

        if memory_id.startswith("struct_"):
            structured = await self.storage.get_structured(memory_id, user_id)
            if structured is None:
                raise NotFoundError("StructuredMemory", memory_id)
            source_episode_ids = [structured.source_episode_id]

        elif memory_id.startswith("sem_"):
            semantic = await self.storage.get_semantic(memory_id, user_id)
            if semantic is None:
                raise NotFoundError("SemanticMemory", memory_id)
            source_episode_ids = semantic.source_episode_ids

        elif memory_id.startswith("proc_"):
            procedural = await self.storage.get_procedural(memory_id, user_id)
            if procedural is None:
                raise NotFoundError("ProceduralMemory", memory_id)
            source_episode_ids = procedural.source_episode_ids

        else:
            raise ValidationError(
                "memory_id",
                f"Cannot determine memory type from ID: {memory_id}. "
                "Expected prefix: struct_, sem_, or proc_",
            )

        # Fetch source episodes
        episodes: list[Episode] = []
        for ep_id in source_episode_ids:
            ep = await self.storage.get_episode(ep_id, user_id)
            if ep is not None:
                episodes.append(ep)

        # Sort by timestamp (chronological order)
        episodes.sort(key=lambda e: e.timestamp)

        return episodes

    async def get_provenance(
        self,
        memory_id: str,
        user_id: str,
    ) -> ProvenanceChain:
        """Get complete provenance chain for a derived memory.

        Traces a derived memory back through its entire derivation chain,
        from source episodes through intermediate memories, recording
        how and when each derivation occurred.

        Args:
            memory_id: ID of the memory to trace.
            user_id: User ID for multi-tenancy isolation.

        Returns:
            ProvenanceChain with complete derivation history.

        Raises:
            ValueError: If memory type cannot be determined from ID prefix.
            KeyError: If memory not found.

        Example:
            ```python
            # Get full provenance for a semantic memory
            chain = await engram.get_provenance("sem_abc123", user_id="u1")
            print(f"Derived via: {chain.derivation_method}")
            print(f"Source episodes: {len(chain.source_episodes)}")
            for event in chain.timeline:
                print(f"  {event.timestamp}: {event.description}")
            ```
        """
        timeline: list[ProvenanceEvent] = []
        source_episodes: list[dict[str, Any]] = []
        intermediate_memories: list[dict[str, Any]] = []

        if memory_id.startswith("struct_"):
            structured = await self.storage.get_structured(memory_id, user_id)
            if structured is None:
                raise NotFoundError("StructuredMemory", memory_id)

            # Get source episode
            episode = await self.storage.get_episode(structured.source_episode_id, user_id)
            if episode:
                source_episodes.append(
                    {
                        "id": episode.id,
                        "content": episode.content,
                        "role": episode.role,
                        "timestamp": episode.timestamp.isoformat(),
                    }
                )
                timeline.append(
                    ProvenanceEvent(
                        timestamp=episode.timestamp,
                        event_type="stored",
                        description=f"Episode stored: {episode.id}",
                        memory_id=episode.id,
                    )
                )

            # Extraction event
            timeline.append(
                ProvenanceEvent(
                    timestamp=structured.derived_at,
                    event_type="extracted",
                    description=f"StructuredMemory extracted via {structured.derivation_method}",
                    memory_id=structured.id,
                    metadata={
                        "mode": structured.mode,
                        "enriched": structured.enriched,
                        "extract_counts": {
                            "emails": len(structured.emails),
                            "phones": len(structured.phones),
                            "urls": len(structured.urls),
                            "people": len(structured.people),
                            "preferences": len(structured.preferences),
                            "negations": len(structured.negations),
                        },
                    },
                )
            )

            return ProvenanceChain(
                memory_id=memory_id,
                memory_type="structured",
                derivation_method=structured.derivation_method,
                derivation_reasoning=None,
                derived_at=structured.derived_at,
                source_episodes=source_episodes,
                intermediate_memories=[],
                timeline=sorted(timeline, key=lambda e: e.timestamp),
            )

        elif memory_id.startswith("sem_"):
            semantic = await self.storage.get_semantic(memory_id, user_id)
            if semantic is None:
                raise NotFoundError("SemanticMemory", memory_id)

            # Get source episodes
            for ep_id in semantic.source_episode_ids:
                episode = await self.storage.get_episode(ep_id, user_id)
                if episode:
                    source_episodes.append(
                        {
                            "id": episode.id,
                            "content": episode.content,
                            "role": episode.role,
                            "timestamp": episode.timestamp.isoformat(),
                        }
                    )
                    timeline.append(
                        ProvenanceEvent(
                            timestamp=episode.timestamp,
                            event_type="stored",
                            description=f"Episode stored: {episode.id}",
                            memory_id=episode.id,
                        )
                    )

            # Check if there are intermediate structured memories
            for ep_id in semantic.source_episode_ids:
                struct = await self.storage.get_structured_for_episode(ep_id, user_id)
                if struct:
                    intermediate_memories.append(
                        {
                            "id": struct.id,
                            "type": "structured",
                            "summary": struct.summary,
                            "derivation_method": struct.derivation_method,
                            "derived_at": struct.derived_at.isoformat(),
                        }
                    )
                    timeline.append(
                        ProvenanceEvent(
                            timestamp=struct.derived_at,
                            event_type="extracted",
                            description=f"StructuredMemory extracted: {struct.id}",
                            memory_id=struct.id,
                        )
                    )

            # Consolidation event
            timeline.append(
                ProvenanceEvent(
                    timestamp=semantic.derived_at,
                    event_type="inferred",
                    description=f"SemanticMemory consolidated via {semantic.derivation_method}",
                    memory_id=semantic.id,
                    metadata={
                        "source_count": len(semantic.source_episode_ids),
                        "consolidation_strength": semantic.consolidation_strength,
                    },
                )
            )

            return ProvenanceChain(
                memory_id=memory_id,
                memory_type="semantic",
                derivation_method=semantic.derivation_method,
                derivation_reasoning=semantic.derivation_reasoning,
                derived_at=semantic.derived_at,
                source_episodes=source_episodes,
                intermediate_memories=intermediate_memories,
                timeline=sorted(timeline, key=lambda e: e.timestamp),
            )

        elif memory_id.startswith("proc_"):
            procedural = await self.storage.get_procedural(memory_id, user_id)
            if procedural is None:
                raise NotFoundError("ProceduralMemory", memory_id)

            # Get source episodes
            for ep_id in procedural.source_episode_ids:
                episode = await self.storage.get_episode(ep_id, user_id)
                if episode:
                    source_episodes.append(
                        {
                            "id": episode.id,
                            "content": episode.content,
                            "role": episode.role,
                            "timestamp": episode.timestamp.isoformat(),
                        }
                    )
                    timeline.append(
                        ProvenanceEvent(
                            timestamp=episode.timestamp,
                            event_type="stored",
                            description=f"Episode stored: {episode.id}",
                            memory_id=episode.id,
                        )
                    )

            # Get intermediate semantic memories
            for sem_id in procedural.source_semantic_ids:
                semantic = await self.storage.get_semantic(sem_id, user_id)
                if semantic:
                    intermediate_memories.append(
                        {
                            "id": semantic.id,
                            "type": "semantic",
                            "content": semantic.content,
                            "derivation_method": semantic.derivation_method,
                            "derived_at": semantic.derived_at.isoformat(),
                        }
                    )
                    timeline.append(
                        ProvenanceEvent(
                            timestamp=semantic.derived_at,
                            event_type="inferred",
                            description=f"SemanticMemory consolidated: {semantic.id}",
                            memory_id=semantic.id,
                        )
                    )

            # Synthesis event
            timeline.append(
                ProvenanceEvent(
                    timestamp=procedural.derived_at,
                    event_type="synthesized",
                    description=f"ProceduralMemory synthesized via {procedural.derivation_method}",
                    memory_id=procedural.id,
                    metadata={
                        "source_semantic_count": len(procedural.source_semantic_ids),
                        "consolidation_strength": procedural.consolidation_strength,
                    },
                )
            )

            return ProvenanceChain(
                memory_id=memory_id,
                memory_type="procedural",
                derivation_method=procedural.derivation_method,
                derivation_reasoning=procedural.derivation_reasoning,
                derived_at=procedural.derived_at,
                source_episodes=source_episodes,
                intermediate_memories=intermediate_memories,
                timeline=sorted(timeline, key=lambda e: e.timestamp),
            )

        else:
            raise ValidationError(
                "memory_id",
                f"Cannot determine memory type from ID: {memory_id}. "
                "Expected prefix: struct_, sem_, or proc_",
            )

    async def verify(
        self,
        memory_id: str,
        user_id: str,
    ) -> VerificationResult:
        """Verify a memory against its source episodes.

        Traces a derived memory back to its source episode(s) and
        generates a human-readable explanation of how it was derived.

        Args:
            memory_id: ID of the memory to verify.
            user_id: User ID for multi-tenancy isolation.

        Returns:
            VerificationResult with source traceability and explanation.

        Raises:
            ValueError: If memory type cannot be determined from ID prefix.
            KeyError: If memory not found.

        Example:
            ```python
            # Verify a structured memory and see its derivation
            result = await engram.verify("struct_abc123", user_id="u1")
            print(result.explanation)
            # "Pattern-matched from source episode(s).
            #  Confidence: 0.90: extracted, 1 source, confirmed just now"

            for ep in result.source_episodes:
                print(f"Source: {ep['content']}")
            ```
        """
        from engram.models import ConfidenceScore, ExtractionMethod

        # Get memory and its metadata based on type
        content: str
        confidence: ConfidenceScore
        memory_type: str
        category: str = ""

        if memory_id.startswith("struct_"):
            memory_type = "structured"
            structured = await self.storage.get_structured(memory_id, user_id)
            if structured is None:
                raise NotFoundError("StructuredMemory", memory_id)
            content = structured.to_embedding_text() or structured.summary
            confidence = structured.confidence
            category = "structured" if structured.enriched else "regex-extracted"

        elif memory_id.startswith("sem_"):
            memory_type = "semantic"
            semantic = await self.storage.get_semantic(memory_id, user_id)
            if semantic is None:
                raise NotFoundError("SemanticMemory", memory_id)
            content = semantic.content
            confidence = semantic.confidence

        elif memory_id.startswith("proc_"):
            memory_type = "procedural"
            procedural = await self.storage.get_procedural(memory_id, user_id)
            if procedural is None:
                raise NotFoundError("ProceduralMemory", memory_id)
            content = procedural.content
            confidence = procedural.confidence

        else:
            raise ValidationError(
                "memory_id",
                f"Cannot determine memory type from ID: {memory_id}. "
                "Expected prefix: struct_, sem_, or proc_",
            )

        # Get source episodes
        source_episodes = await self.get_sources(memory_id, user_id)

        # Build source episode summaries
        episode_details: list[dict[str, Any]] = []
        for ep in source_episodes:
            episode_details.append(
                {
                    "id": ep.id,
                    "content": ep.content,
                    "role": ep.role,
                    "timestamp": ep.timestamp.isoformat(),
                }
            )

        # Generate explanation
        explanation_parts: list[str] = []

        # Describe extraction
        if confidence.extraction_method == ExtractionMethod.EXTRACTED:
            if category:
                explanation_parts.append(f"Pattern-matched {category} from source episode(s).")
            else:
                explanation_parts.append("Pattern-matched from source episode(s).")
        elif confidence.extraction_method == ExtractionMethod.INFERRED:
            explanation_parts.append("LLM-inferred from source episode(s).")
        else:  # VERBATIM
            explanation_parts.append("Verbatim quote from source episode.")

        # Source info
        if source_episodes:
            if len(source_episodes) == 1:
                ep = source_episodes[0]
                explanation_parts.append(
                    f"Source: {ep.id} ({ep.timestamp.strftime('%Y-%m-%d %H:%M')})."
                )
            else:
                explanation_parts.append(f"Sources: {len(source_episodes)} episodes.")

        # Confidence explanation
        explanation_parts.append(f"Confidence: {confidence.explain()}")

        explanation = " ".join(explanation_parts)

        return VerificationResult(
            memory_id=memory_id,
            memory_type=memory_type,
            content=content,
            verified=len(source_episodes) > 0,
            source_episodes=episode_details,
            extraction_method=confidence.extraction_method.value,
            confidence=confidence.value,
            explanation=explanation,
        )

    async def consolidate(
        self,
        user_id: str,
        org_id: str | None = None,
    ) -> ConsolidationResult:
        """Consolidate unsummarized episodes into a semantic summary.

        Implements hierarchical compression: N episodes → 1 semantic memory.
        Uses map-reduce for large batches that exceed token limits.

        This method:
        1. Fetches ALL unsummarized episodes for the user
        2. Creates ONE coherent semantic summary via LLM
        3. Marks episodes as summarized with link to the summary
        4. Links new summary to similar existing memories

        Based on Complementary Learning Systems (McClelland et al., 1995):
        hippocampus (episodic) → neocortex (semantic) transfer with compression.

        Uses the configured workflow backend for execution, which may provide
        durability guarantees depending on the backend (DBOS, Temporal, Prefect).

        Args:
            user_id: User ID for multi-tenancy isolation.
            org_id: Optional organization ID for further isolation.

        Returns:
            ConsolidationResult with processing statistics:
            - episodes_processed: Number of episodes summarized
            - semantic_memories_created: Number of summaries created (typically 1)
            - links_created: Number of memory links built
            - compression_ratio: Ratio of input episodes to output memories

        Example:
            ```python
            # Consolidate all unsummarized episodes
            result = await engram.consolidate(user_id="u1")
            print(f"{result.episodes_processed} episodes → {result.semantic_memories_created} summary")
            print(f"Compression ratio: {result.compression_ratio:.1f}:1")
            ```
        """
        assert self.workflow_backend is not None, "workflow_backend not initialized"
        return await self.workflow_backend.run_consolidation(
            storage=self.storage,
            embedder=self.embedder,
            user_id=user_id,
            org_id=org_id,
        )

    async def create_procedural(
        self,
        user_id: str,
        org_id: str | None = None,
    ) -> SynthesisResult:
        """Synthesize a procedural memory from all semantic memories.

        Creates or updates ONE procedural memory that captures the user's
        behavioral patterns, preferences, and communication style.

        This method:
        1. Fetches ALL semantic memories for the user
        2. Uses LLM to synthesize a behavioral profile
        3. Creates or replaces the user's procedural memory
        4. Links procedural to all source semantic IDs

        Design decision: ONE procedural per user (replaces existing).

        Uses the configured workflow backend for execution, which may provide
        durability guarantees depending on the backend (DBOS, Temporal, Prefect).

        Args:
            user_id: User ID for multi-tenancy isolation.
            org_id: Optional organization ID for further isolation.

        Returns:
            SynthesisResult with processing statistics:
            - semantics_analyzed: Number of semantic memories analyzed
            - procedural_created: True if new procedural was created
            - procedural_updated: True if existing procedural was updated
            - procedural_id: ID of the created/updated procedural

        Example:
            ```python
            # Create/update the user's behavioral profile
            result = await engram.create_procedural(user_id="u1")
            if result.procedural_created:
                print(f"Created procedural: {result.procedural_id}")
            elif result.procedural_updated:
                print(f"Updated procedural: {result.procedural_id}")
            ```
        """
        assert self.workflow_backend is not None, "workflow_backend not initialized"
        return await self.workflow_backend.run_promotion(
            storage=self.storage,
            embedder=self.embedder,
            user_id=user_id,
            org_id=org_id,
        )

    async def get_memory_history(
        self,
        memory_id: str,
        user_id: str,
        since: datetime | None = None,
        limit: int = 100,
    ) -> list[HistoryEntry]:
        """Get change history for a specific memory.

        Returns all changes made to a memory over time, enabling full
        audit trail and debugging. Each entry shows before/after state,
        what triggered the change, and when.

        Args:
            memory_id: ID of the memory to get history for.
            user_id: User ID for multi-tenancy isolation.
            since: Optional timestamp to filter entries after.
            limit: Maximum entries to return (default 100).

        Returns:
            List of HistoryEntry sorted by timestamp (newest first).

        Example:
            ```python
            history = await engram.get_memory_history("sem_abc123", user_id="u1")
            for entry in history:
                print(f"{entry.timestamp}: {entry.change_type} ({entry.trigger})")
                if entry.diff:
                    print(f"  Changed: {list(entry.diff.keys())}")
            ```
        """
        return await self.storage.get_memory_history(
            memory_id=memory_id,
            user_id=user_id,
            since=since,
            limit=limit,
        )

    async def get_user_history(
        self,
        user_id: str,
        org_id: str | None = None,
        memory_type: str | None = None,
        change_type: str | None = None,
        since: datetime | None = None,
        limit: int = 100,
    ) -> list[HistoryEntry]:
        """Get change history for all memories of a user.

        Returns a timeline of all changes across all memory types,
        useful for understanding how the user's memory store evolved.

        Args:
            user_id: User to get history for.
            org_id: Optional organization filter.
            memory_type: Optional filter by type (structured, semantic, procedural).
            change_type: Optional filter by change type (created, updated, etc.).
            since: Optional timestamp to filter entries after.
            limit: Maximum entries to return (default 100).

        Returns:
            List of HistoryEntry sorted by timestamp (newest first).

        Example:
            ```python
            # Get all recent changes
            history = await engram.get_user_history(user_id="u1", limit=50)

            # Get only consolidation-triggered changes
            consolidations = await engram.get_user_history(
                user_id="u1",
                memory_type="semantic",
                change_type="strengthened",
            )
            ```
        """
        return await self.storage.get_user_history(
            user_id=user_id,
            org_id=org_id,
            memory_type=memory_type,
            change_type=change_type,
            since=since,
            limit=limit,
        )


__all__ = ["OperationsMixin"]
