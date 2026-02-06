"""Consolidation workflow for creating semantic summaries from episodes.

This workflow implements hierarchical compression aligned with cognitive science:
- Episodic memories are immutable ground truth
- Consolidation compresses N episodes into 1 semantic summary
- Uses map-reduce for large batches

Based on Complementary Learning Systems (McClelland et al., 1995):
hippocampus (episodic) → neocortex (semantic) transfer with compression.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from engram.models.base import ConfidenceScore

if TYPE_CHECKING:
    from engram.embeddings import Embedder
    from engram.models import SemanticMemory
    from engram.storage import EngramStorage

logger = logging.getLogger(__name__)

# Maximum episodes per LLM call before chunking
MAX_EPISODES_PER_CHUNK = 20

# Minimum token efficiency - skip summarization if already concise
MIN_COMPRESSION_RATIO = 2.0


class SummaryOutput(BaseModel):
    """Structured output from the summarization LLM agent.

    Includes both synthesis AND confidence assessment in a single LLM call.
    """

    model_config = ConfigDict(extra="forbid")

    summary: str = Field(
        description="Knowledge summary of the episodes (2-5 sentences). "
        "States facts directly without 'The user...' framing.",
    )
    key_facts: list[str] = Field(
        default_factory=list,
        description="3-8 concrete facts extracted from the episodes. "
        "Each should be a standalone fact stated directly "
        "(e.g., 'PostgreSQL is the primary database' not 'The user prefers PostgreSQL').",
    )
    keywords: list[str] = Field(
        default_factory=list,
        description="Key terms for retrieval: names, tools, versions, technologies (5-10 words)",
    )
    context: str = Field(
        default="",
        description="Domain/theme (e.g., programming, work, personal)",
    )

    # Confidence assessment (returned alongside synthesis)
    confidence: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Overall confidence in the synthesis (0.0-1.0)",
    )
    confidence_reasoning: str = Field(
        default="",
        description="Brief explanation of the confidence assessment",
    )


class MapReduceSummary(BaseModel):
    """Output from map-reduce summarization of multiple chunks."""

    model_config = ConfigDict(extra="forbid")

    summary: str = Field(description="Final unified summary")
    keywords: list[str] = Field(default_factory=list)
    context: str = Field(default="")

    # Confidence assessment
    confidence: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Overall confidence in the synthesis (0.0-1.0)",
    )
    confidence_reasoning: str = Field(
        default="",
        description="Brief explanation of the confidence assessment",
    )


class ConsolidationResult(BaseModel):
    """Result of a consolidation workflow run.

    Attributes:
        episodes_processed: Number of episodes that were summarized.
        semantic_memories_created: Number of semantic summaries created (typically 1).
        links_created: Number of memory links built.
        compression_ratio: Ratio of input episodes to output memories.
    """

    model_config = ConfigDict(extra="forbid")

    episodes_processed: int = Field(ge=0)
    semantic_memories_created: int = Field(ge=0)
    links_created: int = Field(ge=0, default=0)
    compression_ratio: float = Field(ge=0.0, default=0.0)


def format_episodes_for_llm(episodes: list[dict[str, str]]) -> str:
    """Format episodes for LLM processing.

    Args:
        episodes: List of dicts with 'id', 'role', 'content' keys.

    Returns:
        Formatted text for LLM input.
    """
    lines = ["# Conversation Episodes to Summarize\n"]
    for ep in episodes:
        lines.append(f"[{ep['role'].upper()}] ({ep['id']})")
        lines.append(ep["content"])
        lines.append("")
    return "\n".join(lines)


def format_structured_for_llm(structured_memories: list[dict[str, str | list[str]]]) -> str:
    """Format StructuredMemories for LLM consolidation.

    StructuredMemories already contain per-episode summaries and entities,
    so consolidation just needs to synthesize them.

    Args:
        structured_memories: List of dicts with keys from StructuredMemory.

    Returns:
        Formatted text for LLM input.
    """
    lines = ["# Per-Episode Extractions to Consolidate\n"]
    for struct in structured_memories:
        lines.append(f"## Episode: {struct['source_episode_id']} ({struct['id']})")
        lines.append(f"Summary: {struct['summary']}")

        if struct.get("keywords"):
            keywords = struct["keywords"]
            if isinstance(keywords, list):
                lines.append(f"Keywords: {', '.join(keywords)}")

        if struct.get("people"):
            lines.append(f"People: {struct['people']}")

        if struct.get("organizations"):
            orgs = struct["organizations"]
            if isinstance(orgs, list):
                lines.append(f"Organizations: {', '.join(orgs)}")

        if struct.get("preferences"):
            lines.append(f"Preferences: {struct['preferences']}")

        if struct.get("negations"):
            lines.append(f"Negations: {struct['negations']}")

        lines.append("")

    return "\n".join(lines)


def _format_existing_memories_for_llm(memories: list[SemanticMemory]) -> str:
    """Format existing memories for LLM context.

    Provides the LLM with existing summaries so it can avoid redundancy
    and identify what's new.

    Args:
        memories: List of existing semantic memories.

    Returns:
        Formatted text for LLM input.
    """
    if not memories:
        return ""

    lines = ["\n# Existing Summaries (avoid redundancy)\n"]
    for mem in memories[:10]:  # Limit to avoid token overflow
        lines.append(f"- {mem.content}")
    return "\n".join(lines)


async def _summarize_chunk(
    episodes: list[dict[str, str]],
    existing_context: str,
) -> SummaryOutput:
    """Summarize a chunk of episodes using the LLM.

    Args:
        episodes: Episodes to summarize.
        existing_context: Existing memories for context.

    Returns:
        SummaryOutput with summary and metadata.
    """
    from pydantic_ai import Agent

    from engram.config import settings

    formatted_text = format_episodes_for_llm(episodes) + existing_context

    summarization_prompt = """You are extracting knowledge from conversation episodes.

Create ONE summary that captures concrete, reusable knowledge:
- Specific facts: names, tools, versions, configurations, constraints
- Decisions and their rationale (why something was chosen or rejected)
- Technical details: dependencies, architecture choices, requirements
- Corrections and negations (what was ruled out and why)

The summary should be:
- Concise (2-5 sentences)
- Stated directly as facts, not as observations about a person
- Focused on lasting/stable knowledge, not transient details

Examples:
- BAD: "The user mentioned they prefer PostgreSQL for their database needs."
- GOOD: "PostgreSQL is the primary database. MongoDB was evaluated and rejected due to schema requirements."
- BAD: "The user is deeply engaged in developing their AI platform."
- GOOD: "pgstream should target v0.9.7 because schemalog breaks on v1.0.0."

Be conservative - only include information that appears in the episodes.

CONFIDENCE ASSESSMENT:
After creating the summary, assess your confidence in the synthesis (0.0-1.0):
- 0.9-1.0: Strong source agreement, directly supported by multiple episodes
- 0.7-0.9: Good source agreement, clearly implied by episodes
- 0.5-0.7: Reasonable inference, some gaps in evidence
- 0.3-0.5: Speculative, weak source agreement
- 0.0-0.3: Sources conflict or don't clearly support the summary

Synthesis should generally score lower than extraction. Be conservative."""

    from engram.workflows.llm_utils import run_agent_with_retry

    agent: Agent[None, SummaryOutput] = Agent(
        settings.consolidation_model,
        output_type=SummaryOutput,
        instructions=summarization_prompt,
    )

    return await run_agent_with_retry(agent, formatted_text)


async def _reduce_summaries(summaries: list[SummaryOutput]) -> MapReduceSummary:
    """Reduce multiple chunk summaries into one final summary.

    Args:
        summaries: List of chunk summaries to combine.

    Returns:
        MapReduceSummary with unified summary.
    """
    from pydantic_ai import Agent

    from engram.config import settings

    # Format summaries for reduction
    lines = ["# Summaries to Combine\n"]
    all_keywords: set[str] = set()
    contexts: list[str] = []

    for i, summary in enumerate(summaries, 1):
        lines.append(f"## Summary {i}")
        lines.append(summary.summary)
        lines.append("")
        all_keywords.update(summary.keywords)
        if summary.context:
            contexts.append(summary.context)

    formatted_text = "\n".join(lines)

    reduce_prompt = """You are combining multiple knowledge summaries into one unified summary.

Create a single coherent summary that:
- Integrates all key information without redundancy
- Preserves specific technical details (versions, names, constraints, rationale)
- Is 3-6 sentences total
- States facts directly, not as observations about a person
- Does not generalize specific details into vague statements

Do not add information that wasn't in the original summaries.

CONFIDENCE ASSESSMENT:
After combining, assess your confidence in the unified summary (0.0-1.0):
- 0.9-1.0: Summaries strongly agree, clear integration
- 0.7-0.9: Good agreement, minor gaps
- 0.5-0.7: Some disagreement or gaps
- 0.3-0.5: Significant disagreement
- 0.0-0.3: Summaries conflict or don't support the conclusion

Be conservative - lower confidence is better than false certainty."""

    from engram.workflows.llm_utils import run_agent_with_retry

    agent: Agent[None, MapReduceSummary] = Agent(
        settings.consolidation_model,
        output_type=MapReduceSummary,
        instructions=reduce_prompt,
    )

    output = await run_agent_with_retry(agent, formatted_text)

    # Merge keywords from all chunks
    output.keywords = list(all_keywords.union(set(output.keywords)))[:15]

    # Use most common context or combine
    if contexts:
        output.context = output.context or contexts[0]

    return output


async def _find_semantically_similar_memories(
    query_embedding: list[float],
    storage: EngramStorage,
    user_id: str,
    org_id: str | None,
    limit: int = 5,
    min_score: float = 0.7,
) -> list[SemanticMemory]:
    """Find semantically similar memories using embedding search.

    Args:
        query_embedding: Embedding vector to search with.
        storage: Storage instance.
        user_id: User ID for isolation.
        org_id: Optional org ID.
        limit: Max results to return.
        min_score: Minimum similarity score threshold.

    Returns:
        List of SemanticMemory objects above the similarity threshold.
    """
    results = await storage.search_semantic(
        query_vector=query_embedding,
        user_id=user_id,
        org_id=org_id,
        limit=limit,
    )

    # Filter by minimum score
    return [r.memory for r in results if r.score >= min_score]


async def run_consolidation(
    storage: EngramStorage,
    embedder: Embedder,
    user_id: str,
    org_id: str,
    consolidation_passes: int = 1,
    similarity_threshold: float = 0.7,
    use_llm_linking: bool = False,
) -> ConsolidationResult:
    """Run the consolidation workflow.

    Implements hierarchical compression: N episodes → 1 semantic summary.
    Uses map-reduce for large batches that exceed token limits.
    Scoped to a single org/project to prevent cross-project bleed.

    This workflow:
    1. Fetches ALL unsummarized episodes within the org
    2. Gets existing summaries for context (within the org)
    3. Chunks episodes if needed (map phase)
    4. Creates summaries per chunk
    5. Reduces to ONE final summary (reduce phase)
    6. Stores semantic memory with source episode links
    7. Marks episodes as summarized

    Args:
        storage: EngramStorage instance.
        embedder: Embedder for generating vectors.
        user_id: User ID for multi-tenancy.
        org_id: Organization/project ID for isolation.
        consolidation_passes: Number of times to run (typically 1).
        similarity_threshold: Min similarity score for automatic linking.
        use_llm_linking: Use LLM to discover richer relationship types
            (causal, temporal, contradicts, etc.) beyond embedding similarity.

    Returns:
        ConsolidationResult with processing statistics.
    """
    from engram.models import SemanticMemory

    total_episodes = 0
    total_memories = 0
    total_links = 0

    for pass_num in range(consolidation_passes):
        # 1. Fetch ALL unsummarized episodes
        episodes = await storage.get_unsummarized_episodes(
            user_id=user_id,
            org_id=org_id,
            limit=None,  # Get all
        )

        if not episodes:
            logger.info(f"Pass {pass_num + 1}: No unsummarized episodes found")
            continue

        logger.info(f"Pass {pass_num + 1}: Processing {len(episodes)} unsummarized episodes")

        # Filter out system prompts - they are operational metadata
        user_episodes = [ep for ep in episodes if ep.role != "system"]
        system_episodes = [ep for ep in episodes if ep.role == "system"]
        if system_episodes:
            logger.debug(f"Excluding {len(system_episodes)} system prompts from summarization")

        if not user_episodes:
            # Only system prompts - mark them as summarized and continue
            all_episode_ids = [ep.id for ep in episodes]
            # Create a placeholder summary for system-only batches
            placeholder = SemanticMemory(
                content="System prompt batch (no user content)",
                source_episode_ids=all_episode_ids,
                user_id=user_id,
                org_id=org_id,
                embedding=await embedder.embed("system prompt"),
            )
            await storage.store_semantic(placeholder)
            await storage.mark_episodes_summarized(all_episode_ids, user_id, placeholder.id)
            total_episodes += len(episodes)
            total_memories += 1
            continue

        # 2. Get existing summaries for context (avoid redundancy)
        existing_memories = await storage.list_semantic_memories(user_id, org_id)
        existing_context = _format_existing_memories_for_llm(existing_memories)

        # 3. Format episodes for LLM
        episode_data = [
            {"id": ep.id, "role": ep.role, "content": ep.content} for ep in user_episodes
        ]

        # 4. Map phase: chunk if needed and summarize each chunk
        chunks: list[list[dict[str, str]]] = []
        for i in range(0, len(episode_data), MAX_EPISODES_PER_CHUNK):
            chunks.append(episode_data[i : i + MAX_EPISODES_PER_CHUNK])

        logger.info(f"Created {len(chunks)} chunk(s) for summarization")

        # Parallel summarization with semaphore to control concurrency
        from engram.config import settings

        async def _summarize_with_logging(
            sem: asyncio.Semaphore,
            chunk: list[dict[str, str]],
            ctx: str,
            idx: int,
            total: int,
        ) -> SummaryOutput:
            """Summarize chunk with semaphore to limit concurrent LLM calls."""
            async with sem:
                logger.debug(f"Summarizing chunk {idx + 1}/{total} ({len(chunk)} episodes)")
                return await _summarize_chunk(chunk, ctx)

        # Process all chunks in parallel
        semaphore = asyncio.Semaphore(settings.max_concurrent_llm_calls)
        tasks = [
            _summarize_with_logging(semaphore, chunk, existing_context, i, len(chunks))
            for i, chunk in enumerate(chunks)
        ]
        chunk_summaries = await asyncio.gather(*tasks)

        # 5. Reduce phase: combine chunk summaries into one
        if len(chunk_summaries) == 1:
            final_summary = chunk_summaries[0].summary
            final_keywords = chunk_summaries[0].keywords
            final_context = chunk_summaries[0].context
            final_confidence = chunk_summaries[0].confidence
            final_confidence_reasoning = chunk_summaries[0].confidence_reasoning
        else:
            reduced = await _reduce_summaries(chunk_summaries)
            final_summary = reduced.summary
            final_keywords = reduced.keywords
            final_context = reduced.context
            final_confidence = reduced.confidence
            final_confidence_reasoning = reduced.confidence_reasoning

        # 6. Create semantic memory with source links
        all_episode_ids = [ep.id for ep in episodes]  # Include system prompts for audit
        user_episode_ids = [ep.id for ep in user_episodes]

        embedding = await embedder.embed(final_summary)

        # Get derivation method from settings
        from engram.config import settings

        memory = SemanticMemory(
            content=final_summary,
            source_episode_ids=all_episode_ids,
            user_id=user_id,
            org_id=org_id,
            embedding=embedding,
            keywords=final_keywords,
            context=final_context,
            derivation_method=f"consolidation:{settings.consolidation_model}",
            # Use LLM-assessed confidence
            confidence=ConfidenceScore.for_inferred(
                confidence=final_confidence,
                supporting_episodes=len(user_episode_ids),
                reasoning=final_confidence_reasoning,
            ),
        )

        # Strengthen through consolidation
        memory.strengthen(delta=0.1)

        await storage.store_semantic(memory)
        logger.info(f"Created semantic summary: {memory.id} from {len(episodes)} episodes")

        # 7. Build links to similar existing memories
        links_created = 0
        if existing_memories:
            similar = await _find_semantically_similar_memories(
                query_embedding=embedding,
                storage=storage,
                user_id=user_id,
                org_id=org_id,
                limit=5,
                min_score=similarity_threshold,
            )

            for similar_mem in similar:
                if similar_mem.id == memory.id:
                    continue
                if similar_mem.id not in memory.related_ids:
                    memory.add_link(similar_mem.id)
                    similar_mem.add_link(memory.id)
                    similar_mem.strengthen(delta=0.1)
                    await storage.update_semantic_memory(similar_mem)
                    links_created += 1
                    logger.debug(f"Linked {memory.id} <--> {similar_mem.id}")

            if links_created > 0:
                await storage.update_semantic_memory(memory)

        # 7b. Optional: LLM-driven link discovery for richer relationship types
        if use_llm_linking and similar:
            from engram.linking import discover_links

            candidates_data = [
                {
                    "id": m.id,
                    "content": m.content,
                    "keywords": m.keywords,
                    "tags": m.tags,
                }
                for m in similar
                if m.id != memory.id
            ]

            if candidates_data:
                try:
                    result = await discover_links(
                        new_memory_content=memory.content,
                        new_memory_id=memory.id,
                        candidate_memories=candidates_data,
                        min_confidence=0.6,
                    )

                    # Apply discovered relationship types to existing links
                    for link in result.links:
                        if link.target_id in memory.link_types:
                            # Upgrade from "related" to more specific type
                            memory.link_types[link.target_id] = link.link_type
                            logger.debug(
                                "Upgraded link %s -> %s to type '%s'",
                                memory.id,
                                link.target_id,
                                link.link_type,
                            )
                        elif link.target_id not in memory.related_ids:
                            # New link discovered by LLM
                            memory.add_link(link.target_id, link.link_type)
                            links_created += 1
                            logger.debug(
                                "LLM discovered %s link: %s -> %s",
                                link.link_type,
                                memory.id,
                                link.target_id,
                            )

                    if result.links:
                        await storage.update_semantic_memory(memory)

                    logger.info(
                        "LLM link discovery: %d links analyzed, %d type upgrades",
                        len(result.links),
                        sum(1 for link in result.links if link.target_id in memory.link_types),
                    )
                except Exception as e:
                    logger.warning("LLM link discovery failed, using embedding links: %s", e)

        # 8. Mark ALL episodes as summarized (including system prompts)
        await storage.mark_episodes_summarized(all_episode_ids, user_id, memory.id)

        total_episodes += len(episodes)
        total_memories += 1
        total_links += links_created

    compression_ratio = total_episodes / total_memories if total_memories > 0 else 0.0

    logger.info(
        f"Consolidation complete: {total_episodes} episodes → {total_memories} memories "
        f"({compression_ratio:.1f}:1 compression)"
    )

    return ConsolidationResult(
        episodes_processed=total_episodes,
        semantic_memories_created=total_memories,
        links_created=total_links,
        compression_ratio=compression_ratio,
    )


async def _synthesize_structured(
    structured_memories: list[dict[str, str | list[str]]],
    existing_context: str,
) -> SummaryOutput:
    """Synthesize StructuredMemories into a coherent summary.

    Unlike episode summarization, this works with already-extracted content,
    so the LLM focuses on synthesis and integration.

    Args:
        structured_memories: List of StructuredMemory data.
        existing_context: Existing semantic memories for context.

    Returns:
        SummaryOutput with synthesized summary.
    """
    from pydantic_ai import Agent

    from engram.config import settings

    formatted_text = format_structured_for_llm(structured_memories) + existing_context

    synthesis_prompt = """You are synthesizing pre-extracted memory summaries into concrete knowledge.

Each entry already has:
- A per-episode summary
- Extracted entities (people, organizations)
- Preferences
- Negations/corrections

Your task:
- Combine these into ONE unified summary (3-5 sentences)
- Resolve any contradictions (prefer more recent information)
- Preserve specific facts: names, tools, versions, configurations, constraints
- Preserve decision rationale (why something was chosen or rejected)
- Preserve negations explicitly (what was ruled out and why)

State facts directly, not as observations about a person.

Examples:
- BAD: "The user is a developer who works extensively with data pipelines."
- GOOD: "The data pipeline uses Airflow for orchestration with PostgreSQL as the metadata store. Redis was removed in v2.3 due to connection pooling issues."

Focus on lasting/stable knowledge. Be conservative.

CONFIDENCE ASSESSMENT:
After synthesizing, assess your confidence in the unified summary (0.0-1.0):
- 0.9-1.0: Strong source agreement across all structured memories
- 0.7-0.9: Good agreement, minor gaps or resolved contradictions
- 0.5-0.7: Reasonable synthesis but some gaps in evidence
- 0.3-0.5: Weak source agreement, speculative conclusions
- 0.0-0.3: Sources conflict or don't support the synthesis

Synthesis should generally score lower than extraction. Be conservative."""

    from engram.workflows.llm_utils import run_agent_with_retry

    agent: Agent[None, SummaryOutput] = Agent(
        settings.consolidation_model,
        output_type=SummaryOutput,
        instructions=synthesis_prompt,
    )

    return await run_agent_with_retry(agent, formatted_text)


async def run_consolidation_from_structured(
    storage: EngramStorage,
    embedder: Embedder,
    user_id: str,
    org_id: str,
    similarity_threshold: float = 0.7,
    use_llm_linking: bool = False,
) -> ConsolidationResult:
    """Run consolidation using StructuredMemories as input.

    This is the preferred consolidation path when StructuredMemories exist.
    It synthesizes pre-extracted per-episode summaries into cross-episode
    SemanticMemory. Scoped to a single org/project.

    Workflow:
    1. Get unconsolidated StructuredMemories within the org
    2. Format for LLM synthesis
    3. Create unified SemanticMemory
    4. Link to similar existing memories within the org
    5. Mark StructuredMemories as consolidated

    Args:
        storage: EngramStorage instance.
        embedder: Embedder for generating vectors.
        user_id: User ID for multi-tenancy.
        org_id: Organization/project ID for isolation.
        similarity_threshold: Min similarity for linking.

    Returns:
        ConsolidationResult with processing statistics.
    """
    from engram.models import SemanticMemory

    # 1. Get unconsolidated StructuredMemories
    structured = await storage.get_unconsolidated_structured(
        user_id=user_id,
        org_id=org_id,
        limit=None,
    )

    if not structured:
        logger.info("No unconsolidated StructuredMemories found")
        return ConsolidationResult(
            episodes_processed=0,
            semantic_memories_created=0,
            links_created=0,
            compression_ratio=0.0,
        )

    logger.info(f"Consolidating {len(structured)} StructuredMemories")

    # 2. Get existing summaries for context
    existing_memories = await storage.list_semantic_memories(user_id, org_id)
    existing_context = _format_existing_memories_for_llm(existing_memories)

    # 3. Format StructuredMemories for LLM
    struct_data: list[dict[str, str | list[str]]] = []
    for s in structured:
        data: dict[str, str | list[str]] = {
            "id": s.id,
            "source_episode_id": s.source_episode_id,
            "summary": s.summary,
            "keywords": s.keywords,
        }
        if s.people:
            data["people"] = [f"{p.name} ({p.role})" if p.role else p.name for p in s.people]
        if s.organizations:
            data["organizations"] = s.organizations
        if s.preferences:
            data["preferences"] = [f"{p.topic}: {p.value}" for p in s.preferences]
        if s.negations:
            data["negations"] = [n.content for n in s.negations]
        struct_data.append(data)

    # 4. Synthesize into semantic memory
    synthesis = await _synthesize_structured(struct_data, existing_context)

    # 5. Create SemanticMemory
    source_episode_ids = [s.source_episode_id for s in structured]
    structured_ids = [s.id for s in structured]

    embedding = await embedder.embed(synthesis.summary)

    # Get derivation method from settings
    from engram.config import settings

    memory = SemanticMemory(
        content=synthesis.summary,
        source_episode_ids=source_episode_ids,
        user_id=user_id,
        org_id=org_id,
        embedding=embedding,
        keywords=synthesis.keywords,
        context=synthesis.context,
        derivation_method=f"consolidation:{settings.consolidation_model}",
        # Use LLM-assessed confidence
        confidence=ConfidenceScore.for_inferred(
            confidence=synthesis.confidence,
            supporting_episodes=len(structured),
            reasoning=synthesis.confidence_reasoning,
        ),
    )

    # Strengthen through consolidation
    memory.strengthen(delta=0.1)

    await storage.store_semantic(memory)
    logger.info(f"Created semantic summary: {memory.id} from {len(structured)} StructuredMemories")

    # 6. Build links to similar existing memories
    links_created = 0
    similar: list[SemanticMemory] = []
    if existing_memories:
        similar = await _find_semantically_similar_memories(
            query_embedding=embedding,
            storage=storage,
            user_id=user_id,
            org_id=org_id,
            limit=5,
            min_score=similarity_threshold,
        )

        for similar_mem in similar:
            if similar_mem.id == memory.id:
                continue
            if similar_mem.id not in memory.related_ids:
                memory.add_link(similar_mem.id)
                similar_mem.add_link(memory.id)
                similar_mem.strengthen(delta=0.1)
                await storage.update_semantic_memory(similar_mem)
                links_created += 1
                logger.debug(f"Linked {memory.id} <--> {similar_mem.id}")

        if links_created > 0:
            await storage.update_semantic_memory(memory)

    # 6b. Optional: LLM-driven link discovery for richer relationship types
    if use_llm_linking and existing_memories:
        from engram.linking import discover_links

        # Use the similar memories found above, or get some if not available
        candidates = similar
        if not candidates:
            candidates = existing_memories[:5]

        candidates_data = [
            {
                "id": m.id,
                "content": m.content,
                "keywords": m.keywords,
                "tags": m.tags,
            }
            for m in candidates
            if m.id != memory.id
        ]

        if candidates_data:
            try:
                result = await discover_links(
                    new_memory_content=memory.content,
                    new_memory_id=memory.id,
                    candidate_memories=candidates_data,
                    min_confidence=0.6,
                )

                # Apply discovered relationship types to existing links
                for link in result.links:
                    if link.target_id in memory.link_types:
                        # Upgrade from "related" to more specific type
                        memory.link_types[link.target_id] = link.link_type
                        logger.debug(
                            "Upgraded link %s -> %s to type '%s'",
                            memory.id,
                            link.target_id,
                            link.link_type,
                        )
                    elif link.target_id not in memory.related_ids:
                        # New link discovered by LLM
                        memory.add_link(link.target_id, link.link_type)
                        links_created += 1
                        logger.debug(
                            "LLM discovered %s link: %s -> %s",
                            link.link_type,
                            memory.id,
                            link.target_id,
                        )

                if result.links:
                    await storage.update_semantic_memory(memory)

                logger.info(
                    "LLM link discovery: %d links analyzed, %d type upgrades",
                    len(result.links),
                    sum(1 for link in result.links if link.target_id in memory.link_types),
                )
            except Exception as e:
                logger.warning("LLM link discovery failed, using embedding links: %s", e)

    # 7. Mark StructuredMemories as consolidated
    await storage.mark_structured_consolidated(structured_ids, user_id, memory.id)

    # Also mark source episodes as summarized
    await storage.mark_episodes_summarized(source_episode_ids, user_id, memory.id)

    compression_ratio = len(structured) if len(structured) > 0 else 0.0

    logger.info(
        f"Consolidation complete: {len(structured)} StructuredMemories → 1 SemanticMemory "
        f"({compression_ratio:.1f}:1 compression)"
    )

    return ConsolidationResult(
        episodes_processed=len(structured),
        semantic_memories_created=1,
        links_created=links_created,
        compression_ratio=compression_ratio,
    )


# Models for DurableAgentFactory consolidation output
class ExtractedFact(BaseModel):
    """A semantic fact extracted by the LLM.

    Used by DurableAgentFactory consolidation agents.
    """

    model_config = ConfigDict(extra="forbid")

    content: str = Field(description="The semantic knowledge extracted")
    confidence: float = Field(
        ge=0.0, le=1.0, default=0.6, description="Confidence in this extraction"
    )
    source_episode_ids: list[str] = Field(
        default_factory=list,
        description="Episode IDs that support this fact",
    )
    source_context: str = Field(default="", description="Original context")
    keywords: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    context: str = Field(default="")


class IdentifiedLink(BaseModel):
    """A relationship between memories identified by the LLM."""

    model_config = ConfigDict(extra="forbid")

    source_content: str = Field(description="Content of source memory")
    target_content: str = Field(description="Content of related memory")
    relationship: str = Field(description="Nature of the relationship")


class MemoryEvolution(BaseModel):
    """Suggested update to an existing memory.

    Used by DurableAgentFactory consolidation agents.
    """

    model_config = ConfigDict(extra="forbid")

    target_content: str = Field(description="Content of the existing memory to update")
    add_tags: list[str] = Field(default_factory=list)
    add_keywords: list[str] = Field(default_factory=list)
    update_context: str = Field(default="")
    reason: str = Field(default="")


class LLMExtractionResult(BaseModel):
    """Structured output from the consolidation LLM agent.

    Used by DurableAgentFactory for durable execution wrappers.
    """

    model_config = ConfigDict(extra="forbid")

    semantic_facts: list[ExtractedFact] = Field(default_factory=list)
    links: list[IdentifiedLink] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)
    evolutions: list[MemoryEvolution] = Field(default_factory=list)


def _find_matching_memory(
    content: str,
    memories: dict[str, SemanticMemory],
) -> SemanticMemory | None:
    """Find a memory matching the given content (fallback string matching)."""
    if content in memories:
        return memories[content]

    normalized = content.strip().lower()
    for mem_content, memory in memories.items():
        if mem_content.strip().lower() == normalized:
            return memory

    for mem_content, memory in memories.items():
        mem_normalized = mem_content.strip().lower()
        if normalized in mem_normalized or mem_normalized in normalized:
            return memory

    return None


__all__ = [
    "ConsolidationResult",
    "ExtractedFact",
    "IdentifiedLink",
    "LLMExtractionResult",
    "MapReduceSummary",
    "MemoryEvolution",
    "SummaryOutput",
    "format_episodes_for_llm",
    "format_structured_for_llm",
    "run_consolidation",
    "run_consolidation_from_structured",
]
