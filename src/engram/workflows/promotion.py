"""Procedural synthesis workflow for creating behavioral profiles.

This workflow synthesizes ALL semantic memories into ONE procedural memory
per user. The procedural memory captures behavioral patterns and preferences
that can guide future interactions.

Based on the hierarchical consolidation model:
    Episodic → Semantic (summaries) → Procedural (behavioral synthesis)

Design decisions (from PLAN-consolidation-redesign.md):
- ONE procedural memory per user (replaces existing, doesn't accumulate)
- Synthesized from ALL semantic memories
- Called explicitly, not automatically
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from engram.embeddings import Embedder
    from engram.models import ProceduralMemory, SemanticMemory
    from engram.storage import EngramStorage

logger = logging.getLogger(__name__)


class SynthesisOutput(BaseModel):
    """Structured output from the procedural synthesis LLM agent."""

    model_config = ConfigDict(extra="forbid")

    behavioral_profile: str = Field(
        description="Coherent description of behavioral patterns (3-6 sentences)"
    )
    communication_style: str = Field(
        default="",
        description="How the user prefers to communicate",
    )
    technical_preferences: list[str] = Field(
        default_factory=list,
        description="Technical tools, languages, frameworks preferred",
    )
    work_patterns: list[str] = Field(
        default_factory=list,
        description="How the user approaches problems",
    )
    keywords: list[str] = Field(
        default_factory=list,
        description="Key terms for retrieval (5-10 words)",
    )


class SynthesisResult(BaseModel):
    """Result of a procedural synthesis run.

    Attributes:
        semantics_analyzed: Number of semantic memories analyzed.
        procedural_created: Whether a new procedural memory was created.
        procedural_updated: Whether an existing procedural was replaced.
        procedural_id: ID of the created/updated procedural memory.
    """

    model_config = ConfigDict(extra="forbid")

    semantics_analyzed: int = Field(ge=0)
    procedural_created: bool = Field(default=False)
    procedural_updated: bool = Field(default=False)
    procedural_id: str | None = Field(default=None)


def _format_semantics_for_llm(semantics: list[SemanticMemory]) -> str:
    """Format semantic memories for LLM synthesis.

    Args:
        semantics: List of semantic memories to synthesize.

    Returns:
        Formatted text for LLM input.
    """
    lines = ["# Semantic Memories to Synthesize\n"]
    for i, sem in enumerate(semantics, 1):
        lines.append(f"## Memory {i}")
        lines.append(sem.content)
        if sem.keywords:
            lines.append(f"Keywords: {', '.join(sem.keywords)}")
        if sem.context:
            lines.append(f"Context: {sem.context}")
        lines.append("")
    return "\n".join(lines)


async def _synthesize_behavioral_profile(
    semantics: list[SemanticMemory],
) -> SynthesisOutput:
    """Synthesize behavioral profile from semantic memories using LLM.

    Args:
        semantics: Semantic memories to synthesize.

    Returns:
        SynthesisOutput with behavioral profile.
    """
    from pydantic_ai import Agent

    from engram.config import settings

    formatted_text = _format_semantics_for_llm(semantics)

    synthesis_prompt = """You are analyzing semantic memory summaries to create a behavioral profile.

From these summaries, identify and describe:
- Communication preferences (tone, detail level, format they prefer)
- Technical preferences (languages, tools, frameworks they use)
- Work patterns (how they approach problems, what they prioritize)
- Personal context (role, goals, constraints)

Create a coherent behavioral profile that can guide future interactions.
Write in third person ("The user prefers...", "They tend to...").

Format as a description, not a bulleted list.
Only include patterns that appear across multiple summaries or are explicitly stated."""

    agent: Agent[None, SynthesisOutput] = Agent(
        settings.consolidation_model,
        output_type=SynthesisOutput,
        instructions=synthesis_prompt,
    )

    result = await agent.run(formatted_text)
    return result.output


async def run_synthesis(
    storage: EngramStorage,
    embedder: Embedder,
    user_id: str,
    org_id: str | None = None,
) -> SynthesisResult:
    """Run procedural synthesis workflow.

    Synthesizes ALL semantic memories into ONE procedural memory.
    If a procedural memory already exists for the user, it is replaced.

    This workflow:
    1. Fetches ALL semantic memories for the user
    2. Uses LLM to synthesize behavioral patterns
    3. Creates/replaces ONE procedural memory
    4. Links procedural to all source semantic IDs

    Args:
        storage: EngramStorage instance.
        embedder: Embedder for generating vectors.
        user_id: User ID for multi-tenancy.
        org_id: Optional organization ID.

    Returns:
        SynthesisResult with processing statistics.
    """
    from engram.models import ProceduralMemory

    # 1. Fetch ALL semantic memories
    semantics = await storage.list_semantic_memories(
        user_id=user_id,
        org_id=org_id,
        include_archived=False,
    )

    if not semantics:
        logger.info("No semantic memories found for synthesis")
        return SynthesisResult(
            semantics_analyzed=0,
            procedural_created=False,
            procedural_updated=False,
        )

    logger.info(f"Synthesizing behavioral profile from {len(semantics)} semantic memories")

    # 2. Check for existing procedural memory
    existing_procedurals = await storage.list_procedural_memories(user_id, org_id)
    existing_procedural: ProceduralMemory | None = None
    if existing_procedurals:
        existing_procedural = existing_procedurals[0]  # There should be at most one
        logger.info(f"Found existing procedural memory: {existing_procedural.id}")

    # 3. Synthesize behavioral profile via LLM
    synthesis_output = await _synthesize_behavioral_profile(semantics)

    # 4. Build the procedural content
    content_parts = [synthesis_output.behavioral_profile]

    if synthesis_output.communication_style:
        content_parts.append(f"\nCommunication style: {synthesis_output.communication_style}")

    if synthesis_output.technical_preferences:
        content_parts.append(
            f"\nTechnical preferences: {', '.join(synthesis_output.technical_preferences)}"
        )

    if synthesis_output.work_patterns:
        content_parts.append(f"\nWork patterns: {', '.join(synthesis_output.work_patterns)}")

    full_content = "\n".join(content_parts)

    # 5. Collect all source episode IDs from semantics (deduplicated, order preserved)
    all_source_episode_ids: list[str] = []
    for sem in semantics:
        all_source_episode_ids.extend(sem.source_episode_ids)
    seen: set[str] = set()
    unique_episode_ids: list[str] = []
    for eid in all_source_episode_ids:
        if eid not in seen:
            seen.add(eid)
            unique_episode_ids.append(eid)

    # 6. Create embedding for the behavioral profile
    embedding = await embedder.embed(full_content)

    # 7. Create or update procedural memory
    source_semantic_ids = [sem.id for sem in semantics]

    if existing_procedural:
        # Update existing procedural memory
        existing_procedural.content = full_content
        existing_procedural.source_episode_ids = unique_episode_ids
        existing_procedural.source_semantic_ids = source_semantic_ids
        existing_procedural.embedding = embedding
        existing_procedural.reinforce()  # Reinforce through synthesis

        await storage.update_procedural_memory(existing_procedural)
        procedural_id = existing_procedural.id
        logger.info(f"Updated procedural memory: {procedural_id}")

        return SynthesisResult(
            semantics_analyzed=len(semantics),
            procedural_created=False,
            procedural_updated=True,
            procedural_id=procedural_id,
        )
    else:
        # Create new procedural memory
        procedural = ProceduralMemory(
            content=full_content,
            trigger_context="general interaction",
            source_episode_ids=unique_episode_ids,
            source_semantic_ids=source_semantic_ids,
            user_id=user_id,
            org_id=org_id,
            embedding=embedding,
        )

        await storage.store_procedural(procedural)
        logger.info(f"Created procedural memory: {procedural.id}")

        return SynthesisResult(
            semantics_analyzed=len(semantics),
            procedural_created=True,
            procedural_updated=False,
            procedural_id=procedural.id,
        )


__all__ = [
    "SynthesisOutput",
    "SynthesisResult",
    "run_synthesis",
]
