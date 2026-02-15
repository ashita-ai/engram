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
    """Structured output from the procedural synthesis LLM agent.

    Produces actionable operational knowledge instead of personality profiles.
    Each field contains imperative rules, not descriptive prose.
    """

    model_config = ConfigDict(extra="forbid")

    operational_rules: list[str] = Field(
        default_factory=list,
        description=(
            "Actionable rules in imperative form. "
            "Examples: 'Always use asyncio.Lock with shared OrderedDict caches', "
            "'Run ruff check before committing', 'Use Pydantic for all data models'."
        ),
    )
    technical_constraints: list[str] = Field(
        default_factory=list,
        description=(
            "Technical boundaries and requirements. "
            "Examples: 'PostgreSQL 16 is the primary database', "
            "'mypy strict mode is enforced', 'Never use dataclasses'."
        ),
    )
    rejected_approaches: list[str] = Field(
        default_factory=list,
        description=(
            "Approaches that were tried and rejected, with rationale. "
            "Examples: 'Redis was removed due to connection pooling issues', "
            "'dataclasses rejected in favor of Pydantic for validation'."
        ),
    )
    keywords: list[str] = Field(
        default_factory=list,
        description="Key terms for retrieval",
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


async def _synthesize_operational_knowledge(
    semantics: list[SemanticMemory],
) -> SynthesisOutput:
    """Synthesize operational knowledge from semantic memories using LLM.

    Extracts actionable rules, technical constraints, and rejected approaches
    instead of personality profiles. Output is in imperative form.

    Args:
        semantics: Semantic memories to synthesize.

    Returns:
        SynthesisOutput with operational rules, constraints, and rejections.
    """
    from pydantic_ai import Agent

    from engram.config import settings

    formatted_text = _format_semantics_for_llm(semantics)

    synthesis_prompt = """You are extracting operational knowledge from semantic memory summaries.

Extract three categories of actionable knowledge:

1. OPERATIONAL RULES: Imperative statements about how things should be done.
   Format: "Always X", "Never Y because Z", "When X happens, do Y"
   Examples:
   - "Always use asyncio.Lock with shared OrderedDict caches"
   - "Run ruff check and mypy before committing"
   - "Use Pydantic for all data models, never dataclasses"

2. TECHNICAL CONSTRAINTS: Fixed facts about the system or environment.
   Format: Direct factual statements
   Examples:
   - "PostgreSQL 16 is the primary database"
   - "mypy strict mode is enforced"
   - "Minimum Python version is 3.11"

3. REJECTED APPROACHES: Things that were tried and abandoned, with reasons.
   Format: "X was rejected because Y"
   Examples:
   - "Redis was removed due to connection pooling issues"
   - "Retrieval-induced forgetting was removed due to context mismatch"

CRITICAL RULES:
- Write in imperative form, NOT third person
- NO personality descriptions ("The user is methodical...")
- NO behavioral observations ("They tend to...")
- NO communication style analysis
- ONLY include knowledge that appears in the source memories
- Each rule should be specific enough to act on"""

    from engram.workflows.llm_utils import run_agent_with_retry

    agent: Agent[None, SynthesisOutput] = Agent(
        settings.consolidation_model,
        output_type=SynthesisOutput,
        instructions=synthesis_prompt,
    )

    return await run_agent_with_retry(agent, formatted_text)


async def run_synthesis(
    storage: EngramStorage,
    embedder: Embedder,
    user_id: str,
    org_id: str,
) -> SynthesisResult:
    """Run procedural synthesis workflow.

    Synthesizes ALL semantic memories into ONE procedural memory per
    (user, org) pair. If a procedural memory already exists, it is replaced.
    Scoped to a single org/project to prevent cross-project bleed.

    This workflow:
    1. Fetches ALL semantic memories for the user within the org
    2. Uses LLM to synthesize behavioral patterns
    3. Creates/replaces ONE procedural memory for this org
    4. Links procedural to all source semantic IDs

    Args:
        storage: EngramStorage instance.
        embedder: Embedder for generating vectors.
        user_id: User ID for multi-tenancy.
        org_id: Organization/project ID for isolation.

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

    # 3. Synthesize operational knowledge via LLM
    synthesis_output = await _synthesize_operational_knowledge(semantics)

    # 4. Build the procedural content from structured sections
    content_parts: list[str] = []

    if synthesis_output.operational_rules:
        content_parts.append("## Operational Rules")
        for rule in synthesis_output.operational_rules:
            content_parts.append(f"- {rule}")
        content_parts.append("")

    if synthesis_output.technical_constraints:
        content_parts.append("## Technical Constraints")
        for constraint in synthesis_output.technical_constraints:
            content_parts.append(f"- {constraint}")
        content_parts.append("")

    if synthesis_output.rejected_approaches:
        content_parts.append("## Rejected Approaches")
        for rejection in synthesis_output.rejected_approaches:
            content_parts.append(f"- {rejection}")
        content_parts.append("")

    full_content = (
        "\n".join(content_parts) if content_parts else "No operational knowledge extracted."
    )

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

    # Get derivation method from settings
    from engram.config import settings

    if existing_procedural:
        # Update existing procedural memory
        existing_procedural.content = full_content
        existing_procedural.source_episode_ids = unique_episode_ids
        existing_procedural.source_semantic_ids = source_semantic_ids
        existing_procedural.embedding = embedding
        existing_procedural.derivation_method = f"synthesis:{settings.consolidation_model}"
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
            derivation_method=f"synthesis:{settings.consolidation_model}",
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
