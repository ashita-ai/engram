"""LLM-based certainty assessment for memory confidence.

Uses Pydantic AI to assess how confident we should be in extracted
or synthesized memories. This provides context-aware certainty scoring
that understands nuance beyond simple pattern matching.

Used for:
- StructuredMemory: "How confident is this extraction from the episode?"
- SemanticMemory: "How confident is this synthesis from multiple sources?"

Example:
    >>> from engram.confidence.llm_certainty import assess_extraction_certainty
    >>> result = await assess_extraction_certainty(
    ...     extracted="User prefers Python",
    ...     source_text="I really love Python for web development",
    ... )
    >>> result.certainty  # 0.0-1.0
    0.85
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class CertaintyAssessment(BaseModel):
    """LLM output for certainty assessment.

    Attributes:
        certainty: Confidence score 0.0-1.0.
        reasoning: Explanation of the assessment.
        hedging_detected: List of hedging phrases found.
        clarity: How clearly the information was stated (clear/implied/ambiguous).
        temporal_qualifier: Any time-related qualifiers (current/past/future/none).
    """

    model_config = ConfigDict(extra="forbid")

    certainty: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score from 0.0 (uncertain) to 1.0 (certain)",
    )
    reasoning: str = Field(
        description="Brief explanation of the certainty assessment",
    )
    hedging_detected: list[str] = Field(
        default_factory=list,
        description="Hedging phrases detected in source",
    )
    clarity: str = Field(
        default="clear",
        description="How clearly stated: clear, implied, or ambiguous",
    )
    temporal_qualifier: str = Field(
        default="none",
        description="Time qualifier: current, past, future, or none",
    )


class SynthesisCertaintyAssessment(BaseModel):
    """LLM output for synthesis certainty assessment.

    Used when assessing confidence in SemanticMemory derived from
    multiple StructuredMemory sources.

    Attributes:
        certainty: Confidence score 0.0-1.0.
        reasoning: Explanation of the assessment.
        source_agreement: How well sources agree (strong/moderate/weak/conflicting).
        inference_strength: How strong the inference is (direct/reasonable/speculative).
        gaps: Information gaps that reduce certainty.
    """

    model_config = ConfigDict(extra="forbid")

    certainty: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score from 0.0 (uncertain) to 1.0 (certain)",
    )
    reasoning: str = Field(
        description="Brief explanation of the certainty assessment",
    )
    source_agreement: str = Field(
        default="moderate",
        description="How well sources agree: strong, moderate, weak, conflicting",
    )
    inference_strength: str = Field(
        default="reasonable",
        description="Inference type: direct, reasonable, speculative",
    )
    gaps: list[str] = Field(
        default_factory=list,
        description="Information gaps that reduce certainty",
    )


# System prompts for certainty assessment
EXTRACTION_CERTAINTY_PROMPT = """You are assessing certainty in information extraction.

Given an extracted statement and its source text, evaluate:
1. How clearly was this information stated? (not implied or assumed)
2. Was there hedging language ("I think", "maybe", "probably")?
3. Is this a current fact, past fact, or future intention?
4. How confident should we be in this extraction?

Scoring guide:
- 0.9-1.0: Explicitly stated, no hedging, clear and unambiguous
- 0.7-0.9: Clearly implied or stated with minor hedging
- 0.5-0.7: Reasonably inferred but not directly stated
- 0.3-0.5: Speculative, significant hedging, or ambiguous
- 0.0-0.3: Contradicted, heavily hedged, or likely wrong

Be conservative. When in doubt, lower the certainty score."""

SYNTHESIS_CERTAINTY_PROMPT = """You are assessing certainty in memory synthesis.

Given a synthesized statement and its source memories, evaluate:
1. Do the sources agree on this conclusion?
2. Is this a direct observation or an inference?
3. Are there gaps in the evidence?
4. How confident should we be in this synthesis?

Scoring guide:
- 0.9-1.0: Multiple sources directly state this, strong agreement
- 0.7-0.9: Sources support this with good agreement
- 0.5-0.7: Reasonable inference from sources, some gaps
- 0.3-0.5: Speculative, sources don't clearly support, or weak agreement
- 0.0-0.3: Sources conflict or don't support this conclusion

Be conservative. Synthesis should generally score lower than extraction."""


async def assess_extraction_certainty(
    extracted: str,
    source_text: str,
    context: str = "",
) -> CertaintyAssessment:
    """Assess certainty of an extracted statement from source text.

    Uses LLM to evaluate how confident we should be that the extracted
    information accurately represents what was stated in the source.

    Args:
        extracted: The extracted statement (e.g., "User prefers Python").
        source_text: Original text the extraction came from.
        context: Optional additional context.

    Returns:
        CertaintyAssessment with confidence score and reasoning.

    Example:
        >>> result = await assess_extraction_certainty(
        ...     extracted="User prefers Python",
        ...     source_text="I've been using Python for years and love it",
        ... )
        >>> result.certainty >= 0.8
        True
    """
    from pydantic_ai import Agent

    from engram.config import settings

    agent: Agent[None, CertaintyAssessment] = Agent(
        settings.llm_model,
        output_type=CertaintyAssessment,
        instructions=EXTRACTION_CERTAINTY_PROMPT,
    )

    user_prompt = f"""Assess the certainty of this extraction:

EXTRACTED STATEMENT:
{extracted}

SOURCE TEXT:
{source_text}
"""
    if context:
        user_prompt += f"\nADDITIONAL CONTEXT:\n{context}"

    user_prompt += "\n\nProvide your certainty assessment."

    try:
        from engram.workflows.llm_utils import run_agent_with_retry

        result_output = await run_agent_with_retry(agent, user_prompt)
        logger.debug(
            "Extraction certainty for '%s': %.2f (%s)",
            extracted[:50],
            result_output.certainty,
            result_output.clarity,
        )
        return result_output

    except Exception as e:
        logger.warning("LLM certainty assessment failed: %s", e)
        # Fall back to moderate certainty
        return CertaintyAssessment(
            certainty=0.6,
            reasoning=f"Fallback due to LLM error: {e}",
            clarity="unknown",
        )


async def assess_synthesis_certainty(
    synthesized: str,
    source_memories: list[str],
    context: str = "",
) -> SynthesisCertaintyAssessment:
    """Assess certainty of a synthesized statement from multiple sources.

    Uses LLM to evaluate how confident we should be that the synthesized
    memory accurately represents the combined evidence from sources.

    Args:
        synthesized: The synthesized statement.
        source_memories: List of source memory contents.
        context: Optional additional context.

    Returns:
        SynthesisCertaintyAssessment with confidence score and reasoning.

    Example:
        >>> result = await assess_synthesis_certainty(
        ...     synthesized="User is a Python backend developer",
        ...     source_memories=[
        ...         "User prefers Python",
        ...         "User works on API development",
        ...         "User uses FastAPI framework",
        ...     ],
        ... )
        >>> result.certainty >= 0.7
        True
    """
    from pydantic_ai import Agent

    from engram.config import settings

    agent: Agent[None, SynthesisCertaintyAssessment] = Agent(
        settings.llm_model,
        output_type=SynthesisCertaintyAssessment,
        instructions=SYNTHESIS_CERTAINTY_PROMPT,
    )

    sources_text = "\n".join(f"- {mem}" for mem in source_memories)

    user_prompt = f"""Assess the certainty of this synthesis:

SYNTHESIZED STATEMENT:
{synthesized}

SOURCE MEMORIES:
{sources_text}
"""
    if context:
        user_prompt += f"\nADDITIONAL CONTEXT:\n{context}"

    user_prompt += "\n\nProvide your certainty assessment."

    try:
        from engram.workflows.llm_utils import run_agent_with_retry

        result_output = await run_agent_with_retry(agent, user_prompt)
        logger.debug(
            "Synthesis certainty for '%s': %.2f (%s)",
            synthesized[:50],
            result_output.certainty,
            result_output.source_agreement,
        )
        return result_output

    except Exception as e:
        logger.warning("LLM synthesis certainty assessment failed: %s", e)
        # Fall back to lower certainty for synthesis
        return SynthesisCertaintyAssessment(
            certainty=0.5,
            reasoning=f"Fallback due to LLM error: {e}",
            source_agreement="unknown",
            inference_strength="unknown",
        )


class LLMCertaintyAssessor:
    """Stateful assessor for LLM-based certainty scoring.

    Provides a reusable interface for certainty assessment with
    optional caching and configuration.

    Attributes:
        extraction_calls: Count of extraction assessments made.
        synthesis_calls: Count of synthesis assessments made.
    """

    def __init__(self, enabled: bool = True) -> None:
        """Initialize the assessor.

        Args:
            enabled: If False, returns fallback scores without LLM calls.
        """
        self.enabled = enabled
        self.extraction_calls = 0
        self.synthesis_calls = 0

    async def assess_extraction(
        self,
        extracted: str,
        source_text: str,
        context: str = "",
    ) -> CertaintyAssessment:
        """Assess extraction certainty.

        See assess_extraction_certainty for details.
        """
        self.extraction_calls += 1

        if not self.enabled:
            return CertaintyAssessment(
                certainty=0.7,
                reasoning="LLM assessment disabled",
                clarity="unknown",
            )

        return await assess_extraction_certainty(extracted, source_text, context)

    async def assess_synthesis(
        self,
        synthesized: str,
        source_memories: list[str],
        context: str = "",
    ) -> SynthesisCertaintyAssessment:
        """Assess synthesis certainty.

        See assess_synthesis_certainty for details.
        """
        self.synthesis_calls += 1

        if not self.enabled:
            return SynthesisCertaintyAssessment(
                certainty=0.5,
                reasoning="LLM assessment disabled",
                source_agreement="unknown",
                inference_strength="unknown",
            )

        return await assess_synthesis_certainty(synthesized, source_memories, context)
