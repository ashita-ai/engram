"""Intelligent confidence scoring for Engram.

Architecture:
- Episodic: 1.0 always (verbatim, immutable)
- Structured: LLM assesses confidence during extraction
- Semantic: LLM assesses confidence during synthesis
- Procedural: Bayesian updating with accumulating evidence

Modules:
- llm_certainty: LLM-based certainty assessment (integrated into extraction)
- bayesian: Beta-Bernoulli model for procedural memory confidence
"""

from .bayesian import (
    BayesianConfidence,
    bayesian_update,
    combine_bayesian_confidences,
)
from .llm_certainty import (
    CertaintyAssessment,
    LLMCertaintyAssessor,
    SynthesisCertaintyAssessment,
    assess_extraction_certainty,
    assess_synthesis_certainty,
)

__all__ = [
    # LLM-based (Structured/Semantic)
    "CertaintyAssessment",
    "SynthesisCertaintyAssessment",
    "LLMCertaintyAssessor",
    "assess_extraction_certainty",
    "assess_synthesis_certainty",
    # Bayesian (Procedural)
    "BayesianConfidence",
    "bayesian_update",
    "combine_bayesian_confidences",
]
