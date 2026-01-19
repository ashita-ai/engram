"""Intelligent confidence scoring for Engram.

This module provides sophisticated confidence calculations beyond
simple static weights.

Components:
- hedging: Detect uncertainty language (Phase 1 of #136)
- specificity: Score information density (Phase 1 of #136)
- signals: Unified signal computation for confidence adjustments
- llm_certainty: LLM-based certainty for Structured/Semantic (Phase 2)
- bayesian: Bayesian updating for Procedural memories (Phase 3)

Architecture:
- StructuredMemory: LLM certainty assessment on extraction
- SemanticMemory: LLM certainty assessment on synthesis
- ProceduralMemory: Bayesian updating with accumulating evidence
"""

from .bayesian import (
    BayesianConfidence,
    bayesian_update,
    combine_bayesian_confidences,
)
from .hedging import HedgingDetector, detect_hedging
from .llm_certainty import (
    CertaintyAssessment,
    LLMCertaintyAssessor,
    SynthesisCertaintyAssessment,
    assess_extraction_certainty,
    assess_synthesis_certainty,
)
from .signals import ConfidenceSignals, compute_confidence_signals
from .specificity import SpecificityScorer, calculate_specificity

__all__ = [
    # Phase 1: Pattern-based
    "HedgingDetector",
    "detect_hedging",
    "SpecificityScorer",
    "calculate_specificity",
    "ConfidenceSignals",
    "compute_confidence_signals",
    # Phase 2: LLM-based (Structured/Semantic)
    "CertaintyAssessment",
    "SynthesisCertaintyAssessment",
    "LLMCertaintyAssessor",
    "assess_extraction_certainty",
    "assess_synthesis_certainty",
    # Phase 3: Bayesian (Procedural)
    "BayesianConfidence",
    "bayesian_update",
    "combine_bayesian_confidences",
]
