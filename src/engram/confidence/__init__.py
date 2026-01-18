"""Intelligent confidence scoring for Engram.

This module provides text analysis capabilities to improve confidence
calculations beyond simple static weights.

Components:
- hedging: Detect uncertainty language (Phase 1 of #136)
- specificity: Score information density (Phase 1 of #136)
- signals: Unified signal computation for confidence adjustments
"""

from .hedging import HedgingDetector, detect_hedging
from .signals import ConfidenceSignals, compute_confidence_signals
from .specificity import SpecificityScorer, calculate_specificity

__all__ = [
    "HedgingDetector",
    "detect_hedging",
    "SpecificityScorer",
    "calculate_specificity",
    "ConfidenceSignals",
    "compute_confidence_signals",
]
