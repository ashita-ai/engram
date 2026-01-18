"""Unified confidence signals for intelligent scoring.

This module combines hedging detection and specificity scoring into
a unified signal that can be applied to confidence calculations.

The ConfidenceSignals output provides:
- hedging_penalty: Multiplier from hedging detection (0.7-1.0)
- specificity_boost: Additive boost from specificity (0.0-0.15)
- combined_adjustment: Net effect on confidence

Example:
    >>> from engram.confidence.signals import compute_confidence_signals
    >>> signals = compute_confidence_signals("I think I use Python 3.12")
    >>> signals.combined_adjustment  # Hedging penalty + specificity boost
    -0.05
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from .hedging import HedgingDetector, HedgingResult
from .specificity import SpecificityResult, SpecificityScorer


class ConfidenceSignals(BaseModel):
    """Combined confidence adjustment signals.

    Attributes:
        hedging_result: Full hedging detection result.
        specificity_result: Full specificity scoring result.
        hedging_penalty: Multiplier from hedging (1.0 = no change, <1.0 = reduce).
        specificity_boost: Additive boost from specificity (0.0-0.15).
        combined_adjustment: Net multiplier for confidence.
        explanation: Human-readable explanation of adjustments.
    """

    model_config = ConfigDict(extra="forbid")

    hedging_result: HedgingResult = Field(description="Hedging detection details")
    specificity_result: SpecificityResult = Field(description="Specificity scoring details")
    hedging_penalty: float = Field(
        ge=0.0,
        le=1.0,
        description="Hedging multiplier (1.0 = no penalty)",
    )
    specificity_boost: float = Field(
        ge=-0.1,
        le=0.15,
        description="Specificity additive adjustment",
    )
    combined_adjustment: float = Field(
        description="Net multiplier for confidence score",
    )
    explanation: str = Field(
        default="",
        description="Human-readable explanation",
    )

    def apply_to_confidence(self, base_confidence: float) -> float:
        """Apply signals to a base confidence score.

        The adjustment applies hedging as a multiplier first,
        then adds specificity boost, bounded to [0.0, 1.0].

        Args:
            base_confidence: Original confidence score (0.0-1.0).

        Returns:
            Adjusted confidence score (0.0-1.0).

        Example:
            >>> signals = ConfidenceSignals(
            ...     hedging_penalty=0.8,
            ...     specificity_boost=0.05,
            ...     combined_adjustment=0.85,
            ...     hedging_result=HedgingResult(has_hedging=True, penalty=0.8),
            ...     specificity_result=SpecificityResult(score=0.7, word_count=10),
            ... )
            >>> signals.apply_to_confidence(0.9)
            0.77
        """
        # Apply hedging as multiplier, then add specificity boost
        adjusted = (base_confidence * self.hedging_penalty) + self.specificity_boost
        return min(1.0, max(0.0, adjusted))


class SignalComputer:
    """Compute unified confidence signals from text.

    Combines hedging detection and specificity scoring into
    a single analysis pass.
    """

    def __init__(
        self,
        hedging_detector: HedgingDetector | None = None,
        specificity_scorer: SpecificityScorer | None = None,
        *,
        max_specificity_boost: float = 0.15,
        min_specificity_penalty: float = -0.1,
    ) -> None:
        """Initialize the signal computer.

        Args:
            hedging_detector: Optional custom hedging detector.
            specificity_scorer: Optional custom specificity scorer.
            max_specificity_boost: Maximum specificity bonus (default 0.15).
            min_specificity_penalty: Minimum specificity penalty (default -0.1).
        """
        self._hedging = hedging_detector or HedgingDetector()
        self._specificity = specificity_scorer or SpecificityScorer()
        self._max_boost = max_specificity_boost
        self._min_penalty = min_specificity_penalty

    def compute(self, text: str) -> ConfidenceSignals:
        """Compute confidence signals for the given text.

        Runs hedging detection and specificity scoring, then
        combines results into unified signals.

        Args:
            text: Source text to analyze.

        Returns:
            ConfidenceSignals with combined adjustments.
        """
        # Run both analyses
        hedging_result = self._hedging.detect(text)
        specificity_result = self._specificity.score(text)

        # Calculate specificity boost/penalty
        # Map specificity score (0.0-1.0) to boost range
        # score < 0.5 → penalty (down to min_penalty)
        # score > 0.5 → boost (up to max_boost)
        specificity_score = specificity_result.score
        if specificity_score >= 0.5:
            # Map 0.5-1.0 to 0.0-max_boost
            specificity_boost = (specificity_score - 0.5) * 2 * self._max_boost
        else:
            # Map 0.0-0.5 to min_penalty-0.0
            specificity_boost = (specificity_score - 0.5) * 2 * abs(self._min_penalty)

        # Combined adjustment: hedging multiplier * (1 + specificity adjustment as fraction)
        # This keeps hedging as dominant signal, with specificity as modifier
        hedging_penalty = hedging_result.penalty
        combined = hedging_penalty * (1.0 + specificity_boost / 2)
        combined = min(1.0, max(0.5, combined))  # Floor at 0.5 to avoid over-penalizing

        # Build explanation
        explanation_parts = []
        if hedging_result.has_hedging:
            explanation_parts.append(
                f"hedging detected ({hedging_result.severity}): {hedging_penalty:.0%} penalty"
            )
        if specificity_boost > 0.01:
            explanation_parts.append(f"high specificity: +{specificity_boost:.0%}")
        elif specificity_boost < -0.01:
            explanation_parts.append(f"low specificity: {specificity_boost:.0%}")

        explanation = "; ".join(explanation_parts) if explanation_parts else "no adjustments"

        return ConfidenceSignals(
            hedging_result=hedging_result,
            specificity_result=specificity_result,
            hedging_penalty=hedging_penalty,
            specificity_boost=specificity_boost,
            combined_adjustment=combined,
            explanation=explanation,
        )


# Module-level convenience instance
_default_computer: SignalComputer | None = None


def compute_confidence_signals(text: str) -> ConfidenceSignals:
    """Compute confidence signals using default computer.

    Convenience function that uses a module-level computer instance.

    Args:
        text: Source text to analyze.

    Returns:
        ConfidenceSignals with combined adjustments.

    Example:
        >>> signals = compute_confidence_signals("I definitely prefer Python 3.12")
        >>> signals.hedging_penalty
        1.0
        >>> signals.specificity_boost > 0
        True
    """
    global _default_computer
    if _default_computer is None:
        _default_computer = SignalComputer()
    return _default_computer.compute(text)
