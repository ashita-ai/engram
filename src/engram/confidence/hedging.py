"""Hedging language detection for confidence scoring.

Hedging words and phrases indicate uncertainty in source text. When someone
says "I think I prefer Python" vs "I prefer Python", the confidence in that
preference should differ.

This module provides pattern-based hedging detection as part of Phase 1
of intelligent confidence scoring (#136).

Example:
    >>> from engram.confidence.hedging import detect_hedging
    >>> result = detect_hedging("I think I might prefer Python")
    >>> result.has_hedging
    True
    >>> result.penalty  # 0.8x multiplier
    0.8
"""

from __future__ import annotations

import re
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field


class HedgingResult(BaseModel):
    """Result of hedging detection analysis.

    Attributes:
        has_hedging: Whether hedging language was detected.
        matched_patterns: List of patterns that matched.
        matched_text: Actual text fragments that matched.
        penalty: Confidence multiplier (1.0 = no penalty, <1.0 = reduce confidence).
        severity: Overall severity: none, mild, moderate, strong.
    """

    model_config = ConfigDict(extra="forbid")

    has_hedging: bool = Field(description="Whether hedging language was detected")
    matched_patterns: list[str] = Field(
        default_factory=list,
        description="Pattern names that matched",
    )
    matched_text: list[str] = Field(
        default_factory=list,
        description="Actual text fragments that matched",
    )
    penalty: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence multiplier (1.0 = no penalty)",
    )
    severity: str = Field(
        default="none",
        description="Severity: none, mild, moderate, strong",
    )


class HedgingDetector:
    """Detect hedging language patterns in text.

    Hedging categories (by severity):
    - mild (0.9x): Slight uncertainty ("I believe", "generally")
    - moderate (0.8x): Clear uncertainty ("I think", "probably", "maybe")
    - strong (0.7x): Strong uncertainty ("I'm not sure", "might be wrong")

    Patterns are case-insensitive and use word boundaries to avoid
    false positives (e.g., "something" matching "some").
    """

    # Mild hedging: slight uncertainty (0.9x penalty)
    MILD_PATTERNS: ClassVar[list[tuple[str, str]]] = [
        (r"\bi believe\b", "I believe"),
        (r"\bgenerally\b", "generally"),
        (r"\busually\b", "usually"),
        (r"\btypically\b", "typically"),
        (r"\bin general\b", "in general"),
        (r"\bfor the most part\b", "for the most part"),
        (r"\btends to\b", "tends to"),
        (r"\bseems to\b", "seems to"),
        (r"\bapparently\b", "apparently"),
    ]

    # Moderate hedging: clear uncertainty (0.8x penalty)
    MODERATE_PATTERNS: ClassVar[list[tuple[str, str]]] = [
        (r"\bi think\b", "I think"),
        (r"\bprobably\b", "probably"),
        (r"\bmaybe\b", "maybe"),
        (r"\bperhaps\b", "perhaps"),
        (r"\bmight\b", "might"),
        (r"\bcould be\b", "could be"),
        (r"\bpossibly\b", "possibly"),
        (r"\bseems like\b", "seems like"),
        (r"\bi guess\b", "I guess"),
        (r"\bi suppose\b", "I suppose"),
        (r"\bkind of\b", "kind of"),
        (r"\bsort of\b", "sort of"),
        (r"\bsomewhat\b", "somewhat"),
        (r"\broughly\b", "roughly"),
        (r"\bmore or less\b", "more or less"),
    ]

    # Strong hedging: strong uncertainty (0.7x penalty)
    STRONG_PATTERNS: ClassVar[list[tuple[str, str]]] = [
        (r"\bi'?m not sure\b", "not sure"),
        (r"\bnot certain\b", "not certain"),
        (r"\bi could be wrong\b", "could be wrong"),
        (r"\bmight be wrong\b", "might be wrong"),
        (r"\bif i recall\b", "if I recall"),
        (r"\bif i remember\b", "if I remember"),
        (r"\bdon'?t quote me\b", "don't quote me"),
        (r"\bcorrect me if\b", "correct me if"),
        (r"\bi'?m unsure\b", "unsure"),
        (r"\bhardly certain\b", "hardly certain"),
        (r"\bno idea if\b", "no idea if"),
    ]

    # Penalties for each severity level
    MILD_PENALTY: ClassVar[float] = 0.9
    MODERATE_PENALTY: ClassVar[float] = 0.8
    STRONG_PENALTY: ClassVar[float] = 0.7

    def __init__(self) -> None:
        """Initialize the hedging detector with compiled patterns."""
        self._mild_compiled = [
            (re.compile(pattern, re.IGNORECASE), name) for pattern, name in self.MILD_PATTERNS
        ]
        self._moderate_compiled = [
            (re.compile(pattern, re.IGNORECASE), name) for pattern, name in self.MODERATE_PATTERNS
        ]
        self._strong_compiled = [
            (re.compile(pattern, re.IGNORECASE), name) for pattern, name in self.STRONG_PATTERNS
        ]

    def detect(self, text: str) -> HedgingResult:
        """Detect hedging language in the given text.

        Analyzes text for uncertainty patterns and returns a result with
        matched patterns and calculated penalty.

        Args:
            text: Source text to analyze.

        Returns:
            HedgingResult with detection details and penalty.

        Example:
            >>> detector = HedgingDetector()
            >>> result = detector.detect("I think Python is great")
            >>> result.penalty
            0.8
        """
        if not text or not text.strip():
            return HedgingResult(has_hedging=False)

        matched_patterns: list[str] = []
        matched_text: list[str] = []
        max_severity = "none"
        penalty = 1.0

        # Check strong patterns first (they take precedence)
        for pattern, name in self._strong_compiled:
            match = pattern.search(text)
            if match:
                matched_patterns.append(name)
                matched_text.append(match.group())
                max_severity = "strong"
                penalty = min(penalty, self.STRONG_PENALTY)

        # Check moderate patterns
        for pattern, name in self._moderate_compiled:
            match = pattern.search(text)
            if match:
                matched_patterns.append(name)
                matched_text.append(match.group())
                if max_severity not in ("strong",):
                    max_severity = "moderate"
                    penalty = min(penalty, self.MODERATE_PENALTY)

        # Check mild patterns
        for pattern, name in self._mild_compiled:
            match = pattern.search(text)
            if match:
                matched_patterns.append(name)
                matched_text.append(match.group())
                if max_severity == "none":
                    max_severity = "mild"
                    penalty = min(penalty, self.MILD_PENALTY)

        return HedgingResult(
            has_hedging=len(matched_patterns) > 0,
            matched_patterns=matched_patterns,
            matched_text=matched_text,
            penalty=penalty,
            severity=max_severity,
        )


# Module-level convenience instance
_default_detector: HedgingDetector | None = None


def detect_hedging(text: str) -> HedgingResult:
    """Detect hedging language in text using default detector.

    Convenience function that uses a module-level detector instance.

    Args:
        text: Source text to analyze.

    Returns:
        HedgingResult with detection details.

    Example:
        >>> result = detect_hedging("I might prefer FastAPI")
        >>> result.has_hedging
        True
        >>> result.penalty
        0.8
    """
    global _default_detector
    if _default_detector is None:
        _default_detector = HedgingDetector()
    return _default_detector.detect(text)
