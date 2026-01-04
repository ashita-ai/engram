"""Base models and shared types for Engram memory system."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from enum import Enum
from math import log
from typing import TYPE_CHECKING
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from engram.config import ConfidenceWeights


class ExtractionMethod(str, Enum):
    """How a memory was extracted from source data."""

    VERBATIM = "verbatim"  # Exact quote, immutable (confidence: 1.0)
    EXTRACTED = "extracted"  # Deterministic pattern match (confidence: 0.9)
    INFERRED = "inferred"  # LLM-derived, uncertain (confidence: 0.6)


class ConfidenceScore(BaseModel):
    """Composite confidence score with full auditability.

    Confidence is calculated from multiple factors:
    - extraction_method: How reliably was this extracted?
    - supporting_episodes: How many sources corroborate this?
    - last_confirmed: How recently was this verified?
    - contradictions: Is there conflicting evidence?
    """

    model_config = ConfigDict(extra="forbid")

    value: float = Field(ge=0.0, le=1.0, description="Final composite score 0.0-1.0")
    extraction_method: ExtractionMethod = Field(description="How this memory was extracted")
    extraction_base: float = Field(ge=0.0, le=1.0, description="Base score from extraction method")
    supporting_episodes: int = Field(default=1, ge=1, description="Number of corroborating sources")
    last_confirmed: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="When last seen/confirmed"
    )
    contradictions: int = Field(default=0, ge=0, description="Number of conflicting evidence items")
    verified: bool = Field(default=False, description="Whether format validation passed")

    def recompute(
        self,
        extraction_weight: float = 0.50,
        corroboration_weight: float = 0.25,
        recency_weight: float = 0.15,
        verification_weight: float = 0.10,
        decay_half_life_days: int = 365,
        contradiction_penalty: float = 0.10,
    ) -> ConfidenceScore:
        """Recompute confidence value from all factors.

        Uses weighted sum of:
        - extraction_base: Base score from extraction method
        - corroboration: Logarithmic bonus for multiple sources
        - recency: Exponential decay based on time since confirmation
        - verification: Binary bonus if format validation passed

        Contradictions apply a penalty that scales with count.

        Args:
            extraction_weight: Weight for extraction method (default 0.50)
            corroboration_weight: Weight for source count (default 0.25)
            recency_weight: Weight for recency (default 0.15)
            verification_weight: Weight for validation (default 0.10)
            decay_half_life_days: Days for recency to halve (default 365)
            contradiction_penalty: Fraction to reduce per contradiction (default 0.10)

        Returns:
            Self with updated value field.
        """
        # Corroboration: logarithmic scale, 1 source = 0.5, 2 = 0.7, 5 = 0.85, 10 = 1.0
        # Formula: min(1.0, 0.5 + 0.5 * log2(sources) / log2(10))
        if self.supporting_episodes >= 10:
            corroboration_score = 1.0
        elif self.supporting_episodes == 1:
            corroboration_score = 0.5
        else:
            corroboration_score = min(1.0, 0.5 + 0.5 * log(self.supporting_episodes) / log(10))

        # Recency: exponential decay, 1.0 at confirmation, 0.5 at half-life
        days_since = (datetime.now(UTC) - self.last_confirmed).days
        recency_score = 0.5 ** (days_since / decay_half_life_days)

        # Verification: binary 1.0 if verified, 0.0 if not
        verification_score = 1.0 if self.verified else 0.0

        # Weighted sum
        raw_score = (
            self.extraction_base * extraction_weight
            + corroboration_score * corroboration_weight
            + recency_score * recency_weight
            + verification_score * verification_weight
        )

        # Contradiction penalty: reduce by penalty% per contradiction, floor at 0.1
        if self.contradictions > 0:
            penalty_multiplier = (1.0 - contradiction_penalty) ** self.contradictions
            raw_score = max(0.1, raw_score * penalty_multiplier)

        self.value = min(1.0, max(0.0, raw_score))
        return self

    def recompute_with_weights(self, weights: ConfidenceWeights) -> ConfidenceScore:
        """Recompute confidence using a ConfidenceWeights configuration.

        Convenience method that unpacks weights from a ConfidenceWeights object.

        Args:
            weights: ConfidenceWeights configuration object.

        Returns:
            Self with updated value field.
        """
        return self.recompute(
            extraction_weight=weights.extraction,
            corroboration_weight=weights.corroboration,
            recency_weight=weights.recency,
            verification_weight=weights.verification,
            decay_half_life_days=weights.decay_half_life_days,
            contradiction_penalty=weights.contradiction_penalty,
        )

    def explain(self) -> str:
        """Generate human-readable explanation of confidence score.

        Example: "0.85: extracted, 3 sources, verified, confirmed 2 days ago"
        """
        time_ago = self._time_ago(self.last_confirmed)
        parts = [
            f"{self.value:.2f}:",
            self.extraction_method.value,
            f"{self.supporting_episodes} source{'s' if self.supporting_episodes != 1 else ''}",
        ]
        if self.verified:
            parts.append("verified")
        parts.append(f"confirmed {time_ago}")
        if self.contradictions > 0:
            parts.append(
                f"{self.contradictions} contradiction{'s' if self.contradictions != 1 else ''}"
            )
        return ", ".join(parts)

    @staticmethod
    def _time_ago(dt: datetime) -> str:
        """Format datetime as human-readable relative time."""
        now = datetime.now(UTC)
        delta = now - dt

        if delta < timedelta(minutes=1):
            return "just now"
        elif delta < timedelta(hours=1):
            minutes = int(delta.total_seconds() / 60)
            return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
        elif delta < timedelta(days=1):
            hours = int(delta.total_seconds() / 3600)
            return f"{hours} hour{'s' if hours != 1 else ''} ago"
        elif delta < timedelta(days=30):
            days = delta.days
            return f"{days} day{'s' if days != 1 else ''} ago"
        elif delta < timedelta(days=365):
            months = int(delta.days / 30)
            return f"{months} month{'s' if months != 1 else ''} ago"
        else:
            years = int(delta.days / 365)
            return f"{years} year{'s' if years != 1 else ''} ago"

    @classmethod
    def for_verbatim(cls) -> ConfidenceScore:
        """Create confidence score for verbatim (exact quote) extraction."""
        return cls(
            value=1.0,
            extraction_method=ExtractionMethod.VERBATIM,
            extraction_base=1.0,
        )

    @classmethod
    def for_extracted(cls, supporting_episodes: int = 1) -> ConfidenceScore:
        """Create confidence score for pattern-extracted content."""
        return cls(
            value=0.9,
            extraction_method=ExtractionMethod.EXTRACTED,
            extraction_base=0.9,
            supporting_episodes=supporting_episodes,
        )

    @classmethod
    def for_inferred(cls, confidence: float = 0.6, supporting_episodes: int = 1) -> ConfidenceScore:
        """Create confidence score for LLM-inferred content."""
        return cls(
            value=confidence,
            extraction_method=ExtractionMethod.INFERRED,
            extraction_base=0.6,
            supporting_episodes=supporting_episodes,
        )


def generate_id(prefix: str) -> str:
    """Generate a unique ID with the given prefix.

    Examples:
        generate_id("ep") -> "ep_a1b2c3d4e5f6"
        generate_id("fact") -> "fact_a1b2c3d4e5f6"
    """
    return f"{prefix}_{uuid4().hex[:12]}"


class MemoryBase(BaseModel):
    """Base class for all memory types."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="Unique identifier")
    user_id: str = Field(description="User who owns this memory")
    org_id: str | None = Field(default=None, description="Organization (optional)")
    embedding: list[float] | None = Field(
        default=None, description="Vector embedding for semantic search"
    )
