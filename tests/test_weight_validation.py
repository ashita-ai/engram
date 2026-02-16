"""Tests for weight validation enforcement (SPEC-007).

Verifies that:
1. RerankWeights emits UserWarning when weights don't sum to ~1.0
2. ConfidenceWeights emits UserWarning when weights don't sum to ~1.0
3. Default weights produce no warnings
4. Decay thresholds reject invalid ordering (delete >= archive)
5. Decay thresholds accept valid ordering (delete < archive)
"""

from __future__ import annotations

import warnings

import pytest

from engram.config import ConfidenceWeights, RerankWeights, Settings


class TestRerankWeightsValidation:
    """Tests for RerankWeights sum validation."""

    def test_warns_on_weights_sum_too_high(self) -> None:
        """Weights summing above 1.0 should emit a UserWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            RerankWeights(
                similarity=0.80,
                recency=0.80,
                confidence=0.15,
                session=0.10,
                access=0.05,
            )
        user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
        assert len(user_warnings) == 1
        assert "RerankWeights sum to 1.900" in str(user_warnings[0].message)
        assert "expected ~1.0" in str(user_warnings[0].message)

    def test_warns_on_weights_sum_too_low(self) -> None:
        """Weights summing below 1.0 should emit a UserWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            RerankWeights(
                similarity=0.10,
                recency=0.10,
                confidence=0.05,
                session=0.05,
                access=0.01,
            )
        user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
        assert len(user_warnings) == 1
        assert "RerankWeights sum to 0.310" in str(user_warnings[0].message)

    def test_no_warning_on_default_weights(self) -> None:
        """Default RerankWeights should not emit any warnings."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            RerankWeights()
        user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
        assert len(user_warnings) == 0

    def test_no_warning_on_exact_sum(self) -> None:
        """Weights summing to exactly 1.0 should not emit warnings."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            RerankWeights(
                similarity=0.40,
                recency=0.25,
                confidence=0.20,
                session=0.10,
                access=0.05,
            )
        user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
        assert len(user_warnings) == 0

    def test_no_warning_within_tolerance(self) -> None:
        """Weights within 0.01 of 1.0 should not emit warnings."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            RerankWeights(
                similarity=0.504,
                recency=0.20,
                confidence=0.15,
                session=0.10,
                access=0.05,
            )
        user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
        assert len(user_warnings) == 0


class TestConfidenceWeightsValidation:
    """Tests for ConfidenceWeights sum validation."""

    def test_warns_on_weights_sum_too_high(self) -> None:
        """Weights summing above 1.0 should emit a UserWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ConfidenceWeights(
                extraction=0.70,
                corroboration=0.40,
                recency=0.15,
                verification=0.10,
            )
        user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
        assert len(user_warnings) == 1
        assert "ConfidenceWeights sum to 1.350" in str(user_warnings[0].message)
        assert "miscalibrated" in str(user_warnings[0].message)

    def test_warns_on_weights_sum_too_low(self) -> None:
        """Weights summing below 1.0 should emit a UserWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ConfidenceWeights(
                extraction=0.10,
                corroboration=0.10,
                recency=0.05,
                verification=0.05,
            )
        user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
        assert len(user_warnings) == 1
        assert "ConfidenceWeights sum to 0.300" in str(user_warnings[0].message)

    def test_no_warning_on_default_weights(self) -> None:
        """Default ConfidenceWeights should not emit any warnings."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ConfidenceWeights()
        user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
        assert len(user_warnings) == 0

    def test_no_warning_on_exact_sum(self) -> None:
        """Weights summing to exactly 1.0 should not emit warnings."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ConfidenceWeights(
                extraction=0.60,
                corroboration=0.20,
                recency=0.10,
                verification=0.10,
            )
        user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
        assert len(user_warnings) == 0


class TestDecayThresholdValidation:
    """Tests for decay threshold cross-validation on Settings."""

    def test_rejects_delete_equal_to_archive(self) -> None:
        """Settings should reject decay_delete_threshold == decay_archive_threshold."""
        with pytest.raises(ValueError, match="must be less than"):
            Settings(
                decay_delete_threshold=0.1,
                decay_archive_threshold=0.1,
            )

    def test_rejects_delete_greater_than_archive(self) -> None:
        """Settings should reject decay_delete_threshold > decay_archive_threshold."""
        with pytest.raises(ValueError, match="must be less than"):
            Settings(
                decay_delete_threshold=0.5,
                decay_archive_threshold=0.1,
            )

    def test_accepts_valid_order(self) -> None:
        """Default settings should pass validation (0.01 < 0.1)."""
        s = Settings()
        assert s.decay_delete_threshold < s.decay_archive_threshold

    def test_accepts_custom_valid_order(self) -> None:
        """Custom thresholds with correct ordering should pass."""
        s = Settings(
            decay_delete_threshold=0.05,
            decay_archive_threshold=0.2,
        )
        assert s.decay_delete_threshold == 0.05
        assert s.decay_archive_threshold == 0.2
