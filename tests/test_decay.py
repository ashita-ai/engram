"""Tests for decay workflow."""

from __future__ import annotations

import pytest

from engram.workflows.decay import DecayResult


class TestDecayResult:
    """Tests for DecayResult model."""

    def test_create_result(self) -> None:
        """Test creating decay result."""
        result = DecayResult(
            memories_updated=10,
            memories_archived=5,
            memories_deleted=2,
        )
        assert result.memories_updated == 10
        assert result.memories_archived == 5
        assert result.memories_deleted == 2

    def test_create_empty_result(self) -> None:
        """Test creating empty decay result."""
        result = DecayResult(
            memories_updated=0,
            memories_archived=0,
            memories_deleted=0,
        )
        assert result.memories_updated == 0
        assert result.memories_archived == 0
        assert result.memories_deleted == 0

    def test_counts_non_negative(self) -> None:
        """Test counts cannot be negative."""
        with pytest.raises(ValueError):
            DecayResult(
                memories_updated=-1,
                memories_archived=0,
                memories_deleted=0,
            )

        with pytest.raises(ValueError):
            DecayResult(
                memories_updated=0,
                memories_archived=-1,
                memories_deleted=0,
            )

        with pytest.raises(ValueError):
            DecayResult(
                memories_updated=0,
                memories_archived=0,
                memories_deleted=-1,
            )

    def test_extra_fields_forbidden(self) -> None:
        """Test extra fields are rejected."""
        with pytest.raises(ValueError):
            DecayResult(
                memories_updated=0,
                memories_archived=0,
                memories_deleted=0,
                extra_field=42,  # type: ignore[call-arg]
            )
