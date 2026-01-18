"""Unit tests for temporal reasoning module."""

from datetime import UTC, datetime

import pytest

from engram.temporal import (
    TEMPORAL_PATTERNS,
    StateChange,
    TemporalState,
    Timeline,
    detect_state_changes,
    detect_state_changes_regex,
)


class TestStateChangeModel:
    """Tests for StateChange model."""

    def test_create_state_change(self):
        """Should create a valid state change."""
        change = StateChange(
            entity="MongoDB",
            previous_state="user uses MongoDB",
            current_state="user does not use MongoDB",
            change_type="stopped",
            source_memory_id="mem_123",
            trigger_text="I no longer use MongoDB",
            user_id="user_456",
        )
        assert change.entity == "MongoDB"
        assert change.change_type == "stopped"
        assert change.inferred_previous is True  # Default

    def test_explicit_previous_state(self):
        """Should allow explicit previous state."""
        change = StateChange(
            entity="React",
            previous_state="user uses React",
            current_state="user uses Vue",
            change_type="changed",
            source_memory_id="mem_123",
            trigger_text="I switched from React to Vue",
            inferred_previous=False,
            user_id="user_456",
        )
        assert change.inferred_previous is False


class TestTemporalState:
    """Tests for TemporalState model."""

    def test_is_current(self):
        """Should correctly identify current state."""
        current = TemporalState(
            entity="Python",
            state="user uses Python",
            valid_from=datetime(2020, 1, 1, tzinfo=UTC),
            valid_until=None,  # Still current
        )
        assert current.is_current is True

        past = TemporalState(
            entity="Ruby",
            state="user uses Ruby",
            valid_from=datetime(2018, 1, 1, tzinfo=UTC),
            valid_until=datetime(2020, 1, 1, tzinfo=UTC),
        )
        assert past.is_current is False


class TestTimeline:
    """Tests for Timeline model."""

    def test_current_state(self):
        """Should find current state."""
        timeline = Timeline(
            entity="database",
            states=[
                TemporalState(
                    entity="database",
                    state="uses MySQL",
                    valid_until=datetime(2022, 1, 1, tzinfo=UTC),
                ),
                TemporalState(
                    entity="database",
                    state="uses PostgreSQL",
                    valid_from=datetime(2022, 1, 1, tzinfo=UTC),
                    valid_until=None,
                ),
            ],
        )
        current = timeline.current_state()
        assert current is not None
        assert current.state == "uses PostgreSQL"

    def test_state_at(self):
        """Should find state at specific time."""
        timeline = Timeline(
            entity="framework",
            states=[
                TemporalState(
                    entity="framework",
                    state="uses jQuery",
                    valid_from=datetime(2015, 1, 1, tzinfo=UTC),
                    valid_until=datetime(2018, 1, 1, tzinfo=UTC),
                ),
                TemporalState(
                    entity="framework",
                    state="uses React",
                    valid_from=datetime(2018, 1, 1, tzinfo=UTC),
                    valid_until=datetime(2023, 1, 1, tzinfo=UTC),
                ),
                TemporalState(
                    entity="framework",
                    state="uses Vue",
                    valid_from=datetime(2023, 1, 1, tzinfo=UTC),
                    valid_until=None,
                ),
            ],
        )

        state_2016 = timeline.state_at(datetime(2016, 6, 15, tzinfo=UTC))
        assert state_2016 is not None
        assert state_2016.state == "uses jQuery"

        state_2020 = timeline.state_at(datetime(2020, 6, 15, tzinfo=UTC))
        assert state_2020 is not None
        assert state_2020.state == "uses React"


class TestTemporalPatterns:
    """Tests for regex pattern definitions."""

    def test_patterns_are_valid(self):
        """All patterns should be valid regex."""
        import re

        for pattern, change_type, _groups in TEMPORAL_PATTERNS:
            # Should compile without error
            compiled = re.compile(pattern, re.IGNORECASE)
            assert compiled is not None
            assert change_type in (
                "started",
                "stopped",
                "changed",
                "upgraded",
                "downgraded",
                "resumed",
            )


class TestDetectStateChangesRegex:
    """Tests for regex-based detection."""

    def test_detect_no_longer_use(self):
        """Should detect 'no longer use' pattern."""
        changes = detect_state_changes_regex(
            text="I no longer use MongoDB",
            memory_id="mem_1",
            user_id="user_123",
        )
        assert len(changes) == 1
        assert changes[0].entity.lower() == "mongodb"
        assert changes[0].change_type == "stopped"

    def test_detect_switched_from_to(self):
        """Should detect 'switched from X to Y' pattern."""
        changes = detect_state_changes_regex(
            text="I switched from React to Vue",
            memory_id="mem_2",
            user_id="user_123",
        )
        # Should create two changes: stopped React, started Vue
        assert len(changes) == 2

        stopped = next(c for c in changes if c.change_type == "stopped")
        started = next(c for c in changes if c.change_type == "started")

        assert stopped.entity.lower() == "react"
        assert started.entity.lower() == "vue"

    def test_detect_started_using(self):
        """Should detect 'started using' pattern."""
        changes = detect_state_changes_regex(
            text="I've started using TypeScript",
            memory_id="mem_3",
            user_id="user_123",
        )
        assert len(changes) == 1
        assert changes[0].entity.lower() == "typescript"
        assert changes[0].change_type == "started"

    def test_detect_used_to(self):
        """Should detect 'used to' pattern."""
        changes = detect_state_changes_regex(
            text="I used to use Java",
            memory_id="mem_4",
            user_id="user_123",
        )
        assert len(changes) == 1
        assert changes[0].entity.lower() == "java"
        assert changes[0].change_type == "stopped"

    def test_no_detection_for_simple_preference(self):
        """Should not detect state changes for simple preferences."""
        changes = detect_state_changes_regex(
            text="I like Python",
            memory_id="mem_5",
            user_id="user_123",
        )
        assert len(changes) == 0

    def test_multiple_detections(self):
        """Should detect multiple state changes."""
        text = "I no longer use MongoDB and I switched from React to Vue"
        changes = detect_state_changes_regex(
            text=text,
            memory_id="mem_6",
            user_id="user_123",
        )
        # MongoDB stopped + React stopped + Vue started = 3
        assert len(changes) >= 3


class TestDetectStateChanges:
    """Tests for combined detection function."""

    @pytest.mark.asyncio
    async def test_detect_without_llm(self):
        """Should use regex when use_llm=False."""
        changes = await detect_state_changes(
            text="I no longer use MongoDB",
            memory_id="mem_1",
            user_id="user_123",
            use_llm=False,
        )
        assert len(changes) == 1
        assert changes[0].entity.lower() == "mongodb"

    @pytest.mark.asyncio
    async def test_detect_returns_all_user_fields(self):
        """Should set user_id and org_id on all changes."""
        changes = await detect_state_changes(
            text="I switched from React to Vue",
            memory_id="mem_1",
            user_id="user_123",
            org_id="org_456",
            use_llm=False,
        )
        for change in changes:
            assert change.user_id == "user_123"
            assert change.org_id == "org_456"
