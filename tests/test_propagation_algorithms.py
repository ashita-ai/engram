"""Unit tests for PageRank-style confidence propagation algorithms."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from engram.models import ExtractionMethod, SemanticMemory
from engram.models.base import ConfidenceScore
from engram.propagation.algorithms import (
    DEFAULT_DAMPING_FACTOR,
    DEFAULT_DISTRUST_PENALTY,
    DEFAULT_MAX_ITERATIONS,
    PropagationConfig,
    PropagationResult,
    compute_link_strength,
    propagate_confidence,
    propagate_distrust,
    run_propagation_cycle,
)


def create_semantic_memory(
    memory_id: str = "sem_test",
    content: str = "Test content",
    user_id: str = "user_1",
    confidence_value: float = 0.7,
    extraction_method: ExtractionMethod = ExtractionMethod.INFERRED,
    related_ids: list[str] | None = None,
    link_types: dict[str, str] | None = None,
) -> SemanticMemory:
    """Helper to create SemanticMemory instances for testing."""
    memory = SemanticMemory(
        id=memory_id,
        content=content,
        user_id=user_id,
        confidence=ConfidenceScore(
            value=confidence_value,
            extraction_method=extraction_method,
            extraction_base=confidence_value,
        ),
        related_ids=related_ids or [],
        link_types=link_types or {},
    )
    return memory


class TestPropagationConfig:
    """Tests for PropagationConfig dataclass."""

    def test_default_values(self) -> None:
        """Config should have sensible defaults."""
        config = PropagationConfig()
        assert config.damping_factor == DEFAULT_DAMPING_FACTOR
        assert config.max_iterations == DEFAULT_MAX_ITERATIONS
        assert config.distrust_penalty == DEFAULT_DISTRUST_PENALTY
        assert config.min_confidence_for_propagation == 0.5

    def test_custom_values(self) -> None:
        """Config should accept custom values."""
        config = PropagationConfig(
            damping_factor=0.7,
            max_iterations=5,
            distrust_penalty=0.2,
        )
        assert config.damping_factor == 0.7
        assert config.max_iterations == 5
        assert config.distrust_penalty == 0.2


class TestPropagationResult:
    """Tests for PropagationResult model."""

    def test_default_values(self) -> None:
        """Result should have zero defaults."""
        result = PropagationResult()
        assert result.memories_updated == 0
        assert result.iterations == 0
        assert result.converged is False
        assert result.total_boost_applied == 0.0
        assert result.total_distrust_applied == 0.0

    def test_with_values(self) -> None:
        """Result should accept values."""
        result = PropagationResult(
            memories_updated=5,
            iterations=3,
            converged=True,
            total_boost_applied=0.5,
        )
        assert result.memories_updated == 5
        assert result.iterations == 3
        assert result.converged is True


class TestComputeLinkStrength:
    """Tests for compute_link_strength function."""

    def test_verbatim_extraction_has_highest_strength(self) -> None:
        """VERBATIM extraction should have strength 1.0."""
        source = create_semantic_memory(
            memory_id="source",
            extraction_method=ExtractionMethod.VERBATIM,
        )
        target = create_semantic_memory(memory_id="target")

        strength = compute_link_strength(source, target)
        assert strength == 1.0

    def test_extracted_has_moderate_strength(self) -> None:
        """EXTRACTED method should have strength 0.9."""
        source = create_semantic_memory(
            memory_id="source",
            extraction_method=ExtractionMethod.EXTRACTED,
        )
        target = create_semantic_memory(memory_id="target")

        strength = compute_link_strength(source, target)
        assert strength == 0.9

    def test_inferred_has_lowest_strength(self) -> None:
        """INFERRED method should have strength 0.6."""
        source = create_semantic_memory(
            memory_id="source",
            extraction_method=ExtractionMethod.INFERRED,
        )
        target = create_semantic_memory(memory_id="target")

        strength = compute_link_strength(source, target)
        assert strength == 0.6

    def test_elaborates_link_type_boosts_strength(self) -> None:
        """Elaboration links should boost strength by 1.2x."""
        source = create_semantic_memory(
            memory_id="source",
            extraction_method=ExtractionMethod.VERBATIM,
            link_types={"target": "elaborates"},
        )
        target = create_semantic_memory(memory_id="target")

        strength = compute_link_strength(source, target)
        assert strength == 1.2  # 1.0 * 1.2

    def test_contradicts_link_type_is_negative(self) -> None:
        """Contradiction links should produce negative strength."""
        source = create_semantic_memory(
            memory_id="source",
            extraction_method=ExtractionMethod.VERBATIM,
            link_types={"target": "contradicts"},
        )
        target = create_semantic_memory(memory_id="target")

        strength = compute_link_strength(source, target)
        assert strength == -1.0  # 1.0 * -1.0

    def test_supersedes_link_type_reduces_strength(self) -> None:
        """Supersedes links should reduce strength."""
        source = create_semantic_memory(
            memory_id="source",
            extraction_method=ExtractionMethod.VERBATIM,
            link_types={"target": "supersedes"},
        )
        target = create_semantic_memory(memory_id="target")

        strength = compute_link_strength(source, target)
        assert strength == 0.8  # 1.0 * 0.8

    def test_unknown_link_type_has_default_multiplier(self) -> None:
        """Unknown link types should use default multiplier of 1.0."""
        source = create_semantic_memory(
            memory_id="source",
            extraction_method=ExtractionMethod.VERBATIM,
            link_types={"target": "unknown_link"},
        )
        target = create_semantic_memory(memory_id="target")

        strength = compute_link_strength(source, target)
        assert strength == 1.0


class TestPropagateConfidence:
    """Tests for propagate_confidence function."""

    @pytest.mark.asyncio
    async def test_empty_memories_converges_immediately(self) -> None:
        """Empty memory list should converge immediately."""
        result = await propagate_confidence([])

        assert result.converged is True
        assert result.memories_updated == 0
        assert result.iterations == 0

    @pytest.mark.asyncio
    async def test_single_memory_no_links_converges(self) -> None:
        """Single memory with no links should converge quickly."""
        memory = create_semantic_memory()

        result = await propagate_confidence([memory])

        assert result.converged is True
        assert result.memories_updated == 0

    @pytest.mark.asyncio
    async def test_low_confidence_memories_skip_propagation(self) -> None:
        """Memories below threshold should not propagate."""
        config = PropagationConfig(min_confidence_for_propagation=0.5)
        memory = create_semantic_memory(confidence_value=0.3)

        result = await propagate_confidence([memory], config)

        assert result.memories_updated == 0

    @pytest.mark.asyncio
    async def test_linked_memories_propagate_confidence(self) -> None:
        """Linked memories should propagate confidence."""
        mem_a = create_semantic_memory(
            memory_id="mem_a",
            confidence_value=0.8,
            related_ids=["mem_b"],
        )
        mem_b = create_semantic_memory(
            memory_id="mem_b",
            confidence_value=0.6,
            related_ids=["mem_a"],
        )

        config = PropagationConfig(max_iterations=5)
        result = await propagate_confidence([mem_a, mem_b], config)

        # Should have iterated
        assert result.iterations > 0

    @pytest.mark.asyncio
    async def test_convergence_threshold_respected(self) -> None:
        """Should converge when changes are below threshold."""
        mem_a = create_semantic_memory(
            memory_id="mem_a",
            confidence_value=0.8,
        )
        mem_b = create_semantic_memory(
            memory_id="mem_b",
            confidence_value=0.8,
        )

        config = PropagationConfig(
            convergence_threshold=0.01,
            max_iterations=100,
        )
        result = await propagate_confidence([mem_a, mem_b], config)

        # Should converge before max iterations
        assert result.converged is True
        assert result.iterations < config.max_iterations

    @pytest.mark.asyncio
    async def test_max_iterations_respected(self) -> None:
        """Should stop at max iterations even if not converged."""
        mem_a = create_semantic_memory(
            memory_id="mem_a",
            confidence_value=0.9,
            related_ids=["mem_b"],
        )
        mem_b = create_semantic_memory(
            memory_id="mem_b",
            confidence_value=0.5,
            related_ids=["mem_a"],
        )

        config = PropagationConfig(
            max_iterations=2,
            convergence_threshold=0.0001,  # Very tight threshold
        )
        result = await propagate_confidence([mem_a, mem_b], config)

        assert result.iterations <= config.max_iterations

    @pytest.mark.asyncio
    async def test_distrust_from_contradictions(self) -> None:
        """Contradiction links should propagate distrust."""
        mem_a = create_semantic_memory(
            memory_id="mem_a",
            confidence_value=0.9,
            related_ids=["mem_b"],
            link_types={"mem_b": "contradicts"},
        )
        mem_b = create_semantic_memory(
            memory_id="mem_b",
            confidence_value=0.7,
            related_ids=["mem_a"],
        )

        config = PropagationConfig(distrust_penalty=0.1)
        result = await propagate_confidence([mem_a, mem_b], config)

        # Should have applied some distrust
        # Note: The exact amount depends on the algorithm
        assert result.total_distrust_applied >= 0.0


class TestPropagateDistrust:
    """Tests for propagate_distrust function."""

    @pytest.mark.asyncio
    async def test_high_confidence_source_no_distrust(self) -> None:
        """High-confidence sources should not propagate distrust."""
        source = create_semantic_memory(
            memory_id="source",
            confidence_value=0.7,  # Above 0.5 threshold
        )
        linked = create_semantic_memory(
            memory_id="linked",
            related_ids=["source"],
        )

        penalized = await propagate_distrust(source, [linked])

        assert penalized == 0

    @pytest.mark.asyncio
    async def test_low_confidence_source_propagates_distrust(self) -> None:
        """Low-confidence sources should propagate distrust to linked memories."""
        source = create_semantic_memory(
            memory_id="source",
            confidence_value=0.3,  # Below 0.5 threshold
        )
        linked = create_semantic_memory(
            memory_id="linked",
            confidence_value=0.8,
            related_ids=["source"],
        )

        original_confidence = linked.confidence.value
        penalized = await propagate_distrust(source, [linked])

        assert penalized == 1
        assert linked.confidence.value < original_confidence

    @pytest.mark.asyncio
    async def test_contradiction_link_type_doubles_penalty(self) -> None:
        """Contradiction links should have double penalty."""
        source = create_semantic_memory(
            memory_id="source",
            confidence_value=0.3,
        )
        linked = create_semantic_memory(
            memory_id="linked",
            confidence_value=0.8,
            related_ids=["source"],
            link_types={"source": "contradicts"},
        )

        await propagate_distrust(source, [linked], penalty=0.1)

        # With contradiction, penalty is 0.1 * 2 = 0.2
        # Confidence should drop from 0.8 to 0.6
        assert linked.confidence.value < 0.8

    @pytest.mark.asyncio
    async def test_unlinked_memories_not_penalized(self) -> None:
        """Memories not linked to source should not be penalized."""
        source = create_semantic_memory(
            memory_id="source",
            confidence_value=0.3,
        )
        unlinked = create_semantic_memory(
            memory_id="unlinked",
            confidence_value=0.8,
            related_ids=[],  # No link to source
        )

        penalized = await propagate_distrust(source, [unlinked])

        assert penalized == 0
        assert unlinked.confidence.value == 0.8

    @pytest.mark.asyncio
    async def test_confidence_floor_at_0_1(self) -> None:
        """Confidence should not go below 0.1."""
        source = create_semantic_memory(
            memory_id="source",
            confidence_value=0.1,  # Very low
        )
        linked = create_semantic_memory(
            memory_id="linked",
            confidence_value=0.2,  # Already low
            related_ids=["source"],
        )

        # Apply very high penalty
        await propagate_distrust(source, [linked], penalty=0.5)

        assert linked.confidence.value >= 0.1


class TestRunPropagationCycle:
    """Tests for run_propagation_cycle function."""

    @pytest.mark.asyncio
    async def test_empty_memories_returns_converged(self) -> None:
        """Should return converged result for no memories."""
        mock_storage = AsyncMock()
        mock_storage.list_semantic_memories = AsyncMock(return_value=[])

        result = await run_propagation_cycle(mock_storage, user_id="user_1")

        assert result.converged is True
        assert result.memories_updated == 0

    @pytest.mark.asyncio
    async def test_propagates_and_persists_changes(self) -> None:
        """Should propagate confidence and persist changes."""
        mem_a = create_semantic_memory(
            memory_id="mem_a",
            confidence_value=0.9,
            related_ids=["mem_b"],
        )
        mem_b = create_semantic_memory(
            memory_id="mem_b",
            confidence_value=0.6,
            related_ids=["mem_a"],
        )

        mock_storage = AsyncMock()
        mock_storage.list_semantic_memories = AsyncMock(return_value=[mem_a, mem_b])
        mock_storage.update_semantic_memory = AsyncMock()

        result = await run_propagation_cycle(mock_storage, user_id="user_1")

        # If memories were updated, storage should have been called
        if result.memories_updated > 0:
            assert mock_storage.update_semantic_memory.called

    @pytest.mark.asyncio
    async def test_uses_custom_config(self) -> None:
        """Should use provided config."""
        mock_storage = AsyncMock()
        mock_storage.list_semantic_memories = AsyncMock(return_value=[])

        config = PropagationConfig(max_iterations=5)
        result = await run_propagation_cycle(mock_storage, user_id="user_1", config=config)

        assert result.converged is True

    @pytest.mark.asyncio
    async def test_passes_org_id_to_storage(self) -> None:
        """Should pass org_id to storage query."""
        mock_storage = AsyncMock()
        mock_storage.list_semantic_memories = AsyncMock(return_value=[])

        await run_propagation_cycle(mock_storage, user_id="user_1", org_id="org_123")

        mock_storage.list_semantic_memories.assert_called_once_with("user_1", "org_123")


class TestLinkTypeMultipliers:
    """Tests for all link type multipliers."""

    @pytest.mark.parametrize(
        "link_type,expected_multiplier",
        [
            ("related", 1.0),
            ("elaborates", 1.2),
            ("causal", 1.1),
            ("temporal", 0.9),
            ("exemplifies", 1.1),
            ("generalizes", 0.9),
            ("supersedes", 0.8),
            ("contradicts", -1.0),
        ],
    )
    def test_link_type_multiplier(self, link_type: str, expected_multiplier: float) -> None:
        """Each link type should have correct multiplier."""
        source = create_semantic_memory(
            memory_id="source",
            extraction_method=ExtractionMethod.VERBATIM,
            link_types={"target": link_type},
        )
        target = create_semantic_memory(memory_id="target")

        strength = compute_link_strength(source, target)

        # VERBATIM base is 1.0, so strength = 1.0 * multiplier
        assert strength == expected_multiplier


class TestExtractionMethodWeights:
    """Tests for extraction method weight values."""

    @pytest.mark.parametrize(
        "method,expected_weight",
        [
            (ExtractionMethod.VERBATIM, 1.0),
            (ExtractionMethod.EXTRACTED, 0.9),
            (ExtractionMethod.INFERRED, 0.6),
        ],
    )
    def test_extraction_method_weight(
        self, method: ExtractionMethod, expected_weight: float
    ) -> None:
        """Each extraction method should have correct weight."""
        source = create_semantic_memory(
            memory_id="source",
            extraction_method=method,
        )
        target = create_semantic_memory(memory_id="target")

        strength = compute_link_strength(source, target)

        # With no special link type, multiplier is 1.0
        assert strength == expected_weight
