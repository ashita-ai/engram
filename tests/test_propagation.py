"""Unit tests for confidence propagation module."""

import pytest

from engram.models import ConfidenceScore, ExtractionMethod, SemanticMemory
from engram.propagation import (
    PropagationConfig,
    PropagationResult,
    compute_link_strength,
    propagate_confidence,
    propagate_distrust,
)


def make_semantic_memory(
    id: str,
    content: str,
    confidence: float = 0.7,
    related_ids: list[str] | None = None,
    link_types: dict[str, str] | None = None,
    extraction_method: ExtractionMethod = ExtractionMethod.INFERRED,
) -> SemanticMemory:
    """Create a test semantic memory."""
    return SemanticMemory(
        id=id,
        content=content,
        source_episode_ids=["ep_1"],
        user_id="user_123",
        embedding=[0.1, 0.2, 0.3, 0.4],
        confidence=ConfidenceScore(
            value=confidence,
            extraction_method=extraction_method,
            extraction_base=confidence,
        ),
        related_ids=related_ids or [],
        link_types=link_types or {},
    )


class TestPropagationConfig:
    """Tests for PropagationConfig."""

    def test_default_config(self):
        """Should use sensible defaults."""
        config = PropagationConfig()
        assert config.damping_factor == 0.85
        assert config.max_iterations == 10
        assert config.convergence_threshold == 0.001

    def test_custom_config(self):
        """Should accept custom values."""
        config = PropagationConfig(
            damping_factor=0.9,
            max_iterations=20,
            convergence_threshold=0.0001,
        )
        assert config.damping_factor == 0.9
        assert config.max_iterations == 20


class TestPropagationResult:
    """Tests for PropagationResult model."""

    def test_create_result(self):
        """Should create valid result."""
        result = PropagationResult(
            memories_updated=5,
            iterations=3,
            converged=True,
            total_boost_applied=0.25,
        )
        assert result.memories_updated == 5
        assert result.converged is True


class TestComputeLinkStrength:
    """Tests for link strength calculation."""

    def test_verbatim_extraction_highest_strength(self):
        """Verbatim extraction should have highest strength."""
        source = make_semantic_memory(
            "src_1",
            "content",
            extraction_method=ExtractionMethod.VERBATIM,
        )
        target = make_semantic_memory("tgt_1", "target")

        strength = compute_link_strength(source, target)
        assert strength == 1.0  # Base strength for verbatim

    def test_inferred_extraction_lower_strength(self):
        """Inferred extraction should have lower strength."""
        source = make_semantic_memory(
            "src_1",
            "content",
            extraction_method=ExtractionMethod.INFERRED,
        )
        target = make_semantic_memory("tgt_1", "target")

        strength = compute_link_strength(source, target)
        assert strength == 0.6  # Base for inferred

    def test_contradicts_link_negative_strength(self):
        """Contradicts link type should have negative strength."""
        source = make_semantic_memory(
            "src_1",
            "content",
            link_types={"tgt_1": "contradicts"},
        )
        target = make_semantic_memory("tgt_1", "target")

        strength = compute_link_strength(source, target)
        assert strength < 0  # Negative for contradictions

    def test_elaborates_link_boosted_strength(self):
        """Elaborates link type should boost strength."""
        source = make_semantic_memory(
            "src_1",
            "content",
            extraction_method=ExtractionMethod.VERBATIM,
            link_types={"tgt_1": "elaborates"},
        )
        target = make_semantic_memory("tgt_1", "target")

        strength = compute_link_strength(source, target)
        assert strength > 1.0  # Boosted


class TestPropagateConfidence:
    """Tests for confidence propagation algorithm."""

    @pytest.mark.asyncio
    async def test_empty_memories_converges(self):
        """Should converge immediately with no memories."""
        result = await propagate_confidence([])
        assert result.converged is True
        assert result.memories_updated == 0

    @pytest.mark.asyncio
    async def test_no_links_no_changes(self):
        """Memories without links should not change."""
        memories = [
            make_semantic_memory("m1", "content 1", confidence=0.5),
            make_semantic_memory("m2", "content 2", confidence=0.6),
        ]
        original_confidences = [m.confidence.value for m in memories]

        await propagate_confidence(memories)

        # Confidences should be unchanged (no links)
        for i, mem in enumerate(memories):
            assert mem.confidence.value == original_confidences[i]

    @pytest.mark.asyncio
    async def test_linked_memories_propagate(self):
        """Linked memories should influence each other."""
        # Memory 1 links to Memory 2
        m1 = make_semantic_memory(
            "m1",
            "high confidence memory",
            confidence=0.9,
            related_ids=["m2"],
            link_types={"m2": "related"},
        )
        m2 = make_semantic_memory(
            "m2",
            "linked memory",
            confidence=0.5,
            related_ids=["m1"],
            link_types={"m1": "related"},
        )

        memories = [m1, m2]

        result = await propagate_confidence(memories)

        # M2 should have received boost from M1
        assert result.memories_updated > 0

    @pytest.mark.asyncio
    async def test_respects_max_iterations(self):
        """Should stop at max iterations."""
        config = PropagationConfig(
            max_iterations=2,
            convergence_threshold=0.0,  # Never converge
        )

        m1 = make_semantic_memory("m1", "content", confidence=0.9, related_ids=["m2"])
        m2 = make_semantic_memory("m2", "content", confidence=0.5, related_ids=["m1"])

        result = await propagate_confidence([m1, m2], config)

        assert result.iterations == 2
        assert result.converged is False

    @pytest.mark.asyncio
    async def test_converges_when_stable(self):
        """Should converge when changes are below threshold."""
        config = PropagationConfig(
            convergence_threshold=1.0,  # Very high threshold
        )

        m1 = make_semantic_memory("m1", "content", confidence=0.7)
        result = await propagate_confidence([m1], config)

        assert result.converged is True

    @pytest.mark.asyncio
    async def test_low_confidence_skipped(self):
        """Memories below min confidence should not propagate."""
        config = PropagationConfig(
            min_confidence_for_propagation=0.6,
        )

        # M1 is below threshold, should not propagate
        m1 = make_semantic_memory(
            "m1",
            "low confidence",
            confidence=0.4,
            related_ids=["m2"],
        )
        m2 = make_semantic_memory(
            "m2",
            "target",
            confidence=0.5,
            related_ids=["m1"],
        )

        await propagate_confidence([m1, m2], config)

        # M2 should not have changed from M1's low confidence


class TestPropagateDistrust:
    """Tests for distrust propagation."""

    @pytest.mark.asyncio
    async def test_high_confidence_no_distrust(self):
        """High confidence memories should not propagate distrust."""
        source = make_semantic_memory("src", "content", confidence=0.8)
        targets = [make_semantic_memory("tgt", "target", related_ids=["src"])]

        penalized = await propagate_distrust(source, targets)
        assert penalized == 0

    @pytest.mark.asyncio
    async def test_low_confidence_propagates_distrust(self):
        """Low confidence memories should propagate distrust."""
        source = make_semantic_memory(
            "src",
            "contradicted content",
            confidence=0.3,
            related_ids=["tgt"],
        )
        target = make_semantic_memory(
            "tgt",
            "target",
            confidence=0.7,
            related_ids=["src"],
        )

        original_conf = target.confidence.value
        penalized = await propagate_distrust(source, [target])

        assert penalized == 1
        assert target.confidence.value < original_conf

    @pytest.mark.asyncio
    async def test_contradiction_stronger_distrust(self):
        """Contradictions should propagate more distrust."""
        source = make_semantic_memory(
            "src",
            "contradicting",
            confidence=0.3,
            related_ids=["tgt"],
            link_types={"tgt": "related"},
        )
        target_related = make_semantic_memory(
            "tgt",
            "target",
            confidence=0.7,
            related_ids=["src"],
            link_types={"src": "related"},
        )

        source_contradict = make_semantic_memory(
            "src2",
            "contradicting",
            confidence=0.3,
            related_ids=["tgt2"],
            link_types={"tgt2": "contradicts"},
        )
        target_contradict = make_semantic_memory(
            "tgt2",
            "target",
            confidence=0.7,
            related_ids=["src2"],
            link_types={"src2": "contradicts"},
        )

        await propagate_distrust(source, [target_related])
        await propagate_distrust(source_contradict, [target_contradict])

        # Contradiction should cause more penalty
        assert target_contradict.confidence.value < target_related.confidence.value

    @pytest.mark.asyncio
    async def test_distrust_floor(self):
        """Distrust should not reduce confidence below floor."""
        source = make_semantic_memory(
            "src",
            "very low",
            confidence=0.1,
            related_ids=["tgt"],
        )
        target = make_semantic_memory(
            "tgt",
            "target",
            confidence=0.2,
            related_ids=["src"],
        )

        await propagate_distrust(source, [target], penalty=0.5)

        # Should not go below 0.1 floor
        assert target.confidence.value >= 0.1
