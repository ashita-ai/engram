"""Tests for procedural synthesis workflow."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from engram.workflows.promotion import (
    SynthesisOutput,
    SynthesisResult,
    _format_semantics_for_llm,
    run_synthesis,
)


class TestSynthesisResult:
    """Tests for SynthesisResult model."""

    def test_create_result(self) -> None:
        """Test creating synthesis result."""
        result = SynthesisResult(
            semantics_analyzed=5,
            procedural_created=True,
            procedural_updated=False,
            procedural_id="proc_123",
        )
        assert result.semantics_analyzed == 5
        assert result.procedural_created is True
        assert result.procedural_updated is False
        assert result.procedural_id == "proc_123"

    def test_create_empty_result(self) -> None:
        """Test creating empty result."""
        result = SynthesisResult(
            semantics_analyzed=0,
            procedural_created=False,
            procedural_updated=False,
        )
        assert result.semantics_analyzed == 0
        assert result.procedural_created is False
        assert result.procedural_id is None


class TestSynthesisOutput:
    """Tests for SynthesisOutput model."""

    def test_create_output(self) -> None:
        """Test creating synthesis output."""
        output = SynthesisOutput(
            behavioral_profile="The user prefers Python and detailed explanations.",
            communication_style="Concise and technical",
            technical_preferences=["Python", "PostgreSQL"],
            work_patterns=["Asks clarifying questions"],
            keywords=["python", "technical"],
        )
        assert "Python" in output.behavioral_profile
        assert output.communication_style == "Concise and technical"
        assert "Python" in output.technical_preferences
        assert len(output.work_patterns) == 1

    def test_create_minimal_output(self) -> None:
        """Test creating output with only required field."""
        output = SynthesisOutput(
            behavioral_profile="The user is a developer.",
        )
        assert output.behavioral_profile == "The user is a developer."
        assert output.communication_style == ""
        assert output.technical_preferences == []
        assert output.work_patterns == []


class TestFormatSemanticsForLLM:
    """Tests for _format_semantics_for_llm function."""

    def test_format_single_memory(self) -> None:
        """Test formatting a single semantic memory."""
        from engram.models import SemanticMemory

        memory = SemanticMemory(
            content="User prefers Python",
            user_id="test_user",
            embedding=[0.1] * 384,
        )
        memory.keywords = ["python", "preference"]
        memory.context = "programming"

        result = _format_semantics_for_llm([memory])

        assert "## Memory 1" in result
        assert "User prefers Python" in result
        assert "python" in result
        assert "programming" in result

    def test_format_multiple_memories(self) -> None:
        """Test formatting multiple semantic memories."""
        from engram.models import SemanticMemory

        memories = [
            SemanticMemory(
                content="User prefers Python",
                user_id="test_user",
                embedding=[0.1] * 384,
            ),
            SemanticMemory(
                content="User works at TechCorp",
                user_id="test_user",
                embedding=[0.2] * 384,
            ),
        ]

        result = _format_semantics_for_llm(memories)

        assert "## Memory 1" in result
        assert "## Memory 2" in result
        assert "User prefers Python" in result
        assert "User works at TechCorp" in result

    def test_format_empty_list(self) -> None:
        """Test formatting empty list."""
        result = _format_semantics_for_llm([])
        assert "# Semantic Memories to Synthesize" in result


class TestRunSynthesis:
    """Tests for run_synthesis workflow."""

    @pytest.mark.asyncio
    async def test_no_semantics_returns_empty_result(self) -> None:
        """Test that empty semantic list returns zero counts."""
        mock_storage = AsyncMock()
        mock_storage.list_semantic_memories = AsyncMock(return_value=[])

        mock_embedder = AsyncMock()

        result = await run_synthesis(
            storage=mock_storage,
            embedder=mock_embedder,
            user_id="test_user",
        )

        assert result.semantics_analyzed == 0
        assert result.procedural_created is False
        assert result.procedural_updated is False

    @pytest.mark.asyncio
    async def test_creates_procedural_from_semantics(self) -> None:
        """Test creation of procedural memory from semantic memories."""
        from engram.models import SemanticMemory

        memories = [
            SemanticMemory(
                content="User prefers Python programming",
                user_id="test_user",
                embedding=[0.1] * 384,
                source_episode_ids=["ep_001"],
            ),
            SemanticMemory(
                content="User works as a backend developer",
                user_id="test_user",
                embedding=[0.2] * 384,
                source_episode_ids=["ep_002"],
            ),
        ]

        mock_storage = AsyncMock()
        mock_storage.list_semantic_memories = AsyncMock(return_value=memories)
        mock_storage.list_procedural_memories = AsyncMock(return_value=[])  # No existing
        mock_storage.store_procedural = AsyncMock(return_value="proc_123")

        mock_embedder = AsyncMock()
        mock_embedder.embed = AsyncMock(return_value=[0.1] * 384)

        mock_synthesis = SynthesisOutput(
            behavioral_profile="The user is a Python backend developer.",
            communication_style="Technical",
            technical_preferences=["Python"],
        )

        with patch(
            "engram.workflows.promotion._synthesize_behavioral_profile",
            return_value=mock_synthesis,
        ):
            result = await run_synthesis(
                storage=mock_storage,
                embedder=mock_embedder,
                user_id="test_user",
            )

        assert result.semantics_analyzed == 2
        assert result.procedural_created is True
        assert result.procedural_updated is False
        mock_storage.store_procedural.assert_called_once()

    @pytest.mark.asyncio
    async def test_updates_existing_procedural(self) -> None:
        """Test update of existing procedural memory."""
        from engram.models import ProceduralMemory, SemanticMemory

        memories = [
            SemanticMemory(
                content="User prefers Python programming",
                user_id="test_user",
                embedding=[0.1] * 384,
                source_episode_ids=["ep_001"],
            ),
        ]

        existing_procedural = ProceduralMemory(
            id="proc_existing",
            content="Old profile",
            user_id="test_user",
            embedding=[0.1] * 384,
        )

        mock_storage = AsyncMock()
        mock_storage.list_semantic_memories = AsyncMock(return_value=memories)
        mock_storage.list_procedural_memories = AsyncMock(return_value=[existing_procedural])
        mock_storage.update_procedural_memory = AsyncMock(return_value=True)

        mock_embedder = AsyncMock()
        mock_embedder.embed = AsyncMock(return_value=[0.1] * 384)

        mock_synthesis = SynthesisOutput(
            behavioral_profile="Updated profile.",
        )

        with patch(
            "engram.workflows.promotion._synthesize_behavioral_profile",
            return_value=mock_synthesis,
        ):
            result = await run_synthesis(
                storage=mock_storage,
                embedder=mock_embedder,
                user_id="test_user",
            )

        assert result.semantics_analyzed == 1
        assert result.procedural_created is False
        assert result.procedural_updated is True
        assert result.procedural_id == "proc_existing"
        mock_storage.update_procedural_memory.assert_called_once()

    @pytest.mark.asyncio
    async def test_collects_source_episode_ids(self) -> None:
        """Test that source episode IDs are collected from all semantics."""
        from engram.models import SemanticMemory

        memories = [
            SemanticMemory(
                content="Memory 1",
                user_id="test_user",
                embedding=[0.1] * 384,
                source_episode_ids=["ep_001", "ep_002"],
            ),
            SemanticMemory(
                content="Memory 2",
                user_id="test_user",
                embedding=[0.2] * 384,
                source_episode_ids=["ep_002", "ep_003"],  # ep_002 is duplicate
            ),
        ]

        mock_storage = AsyncMock()
        mock_storage.list_semantic_memories = AsyncMock(return_value=memories)
        mock_storage.list_procedural_memories = AsyncMock(return_value=[])
        mock_storage.store_procedural = AsyncMock(return_value="proc_123")

        mock_embedder = AsyncMock()
        mock_embedder.embed = AsyncMock(return_value=[0.1] * 384)

        mock_synthesis = SynthesisOutput(behavioral_profile="Profile.")

        with patch(
            "engram.workflows.promotion._synthesize_behavioral_profile",
            return_value=mock_synthesis,
        ):
            await run_synthesis(
                storage=mock_storage,
                embedder=mock_embedder,
                user_id="test_user",
            )

        # Check the procedural was created with deduplicated episode IDs
        call_args = mock_storage.store_procedural.call_args
        procedural = call_args[0][0]
        assert set(procedural.source_episode_ids) == {"ep_001", "ep_002", "ep_003"}
