"""Tests for consolidation workflow."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from engram.config import Settings
from engram.workflows import DurableAgentFactory, init_workflows, shutdown_workflows
from engram.workflows.consolidation import (
    ConsolidationResult,
    ExtractedFact,
    IdentifiedLink,
    LLMExtractionResult,
    MapReduceSummary,
    MemoryEvolution,
    SummaryOutput,
    _find_matching_memory,
    format_episodes_for_llm,
    run_consolidation,
)


class TestExtractedFact:
    """Tests for ExtractedFact model (legacy, kept for backwards compatibility)."""

    def test_create_with_defaults(self) -> None:
        """Test creating extracted fact with defaults."""
        fact = ExtractedFact(content="User's email is test@example.com")
        assert fact.content == "User's email is test@example.com"
        assert fact.confidence == 0.6
        assert fact.source_context == ""

    def test_create_with_all_fields(self) -> None:
        """Test creating extracted fact with all fields."""
        fact = ExtractedFact(
            content="User prefers dark mode",
            confidence=0.8,
            source_context="When the user said 'I like dark themes'",
        )
        assert fact.content == "User prefers dark mode"
        assert fact.confidence == 0.8
        assert "dark themes" in fact.source_context

    def test_confidence_bounds(self) -> None:
        """Test confidence must be between 0 and 1."""
        with pytest.raises(ValueError):
            ExtractedFact(content="test", confidence=1.5)
        with pytest.raises(ValueError):
            ExtractedFact(content="test", confidence=-0.1)

    # A-MEM inspired tests
    def test_amem_fields_defaults(self) -> None:
        """Test A-MEM fields have sensible defaults."""
        fact = ExtractedFact(content="User prefers Python")
        assert fact.keywords == []
        assert fact.tags == []
        assert fact.context == ""

    def test_create_with_amem_fields(self) -> None:
        """Test creating fact with A-MEM fields."""
        fact = ExtractedFact(
            content="User prefers Python for scripting",
            confidence=0.8,
            keywords=["python", "scripting", "preference"],
            tags=["preference", "technical"],
            context="programming languages",
        )
        assert fact.keywords == ["python", "scripting", "preference"]
        assert fact.tags == ["preference", "technical"]
        assert fact.context == "programming languages"


class TestMemoryEvolution:
    """Tests for MemoryEvolution model (legacy, kept for backwards compatibility)."""

    def test_create_evolution(self) -> None:
        """Test creating a memory evolution."""
        evolution = MemoryEvolution(
            target_content="User prefers Python",
            add_tags=["programming", "preference"],
            add_keywords=["python", "language"],
            update_context="programming tools",
            reason="New context about programming preferences",
        )
        assert evolution.target_content == "User prefers Python"
        assert evolution.add_tags == ["programming", "preference"]
        assert evolution.add_keywords == ["python", "language"]
        assert evolution.update_context == "programming tools"
        assert "programming preferences" in evolution.reason

    def test_evolution_defaults(self) -> None:
        """Test evolution has sensible defaults."""
        evolution = MemoryEvolution(target_content="Some memory")
        assert evolution.add_tags == []
        assert evolution.add_keywords == []
        assert evolution.update_context == ""
        assert evolution.reason == ""


class TestIdentifiedLink:
    """Tests for IdentifiedLink model."""

    def test_create_link(self) -> None:
        """Test creating a link."""
        link = IdentifiedLink(
            source_content="User's email",
            target_content="Work contact information",
            relationship="is_part_of",
        )
        assert link.source_content == "User's email"
        assert link.target_content == "Work contact information"
        assert link.relationship == "is_part_of"


class TestLLMExtractionResult:
    """Tests for LLMExtractionResult model (legacy, kept for backwards compatibility)."""

    def test_create_empty(self) -> None:
        """Test creating empty result."""
        result = LLMExtractionResult()
        assert result.semantic_facts == []
        assert result.links == []
        assert result.contradictions == []
        assert result.evolutions == []

    def test_create_with_data(self) -> None:
        """Test creating result with data."""
        result = LLMExtractionResult(
            semantic_facts=[
                ExtractedFact(content="User works at Acme Corp"),
                ExtractedFact(content="User prefers Python", confidence=0.7),
            ],
            links=[
                IdentifiedLink(
                    source_content="Acme Corp",
                    target_content="work email domain",
                    relationship="determines",
                )
            ],
            contradictions=["User previously said they work at Beta Inc"],
        )
        assert len(result.semantic_facts) == 2
        assert len(result.links) == 1
        assert len(result.contradictions) == 1

    def test_create_with_evolutions(self) -> None:
        """Test creating result with A-MEM evolutions."""
        result = LLMExtractionResult(
            semantic_facts=[ExtractedFact(content="User prefers TypeScript")],
            evolutions=[
                MemoryEvolution(
                    target_content="User prefers JavaScript",
                    add_tags=["typescript_related"],
                    reason="TypeScript preference extends JS preference",
                )
            ],
        )
        assert len(result.evolutions) == 1
        assert result.evolutions[0].target_content == "User prefers JavaScript"


class TestSummaryOutput:
    """Tests for SummaryOutput model (new summarization output)."""

    def test_create_with_defaults(self) -> None:
        """Test creating summary output with defaults."""
        output = SummaryOutput(summary="The user prefers Python programming.")
        assert output.summary == "The user prefers Python programming."
        assert output.key_facts == []
        assert output.keywords == []
        assert output.context == ""

    def test_create_with_all_fields(self) -> None:
        """Test creating summary output with all fields."""
        output = SummaryOutput(
            summary="The user is a Python developer who works at TechCorp.",
            key_facts=["Works at TechCorp", "Prefers Python"],
            keywords=["python", "developer", "techcorp"],
            context="professional background",
        )
        assert "TechCorp" in output.summary
        assert len(output.key_facts) == 2
        assert "python" in output.keywords
        assert output.context == "professional background"


class TestMapReduceSummary:
    """Tests for MapReduceSummary model."""

    def test_create_summary(self) -> None:
        """Test creating map-reduce summary."""
        summary = MapReduceSummary(
            summary="Combined summary of user information.",
            keywords=["user", "info"],
            context="general",
        )
        assert summary.summary == "Combined summary of user information."
        assert summary.keywords == ["user", "info"]
        assert summary.context == "general"


class TestConsolidationResult:
    """Tests for ConsolidationResult model."""

    def test_create_result(self) -> None:
        """Test creating consolidation result."""
        result = ConsolidationResult(
            episodes_processed=10,
            semantic_memories_created=1,
            links_created=3,
            compression_ratio=10.0,
        )
        assert result.episodes_processed == 10
        assert result.semantic_memories_created == 1
        assert result.links_created == 3
        assert result.compression_ratio == 10.0

    def test_counts_non_negative(self) -> None:
        """Test counts cannot be negative."""
        with pytest.raises(ValueError):
            ConsolidationResult(
                episodes_processed=-1,
                semantic_memories_created=0,
                links_created=0,
            )

    def test_default_values(self) -> None:
        """Test default values for optional fields."""
        result = ConsolidationResult(
            episodes_processed=5,
            semantic_memories_created=1,
        )
        assert result.links_created == 0
        assert result.compression_ratio == 0.0


class TestFormatEpisodesForLLM:
    """Tests for format_episodes_for_llm function."""

    def test_format_single_episode(self) -> None:
        """Test formatting a single episode."""
        episodes = [{"id": "ep_123", "role": "user", "content": "Hello, world!"}]
        result = format_episodes_for_llm(episodes)
        assert "[USER] (ep_123)" in result
        assert "Hello, world!" in result

    def test_format_multiple_episodes(self) -> None:
        """Test formatting multiple episodes."""
        episodes = [
            {"id": "ep_1", "role": "user", "content": "What's my email?"},
            {"id": "ep_2", "role": "assistant", "content": "Your email is test@example.com"},
            {"id": "ep_3", "role": "user", "content": "Thanks!"},
        ]
        result = format_episodes_for_llm(episodes)
        assert "[USER] (ep_1)" in result
        assert "[ASSISTANT] (ep_2)" in result
        assert "[USER] (ep_3)" in result
        assert "test@example.com" in result

    def test_format_empty_list(self) -> None:
        """Test formatting empty list."""
        result = format_episodes_for_llm([])
        # New consolidation uses "Summarize" instead of "Analyze"
        assert "# Conversation Episodes to Summarize" in result


class TestRunConsolidation:
    """Tests for run_consolidation workflow."""

    @pytest.mark.asyncio
    async def test_no_episodes_returns_empty_result(self) -> None:
        """Test that empty episode list returns zero counts."""
        mock_storage = AsyncMock()
        mock_storage.get_unsummarized_episodes = AsyncMock(return_value=[])

        mock_embedder = AsyncMock()

        result = await run_consolidation(
            storage=mock_storage,
            embedder=mock_embedder,
            user_id="test_user",
            org_id="test_org",
        )

        assert result.episodes_processed == 0
        assert result.semantic_memories_created == 0
        assert result.links_created == 0

    @pytest.mark.asyncio
    async def test_summarizes_episodes_into_single_memory(self) -> None:
        """Test that multiple episodes are summarized into one semantic memory."""
        from engram.models import Episode

        # Create mock episodes
        mock_episodes = []
        for i in range(3):
            mock_episode = MagicMock(spec=Episode)
            mock_episode.id = f"ep_{i}"
            mock_episode.role = "user"
            mock_episode.content = f"Message {i} content"
            mock_episodes.append(mock_episode)

        mock_storage = AsyncMock()
        mock_storage.get_unsummarized_episodes = AsyncMock(return_value=mock_episodes)
        mock_storage.store_semantic = AsyncMock(return_value="sem_123")
        mock_storage.mark_episodes_summarized = AsyncMock(return_value=3)
        mock_storage.list_semantic_memories = AsyncMock(return_value=[])

        mock_embedder = AsyncMock()
        mock_embedder.embed = AsyncMock(return_value=[0.1] * 384)

        # Mock the LLM summarization
        mock_summary = SummaryOutput(
            summary="The user sent three messages.",
            keywords=["messages", "user"],
            context="general",
        )

        with patch(
            "engram.workflows.consolidation._summarize_chunk",
            return_value=mock_summary,
        ):
            result = await run_consolidation(
                storage=mock_storage,
                embedder=mock_embedder,
                user_id="test_user",
                org_id="test_org",
            )

        # Should process all episodes into ONE semantic memory
        assert result.episodes_processed == 3
        assert result.semantic_memories_created == 1
        assert result.compression_ratio == 3.0

        # Should mark all episodes as summarized
        mock_storage.mark_episodes_summarized.assert_called_once()
        call_args = mock_storage.mark_episodes_summarized.call_args
        assert len(call_args[0][0]) == 3  # 3 episode IDs
        assert call_args[0][1] == "test_user"  # user_id

    @pytest.mark.asyncio
    async def test_excludes_system_prompts_from_summarization(self) -> None:
        """Test that system prompts are marked as summarized but not included in summary."""
        from engram.models import Episode

        user_episode = MagicMock(spec=Episode)
        user_episode.id = "ep_user"
        user_episode.role = "user"
        user_episode.content = "Hello!"

        system_episode = MagicMock(spec=Episode)
        system_episode.id = "ep_system"
        system_episode.role = "system"
        system_episode.content = "You are a helpful assistant."

        mock_storage = AsyncMock()
        mock_storage.get_unsummarized_episodes = AsyncMock(
            return_value=[user_episode, system_episode]
        )
        mock_storage.store_semantic = AsyncMock(return_value="sem_123")
        mock_storage.mark_episodes_summarized = AsyncMock(return_value=2)
        mock_storage.list_semantic_memories = AsyncMock(return_value=[])

        mock_embedder = AsyncMock()
        mock_embedder.embed = AsyncMock(return_value=[0.1] * 384)

        mock_summary = SummaryOutput(
            summary="The user said hello.",
            keywords=["greeting"],
            context="conversation",
        )

        with patch(
            "engram.workflows.consolidation._summarize_chunk",
            return_value=mock_summary,
        ) as mock_chunk:
            result = await run_consolidation(
                storage=mock_storage,
                embedder=mock_embedder,
                user_id="test_user",
                org_id="test_org",
            )

            # Verify only user episode was passed to summarization
            chunk_call = mock_chunk.call_args[0][0]
            assert len(chunk_call) == 1
            assert chunk_call[0]["role"] == "user"

        # Both episodes should be marked as summarized
        assert result.episodes_processed == 2
        call_args = mock_storage.mark_episodes_summarized.call_args
        episode_ids = call_args[0][0]
        assert "ep_user" in episode_ids
        assert "ep_system" in episode_ids


class TestDurableAgentFactory:
    """Tests for DurableAgentFactory with DBOS backend."""

    @pytest.fixture(autouse=True)
    def cleanup_workflows(self) -> None:
        """Ensure workflows are shut down after each test."""
        yield
        shutdown_workflows()

    def test_factory_backend_property(self) -> None:
        """Test factory returns configured backend."""
        settings = Settings(openai_api_key="sk-test", durable_backend="dbos")
        factory = DurableAgentFactory(settings)
        assert factory.backend == "dbos"

    def test_factory_prefect_backend(self) -> None:
        """Test factory accepts prefect backend."""
        settings = Settings(openai_api_key="sk-test", durable_backend="prefect")
        factory = DurableAgentFactory(settings)
        assert factory.backend == "prefect"

    def test_factory_not_initialized_raises(self) -> None:
        """Test factory raises when not initialized."""
        settings = Settings(openai_api_key="sk-test", durable_backend="dbos")
        factory = DurableAgentFactory(settings)
        with pytest.raises(RuntimeError, match="not initialized"):
            factory.get_consolidation_agent()

    def test_factory_dbos_initialization(self) -> None:
        """Test DBOS factory initializes correctly with in-memory SQLite."""
        settings = Settings(
            openai_api_key="sk-test",
            durable_backend="dbos",
            database_url="sqlite:///:memory:",
        )
        factory = DurableAgentFactory(settings)

        # Mock at the import locations to avoid OpenAI client creation
        with patch("dbos.DBOS") as mock_dbos_class:
            with patch("pydantic_ai.durable_exec.dbos.DBOSAgent"):
                with patch("engram.workflows.Agent"):
                    factory.initialize()

                    # DBOS should be configured
                    mock_dbos_class.assert_called_once()
                    mock_dbos_class.launch.assert_called_once()

        assert factory._initialized

    def test_factory_returns_wrapped_agents(self) -> None:
        """Test factory returns DBOS-wrapped agents after initialization."""
        settings = Settings(
            openai_api_key="sk-test",
            durable_backend="dbos",
            database_url="sqlite:///:memory:",
        )
        factory = DurableAgentFactory(settings)

        with patch("dbos.DBOS"):
            with patch("pydantic_ai.durable_exec.dbos.DBOSAgent") as mock_dbos_agent:
                with patch("engram.workflows.Agent"):
                    mock_dbos_agent.return_value = MagicMock()
                    factory.initialize()

                    consolidation = factory.get_consolidation_agent()
                    decay = factory.get_decay_agent()

                    assert consolidation is not None
                    assert decay is not None
                    # Both should be wrapped by DBOSAgent
                    assert mock_dbos_agent.call_count == 2

    def test_init_workflows_global_function(self) -> None:
        """Test init_workflows convenience function."""
        settings = Settings(
            openai_api_key="sk-test",
            durable_backend="dbos",
            database_url="sqlite:///:memory:",
        )

        with patch("dbos.DBOS"):
            with patch("pydantic_ai.durable_exec.dbos.DBOSAgent"):
                with patch("engram.workflows.Agent"):
                    factory = init_workflows(settings)

                    assert factory is not None
                    assert factory._initialized

                    # Second call returns same factory
                    factory2 = init_workflows(settings)
                    assert factory is factory2

    def test_shutdown_workflows_clears_global(self) -> None:
        """Test shutdown_workflows clears global factory."""
        settings = Settings(
            openai_api_key="sk-test",
            durable_backend="dbos",
            database_url="sqlite:///:memory:",
        )

        with patch("dbos.DBOS"):
            with patch("pydantic_ai.durable_exec.dbos.DBOSAgent"):
                with patch("engram.workflows.Agent"):
                    init_workflows(settings)

        shutdown_workflows()

        # Should be able to init again
        with patch("dbos.DBOS"):
            with patch("pydantic_ai.durable_exec.dbos.DBOSAgent"):
                with patch("engram.workflows.Agent"):
                    factory = init_workflows(settings)
                    assert factory._initialized


class TestFindMatchingMemory:
    """Tests for _find_matching_memory function."""

    def test_exact_match(self) -> None:
        """Test exact content match."""
        from engram.models import SemanticMemory

        memory = SemanticMemory(
            content="User prefers Python",
            user_id="test_user",
            embedding=[0.1] * 384,
        )
        memories = {"User prefers Python": memory}

        result = _find_matching_memory("User prefers Python", memories)
        assert result is memory

    def test_normalized_match(self) -> None:
        """Test case-insensitive and whitespace-normalized match."""
        from engram.models import SemanticMemory

        memory = SemanticMemory(
            content="User prefers Python",
            user_id="test_user",
            embedding=[0.1] * 384,
        )
        memories = {"User prefers Python": memory}

        result = _find_matching_memory("  user prefers python  ", memories)
        assert result is memory

    def test_substring_match_content_in_memory(self) -> None:
        """Test content is substring of memory content."""
        from engram.models import SemanticMemory

        memory = SemanticMemory(
            content="User strongly prefers Python for data science",
            user_id="test_user",
            embedding=[0.1] * 384,
        )
        memories = {"User strongly prefers Python for data science": memory}

        result = _find_matching_memory("prefers python", memories)
        assert result is memory

    def test_substring_match_memory_in_content(self) -> None:
        """Test memory content is substring of query content."""
        from engram.models import SemanticMemory

        memory = SemanticMemory(
            content="prefers Python",
            user_id="test_user",
            embedding=[0.1] * 384,
        )
        memories = {"prefers Python": memory}

        result = _find_matching_memory("User prefers Python for work", memories)
        assert result is memory

    def test_no_match_returns_none(self) -> None:
        """Test returns None when no match found."""
        from engram.models import SemanticMemory

        memory = SemanticMemory(
            content="User prefers Python",
            user_id="test_user",
            embedding=[0.1] * 384,
        )
        memories = {"User prefers Python": memory}

        result = _find_matching_memory("totally unrelated content", memories)
        assert result is None

    def test_empty_memories_returns_none(self) -> None:
        """Test returns None for empty memory dict."""
        result = _find_matching_memory("any content", {})
        assert result is None


class TestConsolidationLinking:
    """Tests for memory linking during consolidation."""

    @pytest.mark.asyncio
    async def test_links_created_with_existing_memories(self) -> None:
        """Test that new semantic memories link to similar existing memories."""
        from engram.models import Episode, SemanticMemory

        mock_episode = MagicMock(spec=Episode)
        mock_episode.id = "ep_123"
        mock_episode.role = "user"
        mock_episode.content = "I love Python programming"

        # Existing memory to link with
        existing_memory = SemanticMemory(
            id="sem_existing",
            content="User enjoys coding",
            user_id="test_user",
            embedding=[0.1] * 384,
        )

        mock_storage = AsyncMock()
        mock_storage.get_unsummarized_episodes = AsyncMock(return_value=[mock_episode])
        mock_storage.store_semantic = AsyncMock(return_value="sem_new")
        mock_storage.mark_episodes_summarized = AsyncMock(return_value=1)
        mock_storage.list_semantic_memories = AsyncMock(return_value=[existing_memory])
        mock_storage.search_semantic = AsyncMock(
            return_value=[MagicMock(memory=existing_memory, score=0.85)]
        )
        mock_storage.update_semantic_memory = AsyncMock(return_value=True)

        mock_embedder = AsyncMock()
        mock_embedder.embed = AsyncMock(return_value=[0.1] * 384)

        mock_summary = SummaryOutput(
            summary="The user enjoys Python programming.",
            keywords=["python", "programming"],
            context="technical",
        )

        with patch(
            "engram.workflows.consolidation._summarize_chunk",
            return_value=mock_summary,
        ):
            result = await run_consolidation(
                storage=mock_storage,
                embedder=mock_embedder,
                user_id="test_user",
                org_id="test_org",
            )

        # Should have created memory and link
        assert result.semantic_memories_created == 1
        assert result.links_created == 1

        # update_semantic_memory should have been called for linking
        assert mock_storage.update_semantic_memory.call_count >= 1

    @pytest.mark.asyncio
    async def test_no_links_when_no_similar_memories(self) -> None:
        """Test that no links are created when no similar memories exist."""
        from engram.models import Episode

        mock_episode = MagicMock(spec=Episode)
        mock_episode.id = "ep_789"
        mock_episode.role = "user"
        mock_episode.content = "Test content"

        mock_storage = AsyncMock()
        mock_storage.get_unsummarized_episodes = AsyncMock(return_value=[mock_episode])
        mock_storage.store_semantic = AsyncMock(return_value="sem_test")
        mock_storage.mark_episodes_summarized = AsyncMock(return_value=1)
        mock_storage.list_semantic_memories = AsyncMock(return_value=[])  # No existing
        mock_storage.search_semantic = AsyncMock(return_value=[])  # No similar

        mock_embedder = AsyncMock()
        mock_embedder.embed = AsyncMock(return_value=[0.1] * 384)

        mock_summary = SummaryOutput(
            summary="Test summary.",
            keywords=["test"],
            context="general",
        )

        with patch(
            "engram.workflows.consolidation._summarize_chunk",
            return_value=mock_summary,
        ):
            result = await run_consolidation(
                storage=mock_storage,
                embedder=mock_embedder,
                user_id="test_user",
                org_id="test_org",
            )

        # Memory created but no links
        assert result.semantic_memories_created == 1
        assert result.links_created == 0


class TestConsolidationStrengthening:
    """Tests for memory strengthening during consolidation (Testing Effect)."""

    @pytest.mark.asyncio
    async def test_existing_memory_strengthened_on_link(self) -> None:
        """Test existing memory is strengthened when linked to new memory."""
        from engram.models import Episode, SemanticMemory

        mock_episode = MagicMock(spec=Episode)
        mock_episode.id = "ep_sel_001"
        mock_episode.role = "user"
        mock_episode.content = "I prefer Python programming"

        # Existing memory with initial strength 0.0
        existing_memory = SemanticMemory(
            id="sem_existing",
            content="User likes programming",
            user_id="test_user",
            embedding=[0.1] * 384,
        )
        assert existing_memory.consolidation_strength == 0.0

        mock_storage = AsyncMock()
        mock_storage.get_unsummarized_episodes = AsyncMock(return_value=[mock_episode])
        mock_storage.store_semantic = AsyncMock(return_value="sem_new")
        mock_storage.mark_episodes_summarized = AsyncMock(return_value=1)
        mock_storage.list_semantic_memories = AsyncMock(return_value=[existing_memory])
        mock_storage.search_semantic = AsyncMock(
            return_value=[MagicMock(memory=existing_memory, score=0.85)]
        )
        mock_storage.update_semantic_memory = AsyncMock(return_value=True)

        mock_embedder = AsyncMock()
        mock_embedder.embed = AsyncMock(return_value=[0.1] * 384)

        mock_summary = SummaryOutput(
            summary="User prefers Python.",
            keywords=["python"],
            context="programming",
        )

        with patch(
            "engram.workflows.consolidation._summarize_chunk",
            return_value=mock_summary,
        ):
            result = await run_consolidation(
                storage=mock_storage,
                embedder=mock_embedder,
                user_id="test_user",
                org_id="test_org",
            )

        # Should have created memory and strengthened existing
        assert result.semantic_memories_created == 1
        assert result.links_created == 1

        # Existing memory should have increased strength
        assert existing_memory.consolidation_strength == 0.1  # Increased by 0.1
        assert existing_memory.consolidation_passes == 1
