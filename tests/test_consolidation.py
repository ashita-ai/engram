"""Tests for consolidation workflow."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from engram.config import Settings
from engram.workflows import DurableAgentFactory, init_workflows, shutdown_workflows
from engram.workflows.consolidation import (
    CheckpointStore,
    ConsolidationResult,
    ExtractedFact,
    IdentifiedLink,
    LLMExtractionResult,
    MapReduceSummary,
    MemoryEvolution,
    SummaryOutput,
    _check_for_near_duplicate,
    _find_matching_memory,
    _format_existing_memories_for_llm,
    _is_profile_style,
    _needs_quality_retry,
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
        output = SummaryOutput(summary="Python is the primary programming language.")
        assert output.summary == "Python is the primary programming language."
        assert output.key_facts == []
        assert output.keywords == []
        assert output.context == ""

    def test_create_with_all_fields(self) -> None:
        """Test creating summary output with all fields."""
        output = SummaryOutput(
            summary="Python is the primary language at TechCorp. The backend runs Django 4.2.",
            key_facts=[
                "TechCorp uses Python for backend services",
                "Django 4.2 is the web framework",
            ],
            keywords=["python", "django", "techcorp"],
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
            summary="PostgreSQL 15 is the primary database. Redis handles caching.",
            keywords=["postgresql", "redis"],
            context="infrastructure",
        )
        assert summary.summary == "PostgreSQL 15 is the primary database. Redis handles caching."
        assert summary.keywords == ["postgresql", "redis"]
        assert summary.context == "infrastructure"


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
        mock_storage.search_semantic = AsyncMock(return_value=[])

        mock_embedder = AsyncMock()
        mock_embedder.embed = AsyncMock(return_value=[0.1] * 384)

        # Mock the LLM summarization
        mock_summary = SummaryOutput(
            summary="Three messages were exchanged covering general topics.",
            keywords=["messages", "conversation"],
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
        mock_storage.search_semantic = AsyncMock(return_value=[])

        mock_embedder = AsyncMock()
        mock_embedder.embed = AsyncMock(return_value=[0.1] * 384)

        mock_summary = SummaryOutput(
            summary="A greeting was exchanged.",
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

    @pytest.mark.asyncio
    async def test_source_episode_ids_excludes_system_prompts(self) -> None:
        """System prompts should NOT appear in source_episode_ids provenance.

        Regression test for bug #7: system episodes were included in
        source_episode_ids, polluting verify() traces with irrelevant
        system prompts that didn't contribute content.
        """
        from engram.models import Episode

        user_episode = MagicMock(spec=Episode)
        user_episode.id = "ep_user"
        user_episode.role = "user"
        user_episode.content = "I prefer dark mode."

        system_episode = MagicMock(spec=Episode)
        system_episode.id = "ep_system"
        system_episode.role = "system"
        system_episode.content = "You are a helpful assistant."

        stored_memory = None

        async def capture_store(memory: object) -> str:
            nonlocal stored_memory
            stored_memory = memory
            return "sem_123"

        mock_storage = AsyncMock()
        mock_storage.get_unsummarized_episodes = AsyncMock(
            return_value=[user_episode, system_episode]
        )
        mock_storage.store_semantic = AsyncMock(side_effect=capture_store)
        mock_storage.mark_episodes_summarized = AsyncMock(return_value=2)
        mock_storage.list_semantic_memories = AsyncMock(return_value=[])
        mock_storage.search_semantic = AsyncMock(return_value=[])

        mock_embedder = AsyncMock()
        mock_embedder.embed = AsyncMock(return_value=[0.1] * 384)

        mock_summary = SummaryOutput(
            summary="User prefers dark mode.",
            keywords=["dark mode", "preference"],
            context="settings",
        )

        with patch(
            "engram.workflows.consolidation._summarize_chunk",
            return_value=mock_summary,
        ):
            await run_consolidation(
                storage=mock_storage,
                embedder=mock_embedder,
                user_id="test_user",
                org_id="test_org",
            )

        # source_episode_ids should ONLY contain user episodes
        assert stored_memory is not None
        assert "ep_user" in stored_memory.source_episode_ids
        assert "ep_system" not in stored_memory.source_episode_ids

        # But ALL episodes (including system) should be marked as summarized
        mark_args = mock_storage.mark_episodes_summarized.call_args[0][0]
        assert "ep_user" in mark_args
        assert "ep_system" in mark_args


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
            summary="Python is used for programming tasks.",
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
            summary="Test content was discussed.",
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
            summary="Python is the preferred programming language.",
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


class TestProfileDetection:
    """Tests for _is_profile_style and profile filtering."""

    def test_detects_personality_profile(self) -> None:
        """Classic personality profile content should be detected."""
        content = (
            "Evan demonstrates a methodical approach to software engineering "
            "and is consistently engaged in improving code quality."
        )
        assert _is_profile_style(content) is True

    def test_detects_behavioral_description(self) -> None:
        """Behavioral description in third person should be detected."""
        content = (
            "The user exhibits strong attention to detail and demonstrates "
            "a systematic approach to debugging."
        )
        assert _is_profile_style(content) is True

    def test_allows_factual_content(self) -> None:
        """Factual technical content should NOT be flagged."""
        content = (
            "PostgreSQL 16 is the primary database. "
            "Redis was removed in v2.3 due to connection pooling issues."
        )
        assert _is_profile_style(content) is False

    def test_allows_actionable_knowledge(self) -> None:
        """Actionable technical knowledge should NOT be flagged."""
        content = (
            "OrderedDict LRU cache needs asyncio.Lock to prevent coroutine races. "
            "ruff check must pass before committing."
        )
        assert _is_profile_style(content) is False

    def test_requires_two_matches(self) -> None:
        """Single pattern match should NOT trigger profile detection."""
        # Only one pattern: "demonstrates"
        content = "The code demonstrates proper error handling with retries."
        assert _is_profile_style(content) is False

    def test_format_filters_profiles(self) -> None:
        """_format_existing_memories_for_llm should filter out profile-style memories."""
        from engram.models import SemanticMemory

        profile_mem = SemanticMemory(
            content=(
                "Evan demonstrates a methodical approach and is consistently engaged "
                "in reviewing code."
            ),
            user_id="test_user",
            embedding=[0.1] * 384,
        )
        factual_mem = SemanticMemory(
            content="PostgreSQL 16 is the primary database.",
            user_id="test_user",
            embedding=[0.2] * 384,
        )

        result = _format_existing_memories_for_llm([profile_mem, factual_mem])

        assert "PostgreSQL 16" in result
        assert "methodical approach" not in result

    def test_format_handles_all_profiles(self) -> None:
        """When all memories are profiles, format returns empty string."""
        from engram.models import SemanticMemory

        profile_mem = SemanticMemory(
            content=(
                "The user demonstrates attention to detail and exhibits "
                "a methodical approach to problem-solving."
            ),
            user_id="test_user",
            embedding=[0.1] * 384,
        )

        result = _format_existing_memories_for_llm([profile_mem])
        assert result == ""

    def test_format_respects_config_limit(self) -> None:
        """Format should limit to consolidation_max_context_memories."""
        from engram.models import SemanticMemory

        memories = [
            SemanticMemory(
                content=f"Fact number {i} about the system.",
                user_id="test_user",
                embedding=[0.1] * 384,
            )
            for i in range(50)
        ]

        result = _format_existing_memories_for_llm(memories)
        # Default limit is 25
        assert result.count("Fact number") == 25


class TestDeduplication:
    """Tests for embedding-based deduplication."""

    @pytest.mark.asyncio
    async def test_duplicate_detected_above_threshold(self) -> None:
        """Near-duplicate above threshold should be returned."""
        from engram.models import SemanticMemory

        existing = SemanticMemory(
            id="sem_existing",
            content="PostgreSQL 16 is the primary database.",
            user_id="test_user",
            embedding=[0.1] * 384,
        )

        mock_storage = AsyncMock()
        mock_storage.search_semantic = AsyncMock(
            return_value=[MagicMock(memory=existing, score=0.92)]
        )

        result = await _check_for_near_duplicate(
            embedding=[0.1] * 384,
            storage=mock_storage,
            user_id="test_user",
            org_id="test_org",
        )

        assert result is existing

    @pytest.mark.asyncio
    async def test_no_duplicate_below_threshold(self) -> None:
        """Memories below threshold should not be considered duplicates."""
        from engram.models import SemanticMemory

        existing = SemanticMemory(
            id="sem_existing",
            content="Redis handles caching.",
            user_id="test_user",
            embedding=[0.1] * 384,
        )

        mock_storage = AsyncMock()
        mock_storage.search_semantic = AsyncMock(
            return_value=[MagicMock(memory=existing, score=0.85)]
        )

        result = await _check_for_near_duplicate(
            embedding=[0.1] * 384,
            storage=mock_storage,
            user_id="test_user",
            org_id="test_org",
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_no_duplicate_empty_results(self) -> None:
        """Empty search results should return None."""
        mock_storage = AsyncMock()
        mock_storage.search_semantic = AsyncMock(return_value=[])

        result = await _check_for_near_duplicate(
            embedding=[0.1] * 384,
            storage=mock_storage,
            user_id="test_user",
            org_id="test_org",
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_consolidation_merges_on_duplicate(self) -> None:
        """run_consolidation should merge into existing when duplicate is found."""
        from engram.models import Episode, SemanticMemory

        mock_episode = MagicMock(spec=Episode)
        mock_episode.id = "ep_new"
        mock_episode.role = "user"
        mock_episode.content = "PostgreSQL 16 is great"

        existing = SemanticMemory(
            id="sem_existing",
            content="- PostgreSQL 16 is the primary database",
            user_id="test_user",
            embedding=[0.1] * 384,
            source_episode_ids=["ep_old"],
            keywords=["postgresql"],
        )

        mock_storage = AsyncMock()
        mock_storage.get_unsummarized_episodes = AsyncMock(return_value=[mock_episode])
        mock_storage.list_semantic_memories = AsyncMock(return_value=[])
        mock_storage.search_semantic = AsyncMock(
            return_value=[MagicMock(memory=existing, score=0.95)]
        )
        mock_storage.update_semantic_memory = AsyncMock(return_value=True)
        mock_storage.mark_episodes_summarized = AsyncMock(return_value=1)
        mock_storage.store_semantic = AsyncMock()

        mock_embedder = AsyncMock()
        mock_embedder.embed = AsyncMock(return_value=[0.1] * 384)

        mock_summary = SummaryOutput(
            summary="PostgreSQL 16 is the primary database.",
            key_facts=["PostgreSQL 16 is the primary database"],
            keywords=["postgresql", "database"],
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

        # Should NOT have created a new memory
        mock_storage.store_semantic.assert_not_called()
        # Should have updated the existing one
        mock_storage.update_semantic_memory.assert_called()
        # Source IDs should be merged
        assert "ep_new" in existing.source_episode_ids
        assert "ep_old" in existing.source_episode_ids
        # Keywords should be merged
        assert "database" in existing.keywords
        # Strength should have increased
        assert existing.consolidation_strength == 0.1
        # Episodes processed but no new memories created
        assert result.episodes_processed == 1
        assert result.semantic_memories_created == 0

    @pytest.mark.asyncio
    async def test_consolidation_creates_when_no_duplicate(self) -> None:
        """run_consolidation should create normally when no duplicate exists."""
        from engram.models import Episode

        mock_episode = MagicMock(spec=Episode)
        mock_episode.id = "ep_unique"
        mock_episode.role = "user"
        mock_episode.content = "Something completely new"

        mock_storage = AsyncMock()
        mock_storage.get_unsummarized_episodes = AsyncMock(return_value=[mock_episode])
        mock_storage.list_semantic_memories = AsyncMock(return_value=[])
        mock_storage.search_semantic = AsyncMock(return_value=[])  # No duplicates
        mock_storage.store_semantic = AsyncMock(return_value="sem_new")
        mock_storage.mark_episodes_summarized = AsyncMock(return_value=1)

        mock_embedder = AsyncMock()
        mock_embedder.embed = AsyncMock(return_value=[0.1] * 384)

        mock_summary = SummaryOutput(
            summary="Something completely new was discussed.",
            key_facts=["Something completely new"],
            keywords=["new"],
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

        # Should have created a new memory
        mock_storage.store_semantic.assert_called_once()
        assert result.semantic_memories_created == 1


class TestDynamicFactCaps:
    """Tests for dynamic consolidation output caps (#209)."""

    def test_fact_cap_scales_with_episode_count(self) -> None:
        """Dynamic fact cap should be max(8, episodes * facts_per_episode), capped at 50."""
        from engram.config import settings

        # With 1 episode: max(8, 1*3) = 8
        assert max(8, 1 * settings.consolidation_max_facts_per_episode) == 8
        # With 5 episodes: max(8, 5*3) = 15
        assert max(8, 5 * settings.consolidation_max_facts_per_episode) == 15
        # With 20 episodes: min(50, max(8, 20*3)) = 50 (capped)
        assert min(50, max(8, 20 * settings.consolidation_max_facts_per_episode)) == 50

    def test_keyword_limit_from_config(self) -> None:
        """Keywords should use consolidation_max_keywords config (not hardcoded 15)."""
        from engram.config import settings

        assert settings.consolidation_max_keywords == 30
        assert settings.consolidation_max_keywords > 15  # Was previously hardcoded

    def test_context_memory_limit_from_config(self) -> None:
        """Existing context should use consolidation_max_context_memories config (not hardcoded 10)."""
        from engram.config import settings

        assert settings.consolidation_max_context_memories == 25
        assert settings.consolidation_max_context_memories > 10  # Was previously hardcoded


class TestConsolidationConfidenceCap:
    """Tests that consolidation creates semantic memories with capped confidence (#215)."""

    @pytest.mark.asyncio
    async def test_created_semantic_memory_uses_inferred_confidence(self) -> None:
        """Semantic memories created by consolidation should use for_inferred() (capped at 0.6)."""
        from engram.models import Episode

        mock_episode = MagicMock(spec=Episode)
        mock_episode.id = "ep_conf"
        mock_episode.role = "user"
        mock_episode.content = "PostgreSQL 16 is the primary database"

        mock_storage = AsyncMock()
        mock_storage.get_unsummarized_episodes = AsyncMock(return_value=[mock_episode])
        mock_storage.list_semantic_memories = AsyncMock(return_value=[])
        mock_storage.search_semantic = AsyncMock(return_value=[])
        mock_storage.store_semantic = AsyncMock(return_value="sem_conf")
        mock_storage.mark_episodes_summarized = AsyncMock(return_value=1)

        mock_embedder = AsyncMock()
        mock_embedder.embed = AsyncMock(return_value=[0.1] * 384)

        # LLM returns high confidence (0.85) — should be capped
        mock_summary = SummaryOutput(
            summary="PostgreSQL 16 is the primary database.",
            key_facts=["PostgreSQL 16 is the primary database"],
            keywords=["postgresql"],
            confidence=0.85,
            confidence_reasoning="Strong source agreement",
        )

        with patch(
            "engram.workflows.consolidation._summarize_chunk",
            return_value=mock_summary,
        ):
            await run_consolidation(
                storage=mock_storage,
                embedder=mock_embedder,
                user_id="test_user",
                org_id="test_org",
            )

        # Verify the stored semantic memory has capped confidence
        stored_memory = mock_storage.store_semantic.call_args[0][0]
        assert (
            stored_memory.confidence.value <= 0.6
        ), f"Inferred confidence should be capped at 0.6, got {stored_memory.confidence.value}"
        assert stored_memory.confidence.extraction_base == 0.6


# ---------------------------------------------------------------------------
# CheckpointStore
# ---------------------------------------------------------------------------


class TestCheckpointStore:
    """Tests for consolidation checkpoint persistence."""

    @pytest.fixture
    def ckpt_dir(self, tmp_path: object) -> str:
        """Return a temporary directory path for checkpoints."""
        from pathlib import Path

        return str(Path(str(tmp_path)) / "ckpts")

    def test_run_key_deterministic(self) -> None:
        """Same inputs always produce the same run key."""
        ids = ["ep_3", "ep_1", "ep_2"]
        key1 = CheckpointStore.run_key("u1", "org1", ids)
        key2 = CheckpointStore.run_key("u1", "org1", list(reversed(ids)))
        assert key1 == key2  # sorted internally

    def test_run_key_varies_with_inputs(self) -> None:
        """Different inputs produce different run keys."""
        key_a = CheckpointStore.run_key("u1", "org1", ["ep_1"])
        key_b = CheckpointStore.run_key("u1", "org2", ["ep_1"])
        assert key_a != key_b

    def test_save_and_load_chunk(self, ckpt_dir: str) -> None:
        """Saved chunk round-trips through JSON correctly."""
        store = CheckpointStore(ckpt_dir)
        summary = SummaryOutput(
            summary="test",
            key_facts=["fact1"],
            keywords=["kw"],
            context="ctx",
            confidence=0.5,
            confidence_reasoning="ok",
        )
        key = "test_run_1"
        store.save_chunk(key, 0, summary)
        loaded = store.load(key)

        assert 0 in loaded
        restored = SummaryOutput.model_validate(loaded[0])
        assert restored.summary == "test"
        assert restored.key_facts == ["fact1"]
        assert restored.confidence == 0.5

    def test_save_multiple_chunks(self, ckpt_dir: str) -> None:
        """Multiple chunks accumulate in one checkpoint file."""
        store = CheckpointStore(ckpt_dir)
        key = "multi_chunk"

        for i in range(3):
            store.save_chunk(
                key,
                i,
                SummaryOutput(summary=f"chunk_{i}", confidence=0.5),
            )

        loaded = store.load(key)
        assert len(loaded) == 3
        assert SummaryOutput.model_validate(loaded[2]).summary == "chunk_2"

    def test_load_nonexistent_returns_empty(self, ckpt_dir: str) -> None:
        """Loading a missing checkpoint returns an empty dict."""
        store = CheckpointStore(ckpt_dir)
        assert store.load("does_not_exist") == {}

    def test_clear_deletes_checkpoint(self, ckpt_dir: str) -> None:
        """Clear removes the checkpoint file."""
        store = CheckpointStore(ckpt_dir)
        key = "to_clear"
        store.save_chunk(key, 0, SummaryOutput(summary="x", confidence=0.5))
        assert store.load(key) != {}

        store.clear(key)
        assert store.load(key) == {}

    def test_clear_missing_is_noop(self, ckpt_dir: str) -> None:
        """Clearing a nonexistent checkpoint doesn't raise."""
        store = CheckpointStore(ckpt_dir)
        store.clear("nonexistent")  # should not raise

    def test_corrupt_checkpoint_handled_gracefully(self, ckpt_dir: str) -> None:
        """Corrupt JSON is treated as missing checkpoint."""

        store = CheckpointStore(ckpt_dir)
        key = "corrupt"
        path = store._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("not valid json {{{")

        loaded = store.load(key)
        assert loaded == {}
        # Corrupt file should be cleaned up
        assert not path.exists()

    def test_creates_directory_if_needed(self, tmp_path: object) -> None:
        """CheckpointStore creates the directory tree on construction."""
        from pathlib import Path

        deep_path = Path(str(tmp_path)) / "a" / "b" / "c"
        store = CheckpointStore(str(deep_path))
        store.save_chunk("k", 0, SummaryOutput(summary="hi", confidence=0.5))
        assert store.load("k") != {}


class TestCheckpointIntegration:
    """Tests that consolidation actually uses checkpoints when configured."""

    @pytest.mark.asyncio
    async def test_run_consolidation_uses_checkpoint(self, tmp_path: object) -> None:
        """Consolidation saves chunks to checkpoint and clears on success."""
        from pathlib import Path

        ckpt_dir = str(Path(str(tmp_path)) / "ckpts")

        mock_storage = MagicMock()
        mock_storage.get_unsummarized_episodes = AsyncMock(return_value=[])
        mock_embedder = MagicMock()

        # Pre-create a checkpoint that simulates a previous partial run
        store = CheckpointStore(ckpt_dir)
        summary = SummaryOutput(
            summary="cached summary",
            key_facts=["cached fact"],
            keywords=["cached"],
            confidence=0.5,
        )

        # Build the episode IDs that will be "fetched"
        from engram.models import Episode

        episodes = [
            Episode(
                id=f"ep_{i}",
                content=f"Content {i}",
                role="user",
                user_id="u1",
                org_id="org1",
                embedding=[0.1] * 3,
            )
            for i in range(5)
        ]
        ep_ids = [ep.id for ep in episodes]
        run_key = CheckpointStore.run_key("u1", "org1", ep_ids)

        # Save a checkpoint for chunk 0 (simulating a crash after chunk 0 completed)
        store.save_chunk(run_key, 0, summary)

        # Configure storage to return our episodes
        mock_storage.get_unsummarized_episodes = AsyncMock(return_value=episodes)
        mock_storage.list_semantic_memories = AsyncMock(return_value=[])
        mock_storage.store_semantic = AsyncMock()
        mock_storage.update_semantic_memory = AsyncMock()
        mock_storage.mark_episodes_summarized = AsyncMock()
        mock_storage.search_semantic = AsyncMock(return_value=[])
        mock_embedder.embed = AsyncMock(return_value=[0.1] * 3)

        # Mock the LLM call — it should only be called for chunk 0 since
        # all 5 episodes fit in one chunk (< MAX_EPISODES_PER_CHUNK=20)
        mock_summary = SummaryOutput(
            summary="fresh summary",
            key_facts=["fresh fact"],
            keywords=["kw"],
            confidence=0.5,
        )

        with (
            patch(
                "engram.workflows.consolidation._summarize_chunk",
                new_callable=AsyncMock,
                return_value=mock_summary,
            ),
            patch(
                "engram.config.settings.consolidation_checkpoint_dir",
                ckpt_dir,
            ),
            patch(
                "engram.workflows.consolidation._check_for_near_duplicate",
                new_callable=AsyncMock,
                return_value=None,
            ),
        ):
            result = await run_consolidation(
                storage=mock_storage,
                embedder=mock_embedder,
                user_id="u1",
                org_id="org1",
            )

        assert result.episodes_processed == 5
        assert result.semantic_memories_created == 1

        # Checkpoint should be cleaned up after successful completion
        assert store.load(run_key) == {}

    @pytest.mark.asyncio
    async def test_checkpoint_resumes_cached_chunks(self, tmp_path: object) -> None:
        """When a checkpoint exists, cached chunks skip the LLM call."""
        from pathlib import Path

        from engram.workflows.consolidation import MAX_EPISODES_PER_CHUNK

        ckpt_dir = str(Path(str(tmp_path)) / "ckpts")

        mock_storage = MagicMock()
        mock_embedder = MagicMock()
        mock_embedder.embed = AsyncMock(return_value=[0.1] * 3)

        # Create enough episodes for 2 chunks
        from engram.models import Episode

        episode_count = MAX_EPISODES_PER_CHUNK + 5
        episodes = [
            Episode(
                id=f"ep_{i}",
                content=f"Content {i}",
                role="user",
                user_id="u1",
                org_id="org1",
                embedding=[0.1] * 3,
            )
            for i in range(episode_count)
        ]

        ep_ids = [ep.id for ep in episodes]
        run_key = CheckpointStore.run_key("u1", "org1", ep_ids)

        # Pre-save chunk 0 as if it completed in a prior run
        store = CheckpointStore(ckpt_dir)
        cached_summary = SummaryOutput(
            summary="cached from prior run",
            key_facts=["prior fact"],
            keywords=["cached"],
            confidence=0.5,
        )
        store.save_chunk(run_key, 0, cached_summary)

        # The LLM should only be called for chunk 1 (chunk 0 is cached)
        fresh_summary = SummaryOutput(
            summary="fresh",
            key_facts=["fresh fact"],
            keywords=["fresh"],
            confidence=0.5,
        )

        mock_storage.get_unsummarized_episodes = AsyncMock(return_value=episodes)
        mock_storage.list_semantic_memories = AsyncMock(return_value=[])
        mock_storage.store_semantic = AsyncMock()
        mock_storage.update_semantic_memory = AsyncMock()
        mock_storage.mark_episodes_summarized = AsyncMock()
        mock_storage.search_semantic = AsyncMock(return_value=[])

        reduced_summary = MapReduceSummary(
            summary="reduced",
            key_facts=["reduced fact"],
            keywords=["kw"],
            confidence=0.5,
        )

        with (
            patch(
                "engram.workflows.consolidation._summarize_chunk",
                new_callable=AsyncMock,
                return_value=fresh_summary,
            ) as mock_summarize,
            patch(
                "engram.workflows.consolidation._reduce_summaries",
                new_callable=AsyncMock,
                return_value=reduced_summary,
            ),
            patch(
                "engram.config.settings.consolidation_checkpoint_dir",
                ckpt_dir,
            ),
            patch(
                "engram.workflows.consolidation._check_for_near_duplicate",
                new_callable=AsyncMock,
                return_value=None,
            ),
        ):
            result = await run_consolidation(
                storage=mock_storage,
                embedder=mock_embedder,
                user_id="u1",
                org_id="org1",
            )

        # _summarize_chunk should only be called once (for chunk 1)
        # chunk 0 was loaded from checkpoint
        assert mock_summarize.call_count == 1
        assert result.episodes_processed == episode_count


class TestNeedsQualityRetry:
    """Unit tests for _needs_quality_retry detection."""

    def test_empty_key_facts_with_episodes(self) -> None:
        """Empty key_facts when episodes exist should trigger retry."""
        output = SummaryOutput(
            summary="Some summary.",
            key_facts=[],
            confidence=0.5,
        )
        assert _needs_quality_retry(output, item_count=3) is True

    def test_empty_key_facts_with_zero_episodes(self) -> None:
        """Empty key_facts with zero episodes should NOT trigger retry."""
        output = SummaryOutput(
            summary="No episodes to summarize.",
            key_facts=[],
            confidence=0.5,
        )
        assert _needs_quality_retry(output, item_count=0) is False

    def test_profile_style_summary(self) -> None:
        """Profile-style summary should trigger retry even with facts."""
        output = SummaryOutput(
            summary=(
                "Evan demonstrates a methodical approach to software engineering "
                "and is consistently engaged in improving code quality."
            ),
            key_facts=["ruff check must pass before committing."],
            confidence=0.5,
        )
        assert _needs_quality_retry(output, item_count=1) is True

    def test_profile_style_key_fact(self) -> None:
        """A profile-style key fact should trigger retry."""
        output = SummaryOutput(
            summary="Technical setup details.",
            key_facts=[
                "PostgreSQL 16 is the primary database.",
                (
                    "The user exhibits strong attention to detail and demonstrates "
                    "a systematic approach to debugging."
                ),
            ],
            confidence=0.5,
        )
        assert _needs_quality_retry(output, item_count=2) is True

    def test_good_output_no_retry(self) -> None:
        """Factual output with key_facts should NOT trigger retry."""
        output = SummaryOutput(
            summary="PostgreSQL 16 is the primary database. Redis handles caching.",
            key_facts=[
                "PostgreSQL 16 is the primary database.",
                "Redis was removed in v2.3 due to connection pooling issues.",
            ],
            keywords=["postgresql", "redis"],
            confidence=0.5,
        )
        assert _needs_quality_retry(output, item_count=3) is False

    def test_single_profile_pattern_not_enough(self) -> None:
        """A single profile-like word in a fact should NOT trigger retry.

        _is_profile_style requires 2+ pattern matches.
        """
        output = SummaryOutput(
            summary="The code demonstrates proper error handling.",
            key_facts=["Error handling uses retries with exponential backoff."],
            confidence=0.5,
        )
        assert _needs_quality_retry(output, item_count=1) is False


class TestQualityRetryIntegration:
    """Integration tests for quality retry in _summarize_chunk and _synthesize_structured."""

    @pytest.mark.asyncio
    async def test_retry_on_empty_key_facts(self) -> None:
        """_summarize_chunk should retry when LLM returns empty key_facts."""
        from engram.workflows.consolidation import _summarize_chunk

        episodes = [{"id": "ep_1", "role": "user", "content": "PostgreSQL 16 is great."}]

        bad_output = SummaryOutput(
            summary="A database discussion occurred.",
            key_facts=[],  # Empty — should trigger retry
            confidence=0.5,
        )
        good_output = SummaryOutput(
            summary="PostgreSQL 16 is the primary database.",
            key_facts=["PostgreSQL 16 is the primary database."],
            confidence=0.5,
        )

        call_count = 0

        async def mock_run_agent(agent: object, prompt: str, **kwargs: object) -> SummaryOutput:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return bad_output
            return good_output

        with patch(
            "engram.workflows.llm_utils.run_agent_with_retry",
            side_effect=mock_run_agent,
        ):
            result = await _summarize_chunk(episodes, "")

        assert call_count == 2, "Should have retried once"
        assert result.key_facts == ["PostgreSQL 16 is the primary database."]

    @pytest.mark.asyncio
    async def test_retry_on_profile_summary(self) -> None:
        """_summarize_chunk should retry when LLM returns profile-style summary."""
        from engram.workflows.consolidation import _summarize_chunk

        episodes = [{"id": "ep_1", "role": "user", "content": "I use ruff for linting."}]

        profile_output = SummaryOutput(
            summary=(
                "Evan demonstrates a methodical approach to software engineering "
                "and is consistently engaged in maintaining code quality."
            ),
            key_facts=["ruff is used for linting."],
            confidence=0.5,
        )
        good_output = SummaryOutput(
            summary="ruff is the linting tool. mypy strict mode is enforced.",
            key_facts=["ruff is used for linting."],
            confidence=0.5,
        )

        call_count = 0

        async def mock_run_agent(agent: object, prompt: str, **kwargs: object) -> SummaryOutput:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return profile_output
            return good_output

        with patch(
            "engram.workflows.llm_utils.run_agent_with_retry",
            side_effect=mock_run_agent,
        ):
            result = await _summarize_chunk(episodes, "")

        assert call_count == 2, "Should have retried once"
        assert "methodical approach" not in result.summary

    @pytest.mark.asyncio
    async def test_no_retry_on_good_output(self) -> None:
        """_summarize_chunk should NOT retry when LLM output is acceptable."""
        from engram.workflows.consolidation import _summarize_chunk

        episodes = [{"id": "ep_1", "role": "user", "content": "PostgreSQL 16 is primary."}]

        good_output = SummaryOutput(
            summary="PostgreSQL 16 is the primary database.",
            key_facts=["PostgreSQL 16 is the primary database."],
            keywords=["postgresql"],
            confidence=0.5,
        )

        call_count = 0

        async def mock_run_agent(agent: object, prompt: str, **kwargs: object) -> SummaryOutput:
            nonlocal call_count
            call_count += 1
            return good_output

        with patch(
            "engram.workflows.llm_utils.run_agent_with_retry",
            side_effect=mock_run_agent,
        ):
            result = await _summarize_chunk(episodes, "")

        assert call_count == 1, "Should NOT have retried"
        assert result.key_facts == ["PostgreSQL 16 is the primary database."]

    @pytest.mark.asyncio
    async def test_synthesize_structured_retries_on_profile(self) -> None:
        """_synthesize_structured should retry on profile-style output."""
        from engram.workflows.consolidation import _synthesize_structured

        struct_data: list[dict[str, str | list[str]]] = [
            {
                "id": "struct_1",
                "source_episode_id": "ep_1",
                "summary": "Uses Airflow for pipeline orchestration.",
                "keywords": ["airflow"],
            }
        ]

        profile_output = SummaryOutput(
            summary=(
                "The user exhibits a systematic approach to data engineering "
                "and demonstrates attention to detail in pipeline design."
            ),
            key_facts=[],
            confidence=0.5,
        )
        good_output = SummaryOutput(
            summary="Airflow orchestrates the data pipeline with PostgreSQL metadata store.",
            key_facts=["Airflow is the pipeline orchestrator."],
            confidence=0.5,
        )

        call_count = 0

        async def mock_run_agent(agent: object, prompt: str, **kwargs: object) -> SummaryOutput:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return profile_output
            return good_output

        with patch(
            "engram.workflows.llm_utils.run_agent_with_retry",
            side_effect=mock_run_agent,
        ):
            result = await _synthesize_structured(struct_data, "")

        assert call_count == 2, "Should have retried once"
        assert result.key_facts == ["Airflow is the pipeline orchestrator."]
