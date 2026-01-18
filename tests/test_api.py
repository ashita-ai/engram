"""Tests for Engram REST API."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from engram.api.router import router, set_service
from engram.api.schemas import (
    EncodeRequest,
    HealthResponse,
    RecallRequest,
)
from engram.models import Episode, StructuredMemory
from engram.service import EncodeResult, EngramService, RecallResult


@pytest.fixture
def mock_service():
    """Create a mock EngramService."""
    service = MagicMock(spec=EngramService)
    service.encode = AsyncMock()
    service.recall = AsyncMock()
    # Add storage mock for delete endpoints
    service.storage = MagicMock()
    # Add embedder and settings for workflow endpoints
    service.embedder = MagicMock()
    service.settings = MagicMock()
    # Add workflow_backend for workflow endpoints
    service.workflow_backend = MagicMock()
    return service


@pytest.fixture
def test_app(mock_service):
    """Create a test FastAPI app with mocked service."""
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    set_service(mock_service)
    yield app
    set_service(None)  # type: ignore[arg-type]


@pytest.fixture
def client(test_app):
    """Create a test client."""
    return TestClient(test_app)


class TestHealthEndpoint:
    """Tests for /health endpoint."""

    def test_health_when_service_initialized(self, client, mock_service):
        """Should return healthy when service is ready."""
        response = client.get("/api/v1/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["storage_connected"] is True
        assert "version" in data

    def test_health_when_service_not_initialized(self):
        """Should return unhealthy when service not ready."""
        app = FastAPI()
        app.include_router(router, prefix="/api/v1")
        set_service(None)  # type: ignore[arg-type]
        test_client = TestClient(app)

        response = test_client.get("/api/v1/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "unhealthy"
        assert data["storage_connected"] is False


class TestEncodeEndpoint:
    """Tests for /encode endpoint."""

    def test_encode_success(self, client, mock_service):
        """Should encode content and return episode with structured extracts."""
        mock_episode = Episode(
            content="Email me at user@example.com",
            role="user",
            user_id="user_123",
            embedding=[0.1, 0.2, 0.3],
        )
        mock_structured = StructuredMemory(
            source_episode_id=mock_episode.id,
            mode="fast",
            user_id="user_123",
            emails=["user@example.com"],
        )
        mock_service.encode.return_value = EncodeResult(
            episode=mock_episode, structured=mock_structured
        )

        response = client.post(
            "/api/v1/encode",
            json={
                "content": "Email me at user@example.com",
                "role": "user",
                "user_id": "user_123",
            },
        )

        assert response.status_code == 201
        data = response.json()
        assert data["episode"]["content"] == "Email me at user@example.com"
        assert data["extract_count"] == 1
        assert data["structured"]["emails"] == ["user@example.com"]

    def test_encode_with_all_options(self, client, mock_service):
        """Should accept all optional parameters."""
        mock_episode = Episode(
            content="Hello",
            role="assistant",
            user_id="user_123",
            org_id="org_456",
            session_id="session_789",
            importance=0.8,
            embedding=[0.1],
        )
        mock_structured = StructuredMemory(
            source_episode_id=mock_episode.id,
            mode="fast",
            user_id="user_123",
        )
        mock_service.encode.return_value = EncodeResult(
            episode=mock_episode, structured=mock_structured
        )

        response = client.post(
            "/api/v1/encode",
            json={
                "content": "Hello",
                "role": "assistant",
                "user_id": "user_123",
                "org_id": "org_456",
                "session_id": "session_789",
                "importance": 0.8,
                "enrich": False,
            },
        )

        assert response.status_code == 201
        mock_service.encode.assert_called_once()
        call_kwargs = mock_service.encode.call_args.kwargs
        assert call_kwargs["org_id"] == "org_456"
        assert call_kwargs["session_id"] == "session_789"
        assert call_kwargs["importance"] == 0.8
        assert call_kwargs["enrich"] is False

    def test_encode_missing_required_fields(self, client, mock_service):
        """Should return 422 for missing required fields."""
        response = client.post(
            "/api/v1/encode",
            json={"content": "Hello"},  # Missing user_id
        )

        assert response.status_code == 422

    def test_encode_empty_content(self, client, mock_service):
        """Should return 422 for empty content."""
        response = client.post(
            "/api/v1/encode",
            json={"content": "", "user_id": "user_123"},
        )

        assert response.status_code == 422

    def test_encode_invalid_importance(self, client, mock_service):
        """Should return 422 for invalid importance."""
        response = client.post(
            "/api/v1/encode",
            json={"content": "Hello", "user_id": "user_123", "importance": 1.5},
        )

        assert response.status_code == 422

    def test_encode_service_not_initialized(self):
        """Should return 503 when service not initialized."""
        app = FastAPI()
        app.include_router(router, prefix="/api/v1")
        set_service(None)  # type: ignore[arg-type]
        test_client = TestClient(app)

        response = test_client.post(
            "/api/v1/encode",
            json={"content": "Hello", "role": "user", "user_id": "user_123"},
        )

        assert response.status_code == 503


class TestRecallEndpoint:
    """Tests for /recall endpoint."""

    def test_recall_success(self, client, mock_service):
        """Should recall memories and return results."""
        mock_service.recall.return_value = [
            RecallResult(
                memory_type="episodic",
                content="Hello world",
                score=0.95,
                memory_id="ep_123",
                metadata={"role": "user"},
            ),
            RecallResult(
                memory_type="structured",
                content="user@example.com",
                score=0.9,
                confidence=0.85,
                memory_id="struct_456",
                source_episode_id="ep_123",
                metadata={"category": "email"},
            ),
        ]

        response = client.post(
            "/api/v1/recall",
            json={"query": "hello", "user_id": "user_123"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "hello"
        assert data["count"] == 2
        assert data["results"][0]["memory_type"] == "episodic"
        assert data["results"][1]["confidence"] == 0.85

    def test_recall_with_all_options(self, client, mock_service):
        """Should accept all optional parameters."""
        mock_service.recall.return_value = []

        response = client.post(
            "/api/v1/recall",
            json={
                "query": "email",
                "user_id": "user_123",
                "org_id": "org_456",
                "limit": 5,
                "min_confidence": 0.8,
                "memory_types": ["structured"],
            },
        )

        assert response.status_code == 200
        mock_service.recall.assert_called_once()
        call_kwargs = mock_service.recall.call_args.kwargs
        assert call_kwargs["org_id"] == "org_456"
        assert call_kwargs["limit"] == 5
        assert call_kwargs["min_confidence"] == 0.8
        assert call_kwargs["memory_types"] == ["structured"]

    def test_recall_empty_results(self, client, mock_service):
        """Should return empty list when no matches."""
        mock_service.recall.return_value = []

        response = client.post(
            "/api/v1/recall",
            json={"query": "nonexistent", "user_id": "user_123"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 0
        assert data["results"] == []

    def test_recall_missing_required_fields(self, client, mock_service):
        """Should return 422 for missing required fields."""
        response = client.post(
            "/api/v1/recall",
            json={"query": "hello"},  # Missing user_id
        )

        assert response.status_code == 422

    def test_recall_empty_query(self, client, mock_service):
        """Should return 422 for empty query."""
        response = client.post(
            "/api/v1/recall",
            json={"query": "", "user_id": "user_123"},
        )

        assert response.status_code == 422

    def test_recall_invalid_limit(self, client, mock_service):
        """Should return 422 for invalid limit."""
        response = client.post(
            "/api/v1/recall",
            json={"query": "hello", "user_id": "user_123", "limit": 0},
        )

        assert response.status_code == 422

        response = client.post(
            "/api/v1/recall",
            json={"query": "hello", "user_id": "user_123", "limit": 101},
        )

        assert response.status_code == 422

    def test_recall_with_expand_query(self, client, mock_service):
        """Should pass expand_query parameter to service."""
        mock_service.recall.return_value = []

        response = client.post(
            "/api/v1/recall",
            json={
                "query": "email address",
                "user_id": "user_123",
                "expand_query": True,
            },
        )

        assert response.status_code == 200
        mock_service.recall.assert_called_once()
        call_kwargs = mock_service.recall.call_args.kwargs
        assert call_kwargs["expand_query"] is True

    def test_recall_expand_query_default_false(self, client, mock_service):
        """Should default expand_query to False."""
        mock_service.recall.return_value = []

        response = client.post(
            "/api/v1/recall",
            json={"query": "hello", "user_id": "user_123"},
        )

        assert response.status_code == 200
        mock_service.recall.assert_called_once()
        call_kwargs = mock_service.recall.call_args.kwargs
        assert call_kwargs["expand_query"] is False

    def test_recall_service_not_initialized(self):
        """Should return 503 when service not initialized."""
        app = FastAPI()
        app.include_router(router, prefix="/api/v1")
        set_service(None)  # type: ignore[arg-type]
        test_client = TestClient(app)

        response = test_client.post(
            "/api/v1/recall",
            json={"query": "hello", "user_id": "user_123"},
        )

        assert response.status_code == 503


class TestSchemas:
    """Tests for API schemas."""

    def test_encode_request_validation(self):
        """Should validate encode request fields."""
        request = EncodeRequest(
            content="Hello world",
            role="user",
            user_id="user_123",
        )
        assert request.content == "Hello world"
        assert request.role == "user"
        assert request.importance is None  # Default (auto-calculated)
        assert request.enrich is False  # Default (regex only)

    def test_encode_request_invalid_role(self):
        """Should reject invalid role."""
        with pytest.raises(ValueError):
            EncodeRequest(
                content="Hello",
                role="invalid",  # type: ignore[arg-type]
                user_id="user_123",
            )

    def test_recall_request_validation(self):
        """Should validate recall request fields."""
        request = RecallRequest(
            query="hello",
            user_id="user_123",
        )
        assert request.query == "hello"
        assert request.limit == 10  # Default
        assert request.memory_types is None  # Default (all types)

    def test_health_response_validation(self):
        """Should validate health response fields."""
        response = HealthResponse(
            status="healthy",
            version="0.1.0",
            storage_connected=True,
        )
        assert response.status == "healthy"

    def test_recall_request_memory_types_specific(self):
        """Should accept specific memory types."""
        request = RecallRequest(
            query="hello",
            user_id="user_123",
            memory_types=["episodic", "structured"],
        )
        assert request.memory_types == ["episodic", "structured"]

    def test_recall_request_memory_types_none_means_all(self):
        """When memory_types is None, all types should be searched."""
        request = RecallRequest(
            query="hello",
            user_id="user_123",
            memory_types=None,
        )
        assert request.memory_types is None

    def test_recall_request_memory_types_empty_valid(self):
        """Empty memory_types array is valid (returns no results)."""
        request = RecallRequest(
            query="hello",
            user_id="user_123",
            memory_types=[],
        )
        assert request.memory_types == []

    def test_recall_request_memory_types_invalid_rejected(self):
        """Invalid memory type should be rejected."""
        with pytest.raises(ValueError):
            RecallRequest(
                query="hello",
                user_id="user_123",
                memory_types=["invalid_type"],  # type: ignore[list-item]
            )


class TestDeleteEndpoint:
    """Tests for DELETE /memories/{memory_id} endpoint."""

    def test_delete_episode_success(self, client, mock_service):
        """Should delete an episodic memory."""
        mock_service.storage.delete_episode = AsyncMock(return_value=True)
        mock_service.storage.log_audit = AsyncMock()

        response = client.delete(
            "/api/v1/memories/ep_123",
            params={"user_id": "user_123"},
        )

        assert response.status_code == 204
        mock_service.storage.delete_episode.assert_called_once_with("ep_123", "user_123")
        mock_service.storage.log_audit.assert_called_once()

    def test_delete_structured_success(self, client, mock_service):
        """Should delete a structured memory."""
        # Mock memory retrieval for history logging
        mock_memory = MagicMock()
        mock_memory.model_dump.return_value = {"id": "struct_456", "summary": "test"}
        mock_service.storage.get_structured = AsyncMock(return_value=mock_memory)
        mock_service.storage.delete_structured = AsyncMock(return_value=True)
        mock_service.storage.log_audit = AsyncMock()
        mock_service.storage.log_history = AsyncMock()

        response = client.delete(
            "/api/v1/memories/struct_456",
            params={"user_id": "user_123"},
        )

        assert response.status_code == 204
        mock_service.storage.delete_structured.assert_called_once_with("struct_456", "user_123")

    def test_delete_semantic_success(self, client, mock_service):
        """Should delete a semantic memory."""
        # Mock memory retrieval for history logging
        mock_memory = MagicMock()
        mock_memory.model_dump.return_value = {"id": "sem_789", "content": "test"}
        mock_service.storage.get_semantic = AsyncMock(return_value=mock_memory)
        mock_service.storage.delete_semantic = AsyncMock(return_value=True)
        mock_service.storage.log_audit = AsyncMock()
        mock_service.storage.log_history = AsyncMock()

        response = client.delete(
            "/api/v1/memories/sem_789",
            params={"user_id": "user_123"},
        )

        assert response.status_code == 204
        mock_service.storage.delete_semantic.assert_called_once_with("sem_789", "user_123")

    def test_delete_procedural_success(self, client, mock_service):
        """Should delete a procedural memory."""
        # Mock memory retrieval for history logging
        mock_memory = MagicMock()
        mock_memory.model_dump.return_value = {"id": "proc_abc", "content": "test"}
        mock_service.storage.get_procedural = AsyncMock(return_value=mock_memory)
        mock_service.storage.delete_procedural = AsyncMock(return_value=True)
        mock_service.storage.log_audit = AsyncMock()
        mock_service.storage.log_history = AsyncMock()

        response = client.delete(
            "/api/v1/memories/proc_abc",
            params={"user_id": "user_123"},
        )

        assert response.status_code == 204
        mock_service.storage.delete_procedural.assert_called_once_with("proc_abc", "user_123")

    def test_delete_not_found(self, client, mock_service):
        """Should return 404 when memory not found."""
        mock_service.storage.delete_episode = AsyncMock(return_value=False)

        response = client.delete(
            "/api/v1/memories/ep_nonexistent",
            params={"user_id": "user_123"},
        )

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_delete_invalid_prefix(self, client, mock_service):
        """Should return 400 for invalid memory ID prefix."""
        response = client.delete(
            "/api/v1/memories/invalid_123",
            params={"user_id": "user_123"},
        )

        assert response.status_code == 400
        assert "Invalid memory ID format" in response.json()["detail"]

    def test_delete_service_not_initialized(self):
        """Should return 503 when service not initialized."""
        app = FastAPI()
        app.include_router(router, prefix="/api/v1")
        set_service(None)  # type: ignore[arg-type]
        test_client = TestClient(app)

        response = test_client.delete(
            "/api/v1/memories/ep_123",
            params={"user_id": "user_123"},
        )

        assert response.status_code == 503


class TestBulkDeleteEndpoint:
    """Tests for DELETE /users/{user_id}/memories endpoint."""

    def test_bulk_delete_success(self, client, mock_service):
        """Should delete all user memories and return counts."""
        mock_service.storage.delete_all_user_memories = AsyncMock(
            return_value={
                "episodic": 5,
                "structured": 3,
                "semantic": 2,
                "procedural": 1,
            }
        )
        mock_service.storage.log_audit = AsyncMock()

        response = client.delete("/api/v1/users/user_123/memories")

        assert response.status_code == 200
        data = response.json()
        assert data["user_id"] == "user_123"
        assert data["total_deleted"] == 11
        assert data["deleted_counts"]["episodic"] == 5
        assert data["deleted_counts"]["structured"] == 3
        mock_service.storage.log_audit.assert_called_once()

    def test_bulk_delete_with_org_filter(self, client, mock_service):
        """Should filter by org_id when provided."""
        mock_service.storage.delete_all_user_memories = AsyncMock(
            return_value={
                "episodic": 2,
                "structured": 1,
                "semantic": 0,
                "procedural": 0,
            }
        )
        mock_service.storage.log_audit = AsyncMock()

        response = client.delete(
            "/api/v1/users/user_123/memories",
            params={"org_id": "org_456"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["org_id"] == "org_456"
        assert data["total_deleted"] == 3
        mock_service.storage.delete_all_user_memories.assert_called_once_with(
            user_id="user_123",
            org_id="org_456",
        )

    def test_bulk_delete_no_memories(self, client, mock_service):
        """Should return 0 counts when user has no memories."""
        mock_service.storage.delete_all_user_memories = AsyncMock(
            return_value={
                "episodic": 0,
                "structured": 0,
                "semantic": 0,
                "procedural": 0,
            }
        )
        mock_service.storage.log_audit = AsyncMock()

        response = client.delete("/api/v1/users/user_123/memories")

        assert response.status_code == 200
        data = response.json()
        assert data["total_deleted"] == 0

    def test_bulk_delete_service_not_initialized(self):
        """Should return 503 when service not initialized."""
        app = FastAPI()
        app.include_router(router, prefix="/api/v1")
        set_service(None)  # type: ignore[arg-type]
        test_client = TestClient(app)

        response = test_client.delete("/api/v1/users/user_123/memories")

        assert response.status_code == 503


class TestUpdateMemoryEndpoint:
    """Tests for memory update endpoint."""

    @pytest.fixture
    def client(self, mock_service):
        """Create test client with mock service."""
        app = FastAPI()
        app.include_router(router, prefix="/api/v1")
        set_service(mock_service)
        return TestClient(app)

    @pytest.fixture
    def mock_service(self):
        """Create mock EngramService."""
        from engram.service import EngramService

        service = MagicMock(spec=EngramService)
        service.storage = MagicMock()
        service.embedder = MagicMock()
        service.embedder.embed = AsyncMock(return_value=[0.1] * 1536)
        return service

    @pytest.fixture
    def mock_semantic_memory(self):
        """Create a mock semantic memory."""
        from engram.models import SemanticMemory

        return SemanticMemory(
            id="sem_test123",
            content="Original content",
            user_id="user_123",
            embedding=[0.1] * 1536,
            tags=["original"],
            keywords=["test"],
            context="test context",
        )

    @pytest.fixture
    def mock_structured_memory(self):
        """Create a mock structured memory."""
        from engram.models import StructuredMemory

        return StructuredMemory(
            id="struct_test123",
            source_episode_id="ep_source123",
            user_id="user_123",
            embedding=[0.1] * 1536,
            summary="Original summary",
        )

    def test_update_semantic_content(self, client, mock_service, mock_semantic_memory):
        """Should update semantic memory content and re-embed."""
        mock_service.storage.get_semantic = AsyncMock(return_value=mock_semantic_memory)
        mock_service.storage.update_semantic_memory = AsyncMock(return_value=True)
        mock_service.storage.log_audit = AsyncMock()
        mock_service.storage.log_history = AsyncMock()

        response = client.patch(
            "/api/v1/memories/sem_test123",
            json={
                "content": "Updated content",
                "user_id": "user_123",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["memory_id"] == "sem_test123"
        assert data["memory_type"] == "semantic"
        assert data["updated"] is True
        assert data["re_embedded"] is True
        assert len(data["changes"]) == 1
        assert data["changes"][0]["field"] == "content"

    def test_update_semantic_tags(self, client, mock_service, mock_semantic_memory):
        """Should update semantic memory tags without re-embedding."""
        mock_service.storage.get_semantic = AsyncMock(return_value=mock_semantic_memory)
        mock_service.storage.update_semantic_memory = AsyncMock(return_value=True)
        mock_service.storage.log_audit = AsyncMock()
        mock_service.storage.log_history = AsyncMock()

        response = client.patch(
            "/api/v1/memories/sem_test123",
            json={
                "tags": ["updated", "new-tag"],
                "user_id": "user_123",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["updated"] is True
        assert data["re_embedded"] is False
        assert len(data["changes"]) == 1
        assert data["changes"][0]["field"] == "tags"

    def test_update_semantic_confidence(self, client, mock_service, mock_semantic_memory):
        """Should update confidence without re-embedding."""
        mock_service.storage.get_semantic = AsyncMock(return_value=mock_semantic_memory)
        mock_service.storage.update_semantic_memory = AsyncMock(return_value=True)
        mock_service.storage.log_audit = AsyncMock()
        mock_service.storage.log_history = AsyncMock()

        response = client.patch(
            "/api/v1/memories/sem_test123",
            json={
                "confidence": 0.95,
                "user_id": "user_123",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["updated"] is True
        assert data["re_embedded"] is False
        assert len(data["changes"]) == 1
        assert data["changes"][0]["field"] == "confidence"

    def test_update_structured_summary(self, client, mock_service, mock_structured_memory):
        """Should update structured memory summary and re-embed."""
        mock_service.storage.get_structured = AsyncMock(return_value=mock_structured_memory)
        mock_service.storage.update_structured_memory = AsyncMock(return_value=True)
        mock_service.storage.log_audit = AsyncMock()
        mock_service.storage.log_history = AsyncMock()

        response = client.patch(
            "/api/v1/memories/struct_test123",
            json={
                "content": "Updated summary",
                "user_id": "user_123",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["memory_id"] == "struct_test123"
        assert data["memory_type"] == "structured"
        assert data["updated"] is True
        assert data["re_embedded"] is True
        assert data["changes"][0]["field"] == "summary"

    def test_update_episodic_rejected(self, client, mock_service):
        """Should reject updates to episodic memories."""
        response = client.patch(
            "/api/v1/memories/ep_test123",
            json={
                "content": "Cannot update",
                "user_id": "user_123",
            },
        )

        assert response.status_code == 400
        assert "immutable" in response.json()["detail"].lower()

    def test_update_memory_not_found(self, client, mock_service):
        """Should return 404 when memory not found."""
        mock_service.storage.get_semantic = AsyncMock(return_value=None)

        response = client.patch(
            "/api/v1/memories/sem_nonexistent",
            json={
                "content": "Updated",
                "user_id": "user_123",
            },
        )

        assert response.status_code == 404

    def test_update_no_changes(self, client, mock_service, mock_semantic_memory):
        """Should return updated=False when no changes made."""
        mock_service.storage.get_semantic = AsyncMock(return_value=mock_semantic_memory)

        response = client.patch(
            "/api/v1/memories/sem_test123",
            json={
                "content": "Original content",  # Same as existing
                "user_id": "user_123",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["updated"] is False
        assert len(data["changes"]) == 0

    def test_update_invalid_prefix(self, client, mock_service):
        """Should reject invalid memory ID prefix."""
        response = client.patch(
            "/api/v1/memories/invalid_test123",
            json={
                "content": "Updated",
                "user_id": "user_123",
            },
        )

        assert response.status_code == 400
        assert "struct_" in response.json()["detail"] or "sem_" in response.json()["detail"]


class TestWorkflowEndpoints:
    """Tests for workflow trigger endpoints."""

    def test_consolidate_success(self, client, mock_service):
        """Should trigger consolidation and return results."""
        from engram.workflows import ConsolidationResult

        mock_result = ConsolidationResult(
            episodes_processed=10,
            semantic_memories_created=3,
            links_created=5,
            compression_ratio=3.33,
        )
        mock_service.workflow_backend = MagicMock()
        mock_service.workflow_backend.run_consolidation = AsyncMock(return_value=mock_result)

        response = client.post(
            "/api/v1/workflows/consolidate",
            json={
                "user_id": "user_123",
                "consolidation_passes": 2,
                "similarity_threshold": 0.8,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["episodes_processed"] == 10
        assert data["semantic_memories_created"] == 3
        assert data["links_created"] == 5
        assert data["compression_ratio"] == 3.33

    def test_consolidate_with_org_filter(self, client, mock_service):
        """Should pass org_id to consolidation."""
        from engram.workflows import ConsolidationResult

        mock_result = ConsolidationResult(
            episodes_processed=5,
            semantic_memories_created=1,
            links_created=2,
            compression_ratio=5.0,
        )
        mock_service.workflow_backend = MagicMock()
        mock_service.workflow_backend.run_consolidation = AsyncMock(return_value=mock_result)

        response = client.post(
            "/api/v1/workflows/consolidate",
            json={
                "user_id": "user_123",
                "org_id": "org_456",
            },
        )

        assert response.status_code == 200
        mock_service.workflow_backend.run_consolidation.assert_called_once()
        call_kwargs = mock_service.workflow_backend.run_consolidation.call_args.kwargs
        assert call_kwargs["user_id"] == "user_123"
        assert call_kwargs["org_id"] == "org_456"

    def test_decay_success(self, client, mock_service):
        """Should trigger decay and return results."""
        from engram.workflows import DecayResult

        mock_result = DecayResult(
            memories_updated=15,
            memories_archived=3,
            memories_deleted=1,
            procedural_promoted=2,
        )
        mock_service.workflow_backend = MagicMock()
        mock_service.workflow_backend.run_decay = AsyncMock(return_value=mock_result)
        mock_service.settings = MagicMock()

        response = client.post(
            "/api/v1/workflows/decay",
            json={
                "user_id": "user_123",
                "run_promotion": True,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["memories_updated"] == 15
        assert data["memories_archived"] == 3
        assert data["memories_deleted"] == 1
        assert data["procedural_promoted"] == 2

    def test_decay_without_promotion(self, client, mock_service):
        """Should pass run_promotion=False to decay."""
        from engram.workflows import DecayResult

        mock_result = DecayResult(
            memories_updated=10,
            memories_archived=2,
            memories_deleted=0,
            procedural_promoted=0,
        )
        mock_service.workflow_backend = MagicMock()
        mock_service.workflow_backend.run_decay = AsyncMock(return_value=mock_result)
        mock_service.settings = MagicMock()

        response = client.post(
            "/api/v1/workflows/decay",
            json={
                "user_id": "user_123",
                "run_promotion": False,
            },
        )

        assert response.status_code == 200
        call_kwargs = mock_service.workflow_backend.run_decay.call_args.kwargs
        assert call_kwargs["run_promotion"] is False

    def test_promote_success(self, client, mock_service):
        """Should trigger promotion and return results."""
        from engram.workflows.promotion import SynthesisResult

        mock_result = SynthesisResult(
            semantics_analyzed=8,
            procedural_created=True,
            procedural_updated=False,
            procedural_id="proc_123",
        )
        mock_service.workflow_backend = MagicMock()
        mock_service.workflow_backend.run_promotion = AsyncMock(return_value=mock_result)

        response = client.post(
            "/api/v1/workflows/promote",
            json={"user_id": "user_123"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["semantics_analyzed"] == 8
        assert data["procedural_created"] is True
        assert data["procedural_updated"] is False
        assert data["procedural_id"] == "proc_123"

    def test_structure_success(self, client, mock_service):
        """Should trigger structure for a single episode."""
        from engram.models import Episode, StructuredMemory
        from engram.workflows import StructureResult

        mock_episode = Episode(
            content="My email is test@example.com",
            role="user",
            user_id="user_123",
            embedding=[0.1] * 1536,
        )
        mock_structured = StructuredMemory(
            source_episode_id=mock_episode.id,
            mode="rich",
            user_id="user_123",
            emails=["test@example.com"],
        )
        mock_result = StructureResult(
            episode_id=mock_episode.id,
            structured_memory_id=mock_structured.id,
            structured=mock_structured,
            extracts_count=1,
            deterministic_count=1,
            llm_count=0,
            processing_time_ms=150,
        )
        mock_service.storage.get_episode = AsyncMock(return_value=mock_episode)
        mock_service.workflow_backend = MagicMock()
        mock_service.workflow_backend.run_structure = AsyncMock(return_value=mock_result)

        response = client.post(
            "/api/v1/workflows/structure",
            json={
                "episode_id": mock_episode.id,
                "user_id": "user_123",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["episode_id"] == mock_episode.id
        assert data["extracts_count"] == 1
        assert data["skipped"] is False

    def test_structure_episode_not_found(self, client, mock_service):
        """Should return 404 when episode not found."""
        mock_service.storage.get_episode = AsyncMock(return_value=None)
        mock_service.workflow_backend = MagicMock()

        response = client.post(
            "/api/v1/workflows/structure",
            json={
                "episode_id": "ep_nonexistent",
                "user_id": "user_123",
            },
        )

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_structure_skipped_already_processed(self, client, mock_service):
        """Should return skipped=True when already structured."""
        from engram.models import Episode

        mock_episode = Episode(
            content="Already processed",
            role="user",
            user_id="user_123",
            embedding=[0.1] * 1536,
        )
        mock_service.storage.get_episode = AsyncMock(return_value=mock_episode)
        mock_service.workflow_backend = MagicMock()
        mock_service.workflow_backend.run_structure = AsyncMock(return_value=None)

        response = client.post(
            "/api/v1/workflows/structure",
            json={
                "episode_id": mock_episode.id,
                "user_id": "user_123",
                "skip_if_structured": True,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["skipped"] is True
        assert data["extracts_count"] == 0

    def test_structure_batch_success(self, client, mock_service):
        """Should process batch of episodes."""
        from engram.models import StructuredMemory
        from engram.workflows import StructureResult

        mock_results = [
            StructureResult(
                episode_id=f"ep_{i}",
                structured_memory_id=f"struct_{i}",
                structured=StructuredMemory(
                    source_episode_id=f"ep_{i}",
                    mode="rich",
                    user_id="user_123",
                ),
                extracts_count=i,
                deterministic_count=i,
                llm_count=0,
                processing_time_ms=100,
            )
            for i in range(3)
        ]
        mock_service.workflow_backend = MagicMock()
        mock_service.workflow_backend.run_structure_batch = AsyncMock(return_value=mock_results)

        response = client.post(
            "/api/v1/workflows/structure/batch",
            json={
                "user_id": "user_123",
                "limit": 10,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["episodes_processed"] == 3
        assert data["total_extracts"] == 3  # 0 + 1 + 2
        assert len(data["results"]) == 3

    def test_workflow_status_success(self, client, mock_service):
        """Should return workflow status."""
        from datetime import datetime

        from engram.workflows import WorkflowState, WorkflowStatus

        mock_status = WorkflowStatus(
            workflow_id="wf_123",
            workflow_type="consolidation",
            state=WorkflowState.COMPLETED,
            started_at=datetime(2024, 1, 1, 12, 0, 0),
            completed_at=datetime(2024, 1, 1, 12, 5, 0),
            result={"episodes_processed": 10},
        )
        mock_service.workflow_backend = MagicMock()
        mock_service.workflow_backend.get_workflow_status = AsyncMock(return_value=mock_status)

        response = client.get("/api/v1/workflows/wf_123/status")

        assert response.status_code == 200
        data = response.json()
        assert data["workflow_id"] == "wf_123"
        assert data["workflow_type"] == "consolidation"
        assert data["state"] == "completed"
        assert data["result"]["episodes_processed"] == 10

    def test_workflow_status_not_found(self, client, mock_service):
        """Should return 404 when workflow not found."""
        mock_service.workflow_backend = MagicMock()
        mock_service.workflow_backend.get_workflow_status = AsyncMock(return_value=None)

        response = client.get("/api/v1/workflows/wf_nonexistent/status")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_workflow_service_not_initialized(self):
        """Should return 503 when service not initialized."""
        app = FastAPI()
        app.include_router(router, prefix="/api/v1")
        set_service(None)  # type: ignore[arg-type]
        test_client = TestClient(app)

        response = test_client.post(
            "/api/v1/workflows/consolidate",
            json={"user_id": "user_123"},
        )

        assert response.status_code == 503


class TestLinkEndpoints:
    """Tests for memory link endpoints."""

    @pytest.fixture
    def client(self, mock_service):
        """Create test client with mock service."""
        app = FastAPI()
        app.include_router(router, prefix="/api/v1")
        set_service(mock_service)
        return TestClient(app)

    @pytest.fixture
    def mock_service(self):
        """Create mock EngramService."""
        from engram.service import EngramService

        service = MagicMock(spec=EngramService)
        service.storage = MagicMock()
        return service

    @pytest.fixture
    def mock_semantic_memory(self):
        """Create a mock semantic memory."""
        from engram.models import SemanticMemory

        return SemanticMemory(
            id="sem_test123",
            content="Test semantic memory",
            user_id="user_123",
            embedding=[0.1] * 1536,
            related_ids=[],
            link_types={},
        )

    @pytest.fixture
    def mock_semantic_memory_2(self):
        """Create a second mock semantic memory."""
        from engram.models import SemanticMemory

        return SemanticMemory(
            id="sem_target456",
            content="Target semantic memory",
            user_id="user_123",
            embedding=[0.1] * 1536,
            related_ids=[],
            link_types={},
        )

    def test_create_link_success(
        self, client, mock_service, mock_semantic_memory, mock_semantic_memory_2
    ):
        """Should create link between memories."""
        mock_service.storage.get_semantic = AsyncMock(
            side_effect=lambda id, _: mock_semantic_memory
            if id == "sem_test123"
            else mock_semantic_memory_2
        )
        mock_service.storage.update_semantic_memory = AsyncMock(return_value=True)

        response = client.post(
            "/api/v1/memories/sem_test123/links",
            json={
                "target_id": "sem_target456",
                "link_type": "related",
                "user_id": "user_123",
            },
        )

        assert response.status_code == 201
        data = response.json()
        assert data["source_id"] == "sem_test123"
        assert data["target_id"] == "sem_target456"
        assert data["link_type"] == "related"
        assert data["bidirectional"] is True

    def test_create_supersedes_link(
        self, client, mock_service, mock_semantic_memory, mock_semantic_memory_2
    ):
        """Should create directional supersedes link."""
        mock_service.storage.get_semantic = AsyncMock(
            side_effect=lambda id, _: mock_semantic_memory
            if id == "sem_test123"
            else mock_semantic_memory_2
        )
        mock_service.storage.update_semantic_memory = AsyncMock(return_value=True)

        response = client.post(
            "/api/v1/memories/sem_test123/links",
            json={
                "target_id": "sem_target456",
                "link_type": "supersedes",
                "user_id": "user_123",
            },
        )

        assert response.status_code == 201
        data = response.json()
        assert data["link_type"] == "supersedes"
        assert data["bidirectional"] is False  # supersedes is directional

    def test_create_link_invalid_source(self, client, mock_service):
        """Should reject invalid source memory ID format."""
        response = client.post(
            "/api/v1/memories/ep_episode123/links",
            json={
                "target_id": "sem_target456",
                "link_type": "related",
                "user_id": "user_123",
            },
        )

        assert response.status_code == 400
        assert "sem_" in response.json()["detail"] or "proc_" in response.json()["detail"]

    def test_create_link_invalid_target(self, client, mock_service, mock_semantic_memory):
        """Should reject invalid target memory ID format."""
        mock_service.storage.get_semantic = AsyncMock(return_value=mock_semantic_memory)

        response = client.post(
            "/api/v1/memories/sem_test123/links",
            json={
                "target_id": "invalid_id",
                "link_type": "related",
                "user_id": "user_123",
            },
        )

        assert response.status_code == 400

    def test_create_link_source_not_found(self, client, mock_service):
        """Should return 404 when source memory not found."""
        mock_service.storage.get_semantic = AsyncMock(return_value=None)

        response = client.post(
            "/api/v1/memories/sem_nonexistent/links",
            json={
                "target_id": "sem_target456",
                "link_type": "related",
                "user_id": "user_123",
            },
        )

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_list_links_success(self, client, mock_service):
        """Should list all links from a memory."""
        from engram.models import SemanticMemory

        memory_with_links = SemanticMemory(
            id="sem_test123",
            content="Test memory",
            user_id="user_123",
            embedding=[0.1] * 1536,
            related_ids=["sem_linked1", "sem_linked2"],
            link_types={"sem_linked1": "related", "sem_linked2": "contradicts"},
        )
        mock_service.storage.get_semantic = AsyncMock(return_value=memory_with_links)

        response = client.get("/api/v1/memories/sem_test123/links?user_id=user_123")

        assert response.status_code == 200
        data = response.json()
        assert data["memory_id"] == "sem_test123"
        assert data["count"] == 2
        assert len(data["links"]) == 2
        link_targets = {link["target_id"] for link in data["links"]}
        assert "sem_linked1" in link_targets
        assert "sem_linked2" in link_targets

    def test_list_links_empty(self, client, mock_service, mock_semantic_memory):
        """Should return empty list when no links."""
        mock_service.storage.get_semantic = AsyncMock(return_value=mock_semantic_memory)

        response = client.get("/api/v1/memories/sem_test123/links?user_id=user_123")

        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 0
        assert data["links"] == []

    def test_list_links_not_found(self, client, mock_service):
        """Should return 404 when memory not found."""
        mock_service.storage.get_semantic = AsyncMock(return_value=None)

        response = client.get("/api/v1/memories/sem_nonexistent/links?user_id=user_123")

        assert response.status_code == 404

    def test_delete_link_success(
        self, client, mock_service, mock_semantic_memory, mock_semantic_memory_2
    ):
        """Should delete link between memories."""
        # Add a link to the source memory
        mock_semantic_memory.related_ids = ["sem_target456"]
        mock_semantic_memory.link_types = {"sem_target456": "related"}
        # Add reverse link to target
        mock_semantic_memory_2.related_ids = ["sem_test123"]
        mock_semantic_memory_2.link_types = {"sem_test123": "related"}

        mock_service.storage.get_semantic = AsyncMock(
            side_effect=lambda id, _: mock_semantic_memory
            if id == "sem_test123"
            else mock_semantic_memory_2
        )
        mock_service.storage.update_semantic_memory = AsyncMock(return_value=True)

        response = client.delete(
            "/api/v1/memories/sem_test123/links/sem_target456?user_id=user_123"
        )

        assert response.status_code == 200
        data = response.json()
        assert data["source_id"] == "sem_test123"
        assert data["target_id"] == "sem_target456"
        assert data["removed"] is True
        assert data["reverse_removed"] is True

    def test_delete_link_not_found(self, client, mock_service, mock_semantic_memory):
        """Should return removed=False when link doesn't exist."""
        mock_service.storage.get_semantic = AsyncMock(return_value=mock_semantic_memory)
        mock_service.storage.update_semantic_memory = AsyncMock(return_value=True)

        response = client.delete(
            "/api/v1/memories/sem_test123/links/sem_nonexistent?user_id=user_123"
        )

        assert response.status_code == 200
        data = response.json()
        assert data["removed"] is False

    def test_delete_link_memory_not_found(self, client, mock_service):
        """Should return 404 when source memory not found."""
        mock_service.storage.get_semantic = AsyncMock(return_value=None)

        response = client.delete(
            "/api/v1/memories/sem_nonexistent/links/sem_target?user_id=user_123"
        )

        assert response.status_code == 404


class TestListMemoriesEndpoint:
    """Tests for GET /memories endpoint."""

    def test_list_memories_basic(self, client, mock_service):
        """Should list memories with basic filter."""
        mock_service.storage.search_memories = AsyncMock(
            return_value=(
                [
                    {
                        "id": "ep_test1",
                        "memory_type": "episodic",
                        "content": "Test content 1",
                        "user_id": "user_123",
                        "org_id": None,
                        "session_id": "session_1",
                        "confidence": None,
                        "created_at": "2025-01-15T10:00:00",
                    },
                    {
                        "id": "sem_test2",
                        "memory_type": "semantic",
                        "content": "Test content 2",
                        "user_id": "user_123",
                        "org_id": None,
                        "session_id": None,
                        "confidence": 0.85,
                        "created_at": "2025-01-14T10:00:00",
                    },
                ],
                2,
            )
        )

        response = client.get("/api/v1/memories?user_id=user_123")

        assert response.status_code == 200
        data = response.json()
        assert len(data["memories"]) == 2
        assert data["total"] == 2
        assert data["limit"] == 50
        assert data["offset"] == 0
        assert data["has_more"] is False
        assert data["memories"][0]["id"] == "ep_test1"
        assert data["memories"][1]["confidence"] == 0.85

    def test_list_memories_with_memory_types_filter(self, client, mock_service):
        """Should filter by memory types."""
        mock_service.storage.search_memories = AsyncMock(return_value=([], 0))

        response = client.get("/api/v1/memories?user_id=user_123&memory_types=episodic,semantic")

        assert response.status_code == 200
        mock_service.storage.search_memories.assert_called_once()
        call_kwargs = mock_service.storage.search_memories.call_args[1]
        assert call_kwargs["memory_types"] == ["episodic", "semantic"]

    def test_list_memories_with_session_id_filter(self, client, mock_service):
        """Should filter by session ID."""
        mock_service.storage.search_memories = AsyncMock(return_value=([], 0))

        response = client.get("/api/v1/memories?user_id=user_123&session_id=session_abc")

        assert response.status_code == 200
        call_kwargs = mock_service.storage.search_memories.call_args[1]
        assert call_kwargs["session_id"] == "session_abc"

    def test_list_memories_with_date_filters(self, client, mock_service):
        """Should filter by date range."""
        mock_service.storage.search_memories = AsyncMock(return_value=([], 0))

        response = client.get(
            "/api/v1/memories?user_id=user_123"
            "&created_after=2025-01-01T00:00:00"
            "&created_before=2025-01-31T23:59:59"
        )

        assert response.status_code == 200
        call_kwargs = mock_service.storage.search_memories.call_args[1]
        assert call_kwargs["created_after"] is not None
        assert call_kwargs["created_before"] is not None

    def test_list_memories_with_min_confidence(self, client, mock_service):
        """Should filter by minimum confidence."""
        mock_service.storage.search_memories = AsyncMock(return_value=([], 0))

        response = client.get("/api/v1/memories?user_id=user_123&min_confidence=0.7")

        assert response.status_code == 200
        call_kwargs = mock_service.storage.search_memories.call_args[1]
        assert call_kwargs["min_confidence"] == 0.7

    def test_list_memories_with_sorting(self, client, mock_service):
        """Should support sorting parameters."""
        mock_service.storage.search_memories = AsyncMock(return_value=([], 0))

        response = client.get("/api/v1/memories?user_id=user_123&sort_by=confidence&sort_order=asc")

        assert response.status_code == 200
        call_kwargs = mock_service.storage.search_memories.call_args[1]
        assert call_kwargs["sort_by"] == "confidence"
        assert call_kwargs["sort_order"] == "asc"

    def test_list_memories_with_pagination(self, client, mock_service):
        """Should support pagination parameters."""
        mock_service.storage.search_memories = AsyncMock(
            return_value=(
                [
                    {
                        "id": "ep_test1",
                        "memory_type": "episodic",
                        "content": "Test",
                        "user_id": "user_123",
                        "org_id": None,
                        "session_id": None,
                        "confidence": None,
                        "created_at": "2025-01-15T10:00:00",
                    }
                ],
                100,
            )
        )

        response = client.get("/api/v1/memories?user_id=user_123&limit=20&offset=40")

        assert response.status_code == 200
        data = response.json()
        assert data["limit"] == 20
        assert data["offset"] == 40
        assert data["total"] == 100
        assert data["has_more"] is True

    def test_list_memories_invalid_memory_types(self, client, mock_service):
        """Should return 400 for invalid memory types."""
        response = client.get("/api/v1/memories?user_id=user_123&memory_types=invalid_type")

        assert response.status_code == 400
        assert "Invalid memory types" in response.json()["detail"]

    def test_list_memories_invalid_sort_by(self, client, mock_service):
        """Should return 400 for invalid sort field."""
        response = client.get("/api/v1/memories?user_id=user_123&sort_by=invalid_field")

        assert response.status_code == 400
        assert "Invalid sort_by" in response.json()["detail"]

    def test_list_memories_invalid_sort_order(self, client, mock_service):
        """Should return 400 for invalid sort order."""
        response = client.get("/api/v1/memories?user_id=user_123&sort_order=invalid")

        assert response.status_code == 400
        assert "Invalid sort_order" in response.json()["detail"]

    def test_list_memories_invalid_limit(self, client, mock_service):
        """Should return 400 for invalid limit."""
        response = client.get("/api/v1/memories?user_id=user_123&limit=500")

        assert response.status_code == 400
        assert "Invalid limit" in response.json()["detail"]

    def test_list_memories_invalid_offset(self, client, mock_service):
        """Should return 400 for invalid offset."""
        response = client.get("/api/v1/memories?user_id=user_123&offset=-10")

        assert response.status_code == 400
        assert "Invalid offset" in response.json()["detail"]

    def test_list_memories_invalid_timestamp(self, client, mock_service):
        """Should return 400 for invalid timestamp format."""
        response = client.get("/api/v1/memories?user_id=user_123&created_after=invalid-date")

        assert response.status_code == 400
        assert "Invalid created_after timestamp" in response.json()["detail"]

    def test_list_memories_invalid_min_confidence(self, client, mock_service):
        """Should return 400 for invalid min_confidence."""
        response = client.get("/api/v1/memories?user_id=user_123&min_confidence=1.5")

        assert response.status_code == 400
        assert "Invalid min_confidence" in response.json()["detail"]

    def test_list_memories_missing_user_id(self, client, mock_service):
        """Should return 422 when user_id is missing."""
        response = client.get("/api/v1/memories")

        assert response.status_code == 422


class TestDeleteMemoryEndpoint:
    """Tests for DELETE /memories/{memory_id} endpoint."""

    def test_delete_episodic_memory(self, client, mock_service):
        """Should delete an episodic memory and return 204."""
        mock_service.storage.delete_episode = AsyncMock(return_value=True)
        mock_service.storage.log_audit = AsyncMock()

        response = client.delete("/api/v1/memories/ep_test123?user_id=user_123")

        assert response.status_code == 204
        mock_service.storage.delete_episode.assert_called_once_with("ep_test123", "user_123")

    def test_delete_structured_memory(self, client, mock_service):
        """Should delete a structured memory and return 204."""
        mock_service.storage.get_structured = AsyncMock(return_value=None)
        mock_service.storage.delete_structured = AsyncMock(return_value=True)
        mock_service.storage.log_audit = AsyncMock()

        response = client.delete("/api/v1/memories/struct_test123?user_id=user_123")

        assert response.status_code == 204

    def test_delete_semantic_memory(self, client, mock_service):
        """Should delete a semantic memory and return 204."""
        mock_service.storage.get_semantic = AsyncMock(return_value=None)
        mock_service.storage.delete_semantic = AsyncMock(return_value=True)
        mock_service.storage.log_audit = AsyncMock()

        response = client.delete("/api/v1/memories/sem_test123?user_id=user_123")

        assert response.status_code == 204

    def test_delete_procedural_memory(self, client, mock_service):
        """Should delete a procedural memory and return 204."""
        mock_service.storage.get_procedural = AsyncMock(return_value=None)
        mock_service.storage.delete_procedural = AsyncMock(return_value=True)
        mock_service.storage.log_audit = AsyncMock()

        response = client.delete("/api/v1/memories/proc_test123?user_id=user_123")

        assert response.status_code == 204

    def test_delete_memory_not_found(self, client, mock_service):
        """Should return 404 when memory doesn't exist."""
        mock_service.storage.delete_episode = AsyncMock(return_value=False)

        response = client.delete("/api/v1/memories/ep_nonexistent?user_id=user_123")

        assert response.status_code == 404
        assert "Memory not found" in response.json()["detail"]

    def test_delete_memory_invalid_format(self, client, mock_service):
        """Should return 400 for invalid memory ID format."""
        response = client.delete("/api/v1/memories/invalid_id?user_id=user_123")

        assert response.status_code == 400
        assert "Invalid memory ID format" in response.json()["detail"]

    def test_delete_memory_missing_user_id(self, client, mock_service):
        """Should return 422 when user_id is missing."""
        response = client.delete("/api/v1/memories/ep_test123")

        assert response.status_code == 422
