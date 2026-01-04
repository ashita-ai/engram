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
from engram.models import Episode, Fact
from engram.service import EncodeResult, EngramService, RecallResult


@pytest.fixture
def mock_service():
    """Create a mock EngramService."""
    service = MagicMock(spec=EngramService)
    service.encode = AsyncMock()
    service.recall = AsyncMock()
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
        """Should encode content and return episode with facts."""
        mock_episode = Episode(
            content="Email me at user@example.com",
            role="user",
            user_id="user_123",
            embedding=[0.1, 0.2, 0.3],
        )
        mock_fact = Fact(
            content="user@example.com",
            category="email",
            source_episode_id=mock_episode.id,
            user_id="user_123",
            embedding=[0.1, 0.2, 0.3],
        )
        mock_service.encode.return_value = EncodeResult(episode=mock_episode, facts=[mock_fact])

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
        assert data["fact_count"] == 1
        assert data["facts"][0]["content"] == "user@example.com"

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
        mock_service.encode.return_value = EncodeResult(episode=mock_episode, facts=[])

        response = client.post(
            "/api/v1/encode",
            json={
                "content": "Hello",
                "role": "assistant",
                "user_id": "user_123",
                "org_id": "org_456",
                "session_id": "session_789",
                "importance": 0.8,
                "run_extraction": False,
            },
        )

        assert response.status_code == 201
        mock_service.encode.assert_called_once()
        call_kwargs = mock_service.encode.call_args.kwargs
        assert call_kwargs["org_id"] == "org_456"
        assert call_kwargs["session_id"] == "session_789"
        assert call_kwargs["importance"] == 0.8
        assert call_kwargs["run_extraction"] is False

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
                memory_type="episode",
                content="Hello world",
                score=0.95,
                memory_id="ep_123",
                metadata={"role": "user"},
            ),
            RecallResult(
                memory_type="fact",
                content="user@example.com",
                score=0.9,
                confidence=0.85,
                memory_id="fact_456",
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
        assert data["results"][0]["memory_type"] == "episode"
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
                "include_episodes": False,
                "include_facts": True,
            },
        )

        assert response.status_code == 200
        mock_service.recall.assert_called_once()
        call_kwargs = mock_service.recall.call_args.kwargs
        assert call_kwargs["org_id"] == "org_456"
        assert call_kwargs["limit"] == 5
        assert call_kwargs["min_confidence"] == 0.8
        assert call_kwargs["include_episodes"] is False
        assert call_kwargs["include_facts"] is True

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
        assert request.importance == 0.5  # Default
        assert request.run_extraction is True  # Default

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
        assert request.include_episodes is True  # Default
        assert request.include_facts is True  # Default

    def test_health_response_validation(self):
        """Should validate health response fields."""
        response = HealthResponse(
            status="healthy",
            version="0.1.0",
            storage_connected=True,
        )
        assert response.status == "healthy"
