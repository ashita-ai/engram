"""Tests for Engram exception hierarchy."""

import pytest

from engram.exceptions import (
    AuthenticationError,
    AuthorizationError,
    ConfigurationError,
    ConsolidationError,
    EmbeddingError,
    EngramError,
    ExtractionError,
    NotFoundError,
    RateLimitError,
    StorageError,
    ValidationError,
)


class TestEngramError:
    """Tests for the base EngramError class."""

    def test_error_message(self):
        """Should store and return message."""
        error = EngramError("Something went wrong")
        assert error.message == "Something went wrong"
        assert str(error) == "Something went wrong"

    def test_error_code(self):
        """Should have default error code."""
        error = EngramError("test")
        assert error.code == "engram_error"

    def test_to_dict(self):
        """Should convert to API-friendly dict."""
        error = EngramError("Something went wrong")
        result = error.to_dict()
        assert result == {
            "error": {
                "code": "engram_error",
                "message": "Something went wrong",
            }
        }

    def test_inheritance(self):
        """All custom exceptions should inherit from EngramError."""
        exceptions = [
            ValidationError("field", "invalid"),
            NotFoundError("resource", "id"),
            StorageError("failed"),
            EmbeddingError("failed"),
            ExtractionError("failed"),
            ConsolidationError("failed"),
            RateLimitError(60),
            ConfigurationError("missing"),
            AuthenticationError("invalid"),
            AuthorizationError("forbidden"),
        ]
        for exc in exceptions:
            assert isinstance(exc, EngramError)
            assert isinstance(exc, Exception)


class TestValidationError:
    """Tests for ValidationError."""

    def test_field_and_message(self):
        """Should store field and message."""
        error = ValidationError("email", "Invalid email format")
        assert error.field == "email"
        assert "email" in error.message
        assert "Invalid email format" in error.message

    def test_error_code(self):
        """Should have validation_error code."""
        error = ValidationError("field", "message")
        assert error.code == "validation_error"

    def test_to_dict_includes_field(self):
        """Should include field in dict representation."""
        error = ValidationError("user_id", "Required field")
        result = error.to_dict()
        assert result["error"]["code"] == "validation_error"
        assert result["error"]["field"] == "user_id"
        assert "user_id" in result["error"]["message"]


class TestNotFoundError:
    """Tests for NotFoundError."""

    def test_resource_info(self):
        """Should store resource type and ID."""
        error = NotFoundError("Episode", "ep_123")
        assert error.resource_type == "Episode"
        assert error.resource_id == "ep_123"
        assert "Episode not found: ep_123" in error.message

    def test_error_code(self):
        """Should have not_found code."""
        error = NotFoundError("resource", "id")
        assert error.code == "not_found"

    def test_to_dict_includes_resource_info(self):
        """Should include resource info in dict representation."""
        error = NotFoundError("SemanticMemory", "sem_456")
        result = error.to_dict()
        assert result["error"]["code"] == "not_found"
        assert result["error"]["resource_type"] == "SemanticMemory"
        assert result["error"]["resource_id"] == "sem_456"


class TestRateLimitError:
    """Tests for RateLimitError."""

    def test_retry_after(self):
        """Should store retry_after value."""
        error = RateLimitError(60)
        assert error.retry_after == 60
        assert "60s" in error.message

    def test_error_code(self):
        """Should have rate_limit_exceeded code."""
        error = RateLimitError(30)
        assert error.code == "rate_limit_exceeded"

    def test_to_dict_includes_retry_after(self):
        """Should include retry_after in dict representation."""
        error = RateLimitError(120)
        result = error.to_dict()
        assert result["error"]["code"] == "rate_limit_exceeded"
        assert result["error"]["retry_after"] == 120


class TestSimpleErrors:
    """Tests for simple error types with just messages."""

    def test_storage_error(self):
        """StorageError should have correct code."""
        error = StorageError("Connection failed")
        assert error.code == "storage_error"
        assert error.message == "Connection failed"

    def test_embedding_error(self):
        """EmbeddingError should have correct code."""
        error = EmbeddingError("API timeout")
        assert error.code == "embedding_error"
        assert error.message == "API timeout"

    def test_extraction_error(self):
        """ExtractionError should have correct code."""
        error = ExtractionError("Invalid LLM response")
        assert error.code == "extraction_error"
        assert error.message == "Invalid LLM response"

    def test_consolidation_error(self):
        """ConsolidationError should have correct code."""
        error = ConsolidationError("Workflow failed")
        assert error.code == "consolidation_error"
        assert error.message == "Workflow failed"

    def test_configuration_error(self):
        """ConfigurationError should have correct code."""
        error = ConfigurationError("Missing API key")
        assert error.code == "configuration_error"
        assert error.message == "Missing API key"

    def test_authentication_error(self):
        """AuthenticationError should have correct code."""
        error = AuthenticationError("Invalid token")
        assert error.code == "authentication_error"
        assert error.message == "Invalid token"

    def test_authorization_error(self):
        """AuthorizationError should have correct code."""
        error = AuthorizationError("Access denied")
        assert error.code == "authorization_error"
        assert error.message == "Access denied"


class TestExceptionCatching:
    """Tests for exception catching patterns."""

    def test_catch_all_engram_errors(self):
        """Should be able to catch all Engram errors with base class."""
        errors_to_raise = [
            ValidationError("field", "message"),
            NotFoundError("type", "id"),
            StorageError("message"),
            RateLimitError(60),
        ]

        for error in errors_to_raise:
            with pytest.raises(EngramError):
                raise error

    def test_catch_specific_error(self):
        """Should be able to catch specific error types."""
        with pytest.raises(ValidationError):
            raise ValidationError("field", "message")

        with pytest.raises(NotFoundError):
            raise NotFoundError("type", "id")

        with pytest.raises(RateLimitError):
            raise RateLimitError(60)
