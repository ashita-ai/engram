"""Engram exception hierarchy.

Provides structured exceptions for error handling throughout the codebase.
All exceptions inherit from EngramError for easy catching.
"""

from __future__ import annotations


class EngramError(Exception):
    """Base exception for all Engram errors.

    All custom exceptions in Engram inherit from this class,
    allowing callers to catch all Engram-related errors with
    a single except clause.

    Attributes:
        message: Human-readable error description.
        code: Machine-readable error code for API responses.
    """

    code: str = "engram_error"

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)

    def to_dict(self) -> dict[str, object]:
        """Convert exception to API-friendly dictionary."""
        return {
            "error": {
                "code": self.code,
                "message": self.message,
            }
        }


class ValidationError(EngramError):
    """Invalid input provided.

    Raised when user input fails validation checks.

    Attributes:
        field: The field that failed validation.
        message: Description of the validation failure.
    """

    code: str = "validation_error"

    def __init__(self, field: str, message: str) -> None:
        self.field = field
        super().__init__(f"{field}: {message}")

    def to_dict(self) -> dict[str, object]:
        """Convert exception to API-friendly dictionary."""
        return {
            "error": {
                "code": self.code,
                "field": self.field,
                "message": self.message,
            }
        }


class NotFoundError(EngramError):
    """Resource not found.

    Raised when a requested resource (memory, episode, etc.) doesn't exist.

    Attributes:
        resource_type: Type of resource (e.g., "episode", "semantic_memory").
        resource_id: ID of the missing resource.
    """

    code: str = "not_found"

    def __init__(self, resource_type: str, resource_id: str) -> None:
        self.resource_type = resource_type
        self.resource_id = resource_id
        super().__init__(f"{resource_type} not found: {resource_id}")

    def to_dict(self) -> dict[str, object]:
        """Convert exception to API-friendly dictionary."""
        return {
            "error": {
                "code": self.code,
                "resource_type": self.resource_type,
                "resource_id": self.resource_id,
                "message": self.message,
            }
        }


class StorageError(EngramError):
    """Storage operation failed.

    Raised when a database or vector store operation fails.
    """

    code: str = "storage_error"


class EmbeddingError(EngramError):
    """Embedding generation failed.

    Raised when the embedding service fails to generate vectors.
    """

    code: str = "embedding_error"


class ExtractionError(EngramError):
    """Information extraction failed.

    Raised when LLM-based extraction produces invalid output.
    """

    code: str = "extraction_error"


class ConsolidationError(EngramError):
    """Memory consolidation failed.

    Raised when background consolidation workflows fail.
    """

    code: str = "consolidation_error"


class RateLimitError(EngramError):
    """Rate limit exceeded.

    Raised when a client exceeds their rate limit.

    Attributes:
        retry_after: Seconds until the client can retry.
    """

    code: str = "rate_limit_exceeded"

    def __init__(self, retry_after: int) -> None:
        self.retry_after = retry_after
        super().__init__(f"Rate limit exceeded. Retry after {retry_after}s")

    def to_dict(self) -> dict[str, object]:
        """Convert exception to API-friendly dictionary."""
        return {
            "error": {
                "code": self.code,
                "retry_after": self.retry_after,
                "message": self.message,
            }
        }


class ConfigurationError(EngramError):
    """Configuration error.

    Raised when required configuration is missing or invalid.
    """

    code: str = "configuration_error"


class AuthenticationError(EngramError):
    """Authentication failed.

    Raised when authentication credentials are invalid or missing.
    """

    code: str = "authentication_error"


class AuthorizationError(EngramError):
    """Authorization failed.

    Raised when user lacks permission to perform an action.
    """

    code: str = "authorization_error"
