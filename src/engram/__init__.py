"""Engram: Memory you can trust.

A memory system for AI applications that preserves ground truth,
tracks confidence, and prevents hallucinations.

Quick Start:
    from engram.service import EngramService

    async with EngramService.create() as engram:
        # Store an interaction
        result = await engram.encode(
            content="My email is john@example.com",
            role="user",
            user_id="user_123",
        )

        # Retrieve relevant memories
        memories = await engram.recall(
            query="What's the user's email?",
            user_id="user_123",
            min_confidence=0.7,
        )

Memory Types:
    - Episode: Immutable ground truth (raw interactions)
    - StructuredMemory: Per-episode extraction (emails, phones, URLs, negations)
    - SemanticMemory: LLM-inferred knowledge (cross-episode synthesis)
    - ProceduralMemory: Behavioral patterns
    - Working: Current session context (in-memory, volatile)

For more information, see: https://github.com/ashita-ai/engram
"""

__version__ = "0.1.0"

# Configuration
from .config import ConfidenceWeights, Settings, settings

# Context Manager
from .context import get_current_memory, memory_context, scoped_memory

# Exceptions
from .exceptions import (
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

# Logging
from .logging import (
    bind_context,
    clear_context,
    configure_logging,
    get_logger,
    logger,
    unbind_context,
)

# Models
from .models import (
    AuditEntry,
    ConfidenceScore,
    Episode,
    ExtractionMethod,
    ProceduralMemory,
    SemanticMemory,
    StructuredMemory,
)

__all__ = [
    # Version
    "__version__",
    # Configuration
    "Settings",
    "ConfidenceWeights",
    "settings",
    # Exceptions
    "EngramError",
    "ValidationError",
    "NotFoundError",
    "StorageError",
    "EmbeddingError",
    "ExtractionError",
    "ConsolidationError",
    "RateLimitError",
    "ConfigurationError",
    "AuthenticationError",
    "AuthorizationError",
    # Logging
    "configure_logging",
    "get_logger",
    "logger",
    "bind_context",
    "clear_context",
    "unbind_context",
    # Context Manager
    "memory_context",
    "scoped_memory",
    "get_current_memory",
    # Models
    "ConfidenceScore",
    "ExtractionMethod",
    "Episode",
    "StructuredMemory",
    "SemanticMemory",
    "ProceduralMemory",
    "AuditEntry",
]
