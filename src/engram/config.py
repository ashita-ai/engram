"""Configuration management for Engram."""

import logging
import os
import secrets
import warnings
from typing import Literal

from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


def _generate_dev_secret_key() -> str:
    """Generate a random secret key for development use.

    In development/test environments, if no secret key is provided,
    we generate a random one at startup. This ensures:
    1. No hardcoded secrets in source code
    2. Tokens are invalidated on restart (acceptable for dev)
    3. Production always requires explicit key configuration

    Returns:
        A cryptographically secure random hex string (64 characters).
    """
    return secrets.token_hex(32)


class RerankWeights(BaseModel):
    """Configurable weights for context-aware reranking.

    The rerank formula combines multiple signals:
        final_score = (
            similarity * similarity_weight +
            recency_score * recency_weight +
            confidence * confidence_weight +
            session_match * session_weight +
            access_boost * access_weight
        )

    Weights should sum to 1.0 for normalized scores.

    Attributes:
        similarity: Weight for vector similarity (0.50 default).
        recency: Weight for time decay (0.20 default).
        confidence: Weight for memory confidence (0.15 default).
        session: Weight for same-session bonus (0.10 default).
        access: Weight for frequently accessed memories (0.05 default).
        recency_half_life_hours: Hours for recency score to halve (24 default).
        max_access_boost: Maximum boost for high-access memories (0.1 default).
    """

    similarity: float = Field(
        default=0.50,
        ge=0.0,
        le=1.0,
        description="Weight for vector similarity score",
    )
    recency: float = Field(
        default=0.20,
        ge=0.0,
        le=1.0,
        description="Weight for recency/time decay",
    )
    confidence: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Weight for memory confidence score",
    )
    session: float = Field(
        default=0.10,
        ge=0.0,
        le=1.0,
        description="Weight for same-session bonus",
    )
    access: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Weight for frequently accessed memories",
    )
    recency_half_life_hours: float = Field(
        default=24.0,
        ge=0.1,
        description="Hours for recency score to halve",
    )
    max_access_boost: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Maximum boost for high-access memories",
    )

    @model_validator(mode="after")
    def _warn_if_weights_not_normalized(self) -> "RerankWeights":
        """Warn if weights don't sum to approximately 1.0."""
        total = self.similarity + self.recency + self.confidence + self.session + self.access
        if abs(total - 1.0) > 0.01:
            warnings.warn(
                f"RerankWeights sum to {total:.3f}, expected ~1.0. "
                f"Scores may be outside [0, 1]. "
                f"Weights: similarity={self.similarity}, recency={self.recency}, "
                f"confidence={self.confidence}, session={self.session}, access={self.access}",
                UserWarning,
                stacklevel=2,
            )
            logger.warning(
                "RerankWeights sum to %.3f (expected ~1.0): "
                "similarity=%.2f, recency=%.2f, confidence=%.2f, session=%.2f, access=%.2f",
                total,
                self.similarity,
                self.recency,
                self.confidence,
                self.session,
                self.access,
            )
        return self


class ConfidenceWeights(BaseModel):
    """Configurable weights for composite confidence scoring.

    The confidence formula uses these weights:
        confidence = (
            extraction_base * extraction +
            corroboration_score * corroboration +
            recency_score * recency +
            verification_score * verification
        )

    Weights should sum to 1.0 for normalized scores.

    Attributes:
        extraction: Weight for extraction method (0.50 default).
        corroboration: Weight for supporting sources (0.25 default).
        recency: Weight for how recently confirmed (0.15 default).
        verification: Weight for format validation (0.10 default).
        decay_half_life_days: Days for confidence to halve without confirmation.
    """

    extraction: float = Field(
        default=0.50,
        ge=0.0,
        le=1.0,
        description="Weight for extraction method",
    )
    corroboration: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="Weight for number of supporting sources",
    )
    recency: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Weight for how recently confirmed",
    )
    verification: float = Field(
        default=0.10,
        ge=0.0,
        le=1.0,
        description="Weight for format/validity verification",
    )
    decay_half_life_days: int = Field(
        default=365,
        ge=1,
        description="Days for confidence to halve without confirmation",
    )
    contradiction_penalty: float = Field(
        default=0.10,
        ge=0.0,
        le=1.0,
        description="Fraction to reduce confidence per contradiction",
    )

    @model_validator(mode="after")
    def _warn_if_weights_not_normalized(self) -> "ConfidenceWeights":
        """Warn if weights don't sum to approximately 1.0."""
        total = self.extraction + self.corroboration + self.recency + self.verification
        if abs(total - 1.0) > 0.01:
            warnings.warn(
                f"ConfidenceWeights sum to {total:.3f}, expected ~1.0. "
                f"Confidence scores may be miscalibrated.",
                UserWarning,
                stacklevel=2,
            )
            logger.warning(
                "ConfidenceWeights sum to %.3f (expected ~1.0): "
                "extraction=%.2f, corroboration=%.2f, recency=%.2f, verification=%.2f",
                total,
                self.extraction,
                self.corroboration,
                self.recency,
                self.verification,
            )
        return self


class Settings(BaseSettings):
    """Engram configuration loaded from environment variables.

    All settings can be overridden via environment variables with
    the ENGRAM_ prefix. For example:
        ENGRAM_QDRANT_URL=http://localhost:6333
        ENGRAM_EMBEDDING_PROVIDER=fastembed

    Security Notes:
        - In production (ENGRAM_ENV=production), auth is enabled by default
        - Using the default secret key in production will raise an error
        - Disabling auth in production will log a warning
    """

    # Environment
    env: Literal["development", "production", "test"] = Field(
        default="development",
        description="Environment: development, production, or test",
    )

    # Storage
    qdrant_url: str = Field(
        default="http://localhost:6333",
        description="Qdrant connection URL",
    )
    qdrant_api_key: str | None = Field(
        default=None,
        description="Qdrant API key (for cloud)",
    )
    collection_prefix: str = Field(
        default="engram",
        description="Prefix for Qdrant collection names",
    )

    # Embeddings
    embedding_provider: Literal["openai", "fastembed"] = Field(
        default="openai",
        description="Embedding provider to use",
    )
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="Embedding model name",
    )
    openai_api_key: str | None = Field(
        default=None,
        description="OpenAI API key",
    )
    anthropic_api_key: str | None = Field(
        default=None,
        description="Anthropic API key",
    )

    # LLM for consolidation
    llm_provider: str = Field(
        default="openai",
        description="LLM provider for semantic extraction",
    )
    llm_model: str = Field(
        default="gpt-4o-mini",
        description="LLM model for consolidation",
    )
    consolidation_model: str = Field(
        default="openai:gpt-4o-mini",
        description="Full model spec for Pydantic AI",
    )

    # Durable execution
    durable_backend: Literal["inprocess", "dbos", "prefect"] = Field(
        default="inprocess",
        description=(
            "Workflow backend: 'inprocess' (no durability), 'dbos' (SQLite/PostgreSQL), "
            "or 'prefect' (flow orchestration)"
        ),
    )

    # Prefect settings (only used if durable_backend == "prefect")
    prefect_api_url: str | None = Field(
        default=None,
        description="Prefect API URL (optional, for Prefect server)",
    )

    # DBOS settings (only used if durable_backend == "dbos")
    database_url: str | None = Field(
        default=None,
        description="Database URL for DBOS (SQLite if not set)",
    )

    # Confidence
    confidence_weights: ConfidenceWeights = Field(
        default_factory=ConfidenceWeights,
        description="Weights for confidence scoring",
    )

    # Decay
    decay_archive_threshold: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Archive memories below this confidence",
    )
    decay_delete_threshold: float = Field(
        default=0.01,
        ge=0.0,
        le=1.0,
        description="Delete memories below this confidence",
    )

    # Consolidation
    high_importance_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Trigger immediate consolidation for episodes at or above this importance",
    )
    consolidation_max_facts_per_episode: int = Field(
        default=3,
        ge=1,
        le=10,
        description=(
            "Maximum facts to extract per episode during consolidation. "
            "Total fact cap = max(8, episode_count * this value), capped at 50."
        ),
    )
    consolidation_max_keywords: int = Field(
        default=30,
        ge=5,
        le=100,
        description="Maximum keywords to retain after map-reduce consolidation",
    )
    consolidation_max_context_memories: int = Field(
        default=25,
        ge=1,
        le=100,
        description="Maximum existing semantic memories to pass as LLM context during consolidation",
    )
    consolidation_dedup_threshold: float = Field(
        default=0.90,
        ge=0.5,
        le=1.0,
        description=(
            "Embedding similarity threshold for near-duplicate detection. "
            "Semantic memories above this threshold are merged instead of created."
        ),
    )
    consolidation_checkpoint_dir: str | None = Field(
        default=None,
        description=(
            "Directory for consolidation checkpoint files. When set, map-reduce "
            "consolidation persists each chunk result to disk as it completes, "
            "enabling resume after process crashes. Set to a writable path "
            "(e.g., '/tmp/engram-checkpoints' or '~/.engram/checkpoints')."
        ),
    )

    # Retrieval Strengthening (Testing Effect)
    retrieval_strengthening_enabled: bool = Field(
        default=True,
        description="Enable Testing Effect: strengthen memories on retrieval",
    )
    retrieval_strengthening_delta: float = Field(
        default=0.05,
        ge=0.0,
        le=0.5,
        description="Strength increase per retrieval (smaller than consolidation's 0.1)",
    )

    # Surprise-based Importance Scoring (Adaptive Compression)
    surprise_scoring_enabled: bool = Field(
        default=True,
        description="Enable surprise/novelty factor in importance calculation",
    )
    surprise_weight: float = Field(
        default=0.15,
        ge=0.0,
        le=0.5,
        description="Weight for surprise factor in importance (0.15 = 15%)",
    )
    surprise_search_limit: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of similar memories to check for surprise calculation",
    )

    # Context-aware Reranking
    rerank_enabled: bool = Field(
        default=True,
        description="Enable context-aware reranking of recall results",
    )
    rerank_weights: RerankWeights = Field(
        default_factory=RerankWeights,
        description="Weights for context-aware reranking signals",
    )

    # Logging
    log_level: str = Field(
        default="INFO",
        description="Logging level",
    )
    log_format: Literal["json", "text"] = Field(
        default="json",
        description="Log output format",
    )

    # Authentication
    auth_enabled: bool | None = Field(
        default=None,
        description=(
            "Enable Bearer token authentication. "
            "If not set, defaults to True in production, False otherwise."
        ),
    )
    auth_secret_key: str | None = Field(
        default=None,
        description=(
            "Secret key for token validation (HMAC). "
            "REQUIRED in production. In dev/test, a random key is generated if not set."
        ),
    )
    auth_algorithm: str = Field(
        default="HS256",
        description="JWT algorithm",
    )
    auth_token_expire_minutes: int = Field(
        default=60,
        ge=1,
        description="Token expiration time in minutes",
    )

    # Runtime-generated dev secret (not from env, generated at startup if needed)
    _runtime_dev_secret: str | None = None

    @model_validator(mode="after")
    def validate_decay_thresholds(self) -> "Settings":
        """Validate decay thresholds are ordered correctly.

        The archive threshold must be strictly greater than the delete threshold
        because memories should be archived before being deleted. Violating this
        invariant would skip the archive stage entirely.
        """
        if self.decay_delete_threshold >= self.decay_archive_threshold:
            raise ValueError(
                f"decay_delete_threshold ({self.decay_delete_threshold}) must be less than "
                f"decay_archive_threshold ({self.decay_archive_threshold}). "
                f"Memories should be archived before being deleted."
            )
        return self

    @model_validator(mode="after")
    def validate_security_settings(self) -> "Settings":
        """Validate security settings based on environment.

        - In production, a secret key MUST be explicitly provided
        - In dev/test, a random key is generated if not provided
        - In production, disabling auth logs a warning
        - Resolves auth_enabled default based on environment
        - Automatically syncs ENGRAM_OPENAI_API_KEY to OPENAI_API_KEY for Pydantic AI
        """
        is_production = self.env == "production"

        # Resolve auth_enabled default based on environment
        if self.auth_enabled is None:
            # In production, auth is enabled by default
            # In development/test, auth is disabled by default
            object.__setattr__(self, "auth_enabled", is_production)

        # Production security checks
        if is_production:
            # Fail-fast if no secret key in production
            if self.auth_secret_key is None:
                raise ValueError(
                    "ENGRAM_AUTH_SECRET_KEY must be set in production. "
                    'Generate one with: python -c "import secrets; print(secrets.token_hex(32))"'
                )

            # Warn if auth is explicitly disabled in production
            if not self.auth_enabled:
                warnings.warn(
                    "Authentication is disabled in production environment. "
                    "This is a security risk. Set ENGRAM_AUTH_ENABLED=true to enable.",
                    UserWarning,
                    stacklevel=2,
                )
                logger.warning("Authentication disabled in production - this is a security risk")
        else:
            # Dev/test: generate a random secret if not provided
            if self.auth_secret_key is None:
                object.__setattr__(self, "_runtime_dev_secret", _generate_dev_secret_key())
                logger.debug(
                    "Generated random auth secret for development (tokens invalid after restart)"
                )

        # Bidirectional sync between ENGRAM_OPENAI_API_KEY and OPENAI_API_KEY.
        # Users commonly have OPENAI_API_KEY set (standard convention) but not
        # ENGRAM_OPENAI_API_KEY. In Docker MCP configs, only env vars present in the
        # parent process get inherited, so we must accept either name.
        if not self.openai_api_key:
            fallback_key = os.environ.get("OPENAI_API_KEY")
            if fallback_key:
                object.__setattr__(self, "openai_api_key", fallback_key)
                logger.debug("Using OPENAI_API_KEY as fallback for ENGRAM_OPENAI_API_KEY")

        # Forward sync: ensure OPENAI_API_KEY is set for libraries that expect it
        # (Pydantic AI, OpenAI SDK, etc.)
        if self.openai_api_key and not os.environ.get("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = self.openai_api_key
            logger.debug("Synced ENGRAM_OPENAI_API_KEY to OPENAI_API_KEY")

        return self

    @property
    def is_auth_enabled(self) -> bool:
        """Get resolved auth_enabled value (always bool, never None)."""
        if self.auth_enabled is None:
            return self.env == "production"
        return self.auth_enabled

    @property
    def effective_auth_secret_key(self) -> str:
        """Get the effective secret key for authentication.

        In production, this is always the explicitly configured key.
        In dev/test, this returns the configured key if set, otherwise
        the runtime-generated random key.

        Returns:
            The secret key to use for token operations.

        Raises:
            ValueError: If no secret key is available (should not happen
                after validation).
        """
        if self.auth_secret_key is not None:
            return self.auth_secret_key
        if self._runtime_dev_secret is not None:
            return self._runtime_dev_secret
        # This should never happen after validation
        raise ValueError("No auth secret key available")

    # Rate Limiting
    rate_limit_enabled: bool = Field(
        default=False,
        description="Enable rate limiting",
    )
    rate_limit_encode: int = Field(
        default=100,
        ge=1,
        description="Max encode requests per minute per user",
    )
    rate_limit_recall: int = Field(
        default=200,
        ge=1,
        description="Max recall requests per minute per user",
    )
    rate_limit_default: int = Field(
        default=60,
        ge=1,
        description="Default rate limit per minute for other endpoints",
    )
    rate_limit_redis_url: str | None = Field(
        default=None,
        description="Redis URL for distributed rate limiting (e.g., redis://localhost:6379). "
        "If not set, uses in-memory rate limiting (not suitable for multi-instance deployments).",
    )

    # Batch Operations
    batch_encode_max_items: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Maximum items allowed in a single batch encode request",
    )

    # Storage Pagination
    storage_max_scroll_limit: int = Field(
        default=10000,
        ge=100,
        le=100000,
        description=(
            "Maximum records to fetch in a single scroll operation. "
            "This affects memory usage for list operations across collections. "
            "For users with very large memory stores (>10k per collection), "
            "reduce this limit to prevent OOM issues."
        ),
    )

    # CORS Configuration
    cors_enabled: bool = Field(
        default=True,
        description="Enable CORS middleware",
    )
    cors_allow_origins: list[str] = Field(
        default_factory=lambda: ["*"],
        description=(
            "List of allowed CORS origins. Use ['*'] for permissive mode (dev only). "
            "In production, specify exact origins like ['https://app.example.com']."
        ),
    )
    cors_allow_methods: list[str] = Field(
        default_factory=lambda: ["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
        description="Allowed HTTP methods for CORS requests",
    )
    cors_allow_headers: list[str] = Field(
        default_factory=lambda: ["*"],
        description="Allowed headers for CORS requests",
    )
    cors_allow_credentials: bool = Field(
        default=False,
        description=(
            "Allow credentials (cookies, auth headers) in CORS requests. "
            "Cannot be True when cors_allow_origins is ['*']."
        ),
    )
    cors_max_age: int = Field(
        default=600,
        ge=0,
        le=86400,
        description="Max age (seconds) for CORS preflight cache",
    )

    # Phone Extraction Configuration
    phone_default_region: str = Field(
        default="US",
        min_length=2,
        max_length=2,
        description=(
            "Default region for phone number parsing (ISO 3166-1 alpha-2 code). "
            "Used when parsing numbers without country codes."
        ),
    )

    # Working Memory Configuration
    working_memory_max_size: int = Field(
        default=1000,
        ge=10,
        le=10000,
        description=(
            "Maximum number of episodes to keep in working memory. "
            "When exceeded, oldest episodes are evicted (FIFO). "
            "Prevents unbounded memory growth in long sessions."
        ),
    )

    # Parallel Processing Configuration
    max_concurrent_llm_calls: int = Field(
        default=5,
        ge=1,
        le=50,
        description=(
            "Maximum concurrent LLM calls for parallel workflow processing. "
            "Controls the semaphore limit to prevent rate limiting. "
            "Increase for higher throughput, decrease to avoid API limits."
        ),
    )

    # Embedding Cache Configuration
    embedding_cache_enabled: bool = Field(
        default=True,
        description="Enable LRU cache for embeddings to prevent redundant computation",
    )
    embedding_cache_size: int = Field(
        default=1000,
        ge=0,
        le=100000,
        description="Maximum number of embeddings to cache (0 to disable cache)",
    )

    model_config = {
        "env_prefix": "ENGRAM_",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "env_nested_delimiter": "__",
    }

    def sync_openai_api_key(self) -> None:
        """Bidirectional sync between ENGRAM_OPENAI_API_KEY and OPENAI_API_KEY.

        NOTE: This is now called automatically at Settings initialization.
        You do not need to call this method manually. It remains for
        backwards compatibility.

        Accepts either ENGRAM_OPENAI_API_KEY or OPENAI_API_KEY. If only one
        is set, the other is populated so both the engram config and downstream
        libraries (Pydantic AI, OpenAI SDK) can find the key.
        """
        if not self.openai_api_key:
            fallback_key = os.environ.get("OPENAI_API_KEY")
            if fallback_key:
                object.__setattr__(self, "openai_api_key", fallback_key)
                logger.debug("Using OPENAI_API_KEY as fallback for ENGRAM_OPENAI_API_KEY")

        if self.openai_api_key and not os.environ.get("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = self.openai_api_key
            logger.debug("Synced ENGRAM_OPENAI_API_KEY to OPENAI_API_KEY")


# Global settings instance
settings = Settings()
