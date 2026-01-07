"""Configuration management for Engram."""

from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


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

    def validate_weights_sum(self) -> bool:
        """Check if weights sum to approximately 1.0."""
        total = self.extraction + self.corroboration + self.recency + self.verification
        return abs(total - 1.0) < 0.01


class Settings(BaseSettings):
    """Engram configuration loaded from environment variables.

    All settings can be overridden via environment variables with
    the ENGRAM_ prefix. For example:
        ENGRAM_QDRANT_URL=http://localhost:6333
        ENGRAM_EMBEDDING_PROVIDER=fastembed
    """

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
    durable_backend: Literal["dbos", "temporal", "prefect"] = Field(
        default="dbos",
        description="Workflow backend for durable execution",
    )

    # Temporal settings (only used if durable_backend == "temporal")
    temporal_address: str = Field(
        default="localhost:7233",
        description="Temporal server address",
    )
    temporal_namespace: str = Field(
        default="default",
        description="Temporal namespace",
    )
    temporal_task_queue: str = Field(
        default="engram-consolidation",
        description="Temporal task queue name",
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

    # Logging
    log_level: str = Field(
        default="INFO",
        description="Logging level",
    )
    log_format: Literal["json", "text"] = Field(
        default="json",
        description="Log output format",
    )

    model_config = {
        "env_prefix": "ENGRAM_",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "env_nested_delimiter": "__",
    }


# Global settings instance
settings = Settings()
