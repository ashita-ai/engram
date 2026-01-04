"""Configuration management for Engram."""

from typing import Literal

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Engram configuration loaded from environment variables."""

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None

    # Embedding
    embedding_provider: str = "openai"
    embedding_model: str = "text-embedding-3-small"

    # LLM for consolidation
    consolidation_model: str = "openai:gpt-4o-mini"

    # Durable execution
    durable_backend: Literal["dbos", "temporal"] = "dbos"

    # Temporal settings (only used if durable_backend == "temporal")
    temporal_address: str = "localhost:7233"
    temporal_namespace: str = "default"
    temporal_task_queue: str = "engram-consolidation"

    # DBOS settings (only used if durable_backend == "dbos")
    # Uses SQLite by default, set database_url for Postgres
    database_url: str | None = None

    # Decay
    decay_archive_threshold: float = 0.1
    decay_delete_threshold: float = 0.01

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
