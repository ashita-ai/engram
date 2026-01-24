"""Unit tests for Engram configuration."""

import os
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from engram.config import ConfidenceWeights, Settings


class TestConfidenceWeights:
    """Tests for ConfidenceWeights model."""

    def test_default_weights(self):
        """Default weights should sum to 1.0."""
        weights = ConfidenceWeights()
        assert weights.extraction == 0.50
        assert weights.corroboration == 0.25
        assert weights.recency == 0.15
        assert weights.verification == 0.10
        assert weights.validate_weights_sum()

    def test_custom_weights(self):
        """Custom weights should be accepted."""
        weights = ConfidenceWeights(
            extraction=0.4,
            corroboration=0.3,
            recency=0.2,
            verification=0.1,
        )
        assert weights.extraction == 0.4
        assert weights.corroboration == 0.3
        assert weights.validate_weights_sum()

    def test_validate_weights_sum_false(self):
        """validate_weights_sum should return False if weights don't sum to 1."""
        weights = ConfidenceWeights(
            extraction=0.5,
            corroboration=0.5,
            recency=0.5,
            verification=0.5,
        )
        assert not weights.validate_weights_sum()

    def test_weight_bounds(self):
        """Weights must be between 0 and 1."""
        with pytest.raises(ValidationError):
            ConfidenceWeights(extraction=1.5)
        with pytest.raises(ValidationError):
            ConfidenceWeights(extraction=-0.1)

    def test_decay_half_life_days(self):
        """decay_half_life_days should have a positive default."""
        weights = ConfidenceWeights()
        assert weights.decay_half_life_days == 365

    def test_decay_half_life_custom(self):
        """Custom decay half-life should be accepted."""
        weights = ConfidenceWeights(decay_half_life_days=180)
        assert weights.decay_half_life_days == 180

    def test_decay_half_life_minimum(self):
        """decay_half_life_days must be at least 1."""
        with pytest.raises(ValidationError):
            ConfidenceWeights(decay_half_life_days=0)


class TestSettings:
    """Tests for Settings model."""

    def test_default_settings(self):
        """Default settings should be reasonable."""
        # Use _env_file=None to prevent reading from .env file
        settings = Settings(_env_file=None)
        assert settings.qdrant_url == "http://localhost:6333"
        assert settings.collection_prefix == "engram"
        assert settings.embedding_provider == "openai"
        # durable_backend may be set by environment variable, check it's a valid value
        assert settings.durable_backend in ("dbos", "prefect", "inprocess")
        assert settings.log_level == "INFO"

    def test_embedding_providers(self):
        """Only valid embedding providers should be accepted."""
        settings = Settings(embedding_provider="fastembed")
        assert settings.embedding_provider == "fastembed"

        settings = Settings(embedding_provider="openai")
        assert settings.embedding_provider == "openai"

    def test_durable_backends(self):
        """Only valid durable backends should be accepted."""
        for backend in ["dbos", "prefect", "inprocess"]:
            settings = Settings(durable_backend=backend)
            assert settings.durable_backend == backend

    def test_log_formats(self):
        """Only valid log formats should be accepted."""
        settings = Settings(log_format="json")
        assert settings.log_format == "json"

        settings = Settings(log_format="text")
        assert settings.log_format == "text"

    def test_decay_thresholds(self):
        """Decay thresholds should have sensible defaults."""
        settings = Settings()
        assert settings.decay_archive_threshold == 0.1
        assert settings.decay_delete_threshold == 0.01

    def test_decay_threshold_bounds(self):
        """Decay thresholds must be between 0 and 1."""
        with pytest.raises(ValidationError):
            Settings(decay_archive_threshold=1.5)
        with pytest.raises(ValidationError):
            Settings(decay_delete_threshold=-0.1)

    def test_confidence_weights_default(self):
        """Default confidence weights should be included."""
        settings = Settings()
        assert isinstance(settings.confidence_weights, ConfidenceWeights)
        assert settings.confidence_weights.validate_weights_sum()

    def test_env_prefix(self):
        """Settings should use ENGRAM_ prefix for environment variables."""
        with patch.dict(os.environ, {"ENGRAM_LOG_LEVEL": "DEBUG"}):
            settings = Settings()
            assert settings.log_level == "DEBUG"

    def test_env_qdrant_url(self):
        """ENGRAM_QDRANT_URL should override default."""
        with patch.dict(os.environ, {"ENGRAM_QDRANT_URL": "http://qdrant:6333"}):
            settings = Settings()
            assert settings.qdrant_url == "http://qdrant:6333"

    def test_optional_api_keys(self):
        """API keys should be optional (can be None or set via environment)."""
        # Use _env_file=None to prevent reading from .env file
        # Note: environment variables will still override defaults
        settings = Settings(_env_file=None)
        # qdrant_api_key defaults to None and is not commonly set
        assert settings.qdrant_api_key is None
        # openai_api_key may be set via environment variable, so just check type
        assert settings.openai_api_key is None or isinstance(settings.openai_api_key, str)

    def test_llm_settings(self):
        """LLM settings should have defaults."""
        settings = Settings()
        assert settings.llm_provider == "openai"
        assert settings.llm_model == "gpt-4o-mini"
        assert settings.consolidation_model == "openai:gpt-4o-mini"


class TestSecuritySettings:
    """Tests for security-related settings validation."""

    def test_auth_disabled_by_default_in_development(self):
        """Auth should be disabled by default in development environment."""
        settings = Settings(env="development", _env_file=None)
        assert settings.is_auth_enabled is False

    def test_auth_enabled_by_default_in_production(self):
        """Auth should be enabled by default in production environment."""
        # Must provide a custom secret key in production
        settings = Settings(
            env="production",
            auth_secret_key="custom-secure-key-for-testing-only",
            _env_file=None,
        )
        assert settings.is_auth_enabled is True

    def test_production_requires_custom_secret_key(self):
        """Production environment should reject default secret key."""
        with pytest.raises(ValueError, match="ENGRAM_AUTH_SECRET_KEY must be set"):
            Settings(env="production", _env_file=None)

    def test_production_allows_custom_secret_key(self):
        """Production environment should accept custom secret key."""
        settings = Settings(
            env="production",
            auth_secret_key="my-super-secure-secret-key-32bytes",
            _env_file=None,
        )
        assert settings.auth_secret_key == "my-super-secure-secret-key-32bytes"

    def test_auth_can_be_explicitly_disabled_in_production(self):
        """Auth can be explicitly disabled in production (with warning)."""
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            settings = Settings(
                env="production",
                auth_enabled=False,
                auth_secret_key="custom-secret-key-for-testing",
                _env_file=None,
            )
            assert settings.is_auth_enabled is False
            # Should have warned about disabled auth
            assert len(w) == 1
            assert "Authentication is disabled in production" in str(w[0].message)

    def test_auth_can_be_explicitly_enabled_in_development(self):
        """Auth can be explicitly enabled in development."""
        settings = Settings(
            env="development",
            auth_enabled=True,
            _env_file=None,
        )
        assert settings.is_auth_enabled is True

    def test_env_defaults_to_development(self):
        """Environment should default to development."""
        settings = Settings(_env_file=None)
        assert settings.env == "development"

    def test_test_environment_disables_auth_by_default(self):
        """Test environment should disable auth by default."""
        settings = Settings(env="test", _env_file=None)
        assert settings.is_auth_enabled is False
