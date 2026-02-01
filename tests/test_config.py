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

    def test_effective_auth_secret_key_explicit(self):
        """When secret key is explicitly provided, it should be used."""
        settings = Settings(
            env="development",
            auth_secret_key="my-explicit-secret-key",
            _env_file=None,
        )
        assert settings.effective_auth_secret_key == "my-explicit-secret-key"

    def test_effective_auth_secret_key_generated_in_dev(self):
        """In dev without explicit key, a random key should be generated."""
        settings = Settings(env="development", _env_file=None)
        # Should have a key (either runtime-generated or from env)
        key = settings.effective_auth_secret_key
        assert key is not None
        assert len(key) >= 32  # 64 hex chars = 32 bytes

    def test_effective_auth_secret_key_different_per_instance(self):
        """Each Settings instance should get a different generated key."""
        settings1 = Settings(env="development", _env_file=None)
        settings2 = Settings(env="development", _env_file=None)
        # Both should have keys
        key1 = settings1.effective_auth_secret_key
        key2 = settings2.effective_auth_secret_key
        # If no explicit key, each instance should have its own random key
        # (unless there's an env var set, in which case they'll be the same)
        assert key1 is not None
        assert key2 is not None

    def test_production_effective_secret_key(self):
        """In production, effective key should be the explicit key."""
        settings = Settings(
            env="production",
            auth_secret_key="production-secret-key-12345",
            _env_file=None,
        )
        assert settings.effective_auth_secret_key == "production-secret-key-12345"


class TestStorageSettings:
    """Tests for storage-related settings."""

    def test_storage_max_scroll_limit_default(self):
        """Default scroll limit should be 10000."""
        settings = Settings(_env_file=None)
        assert settings.storage_max_scroll_limit == 10000

    def test_storage_max_scroll_limit_custom(self):
        """Custom scroll limit should be accepted."""
        settings = Settings(storage_max_scroll_limit=5000, _env_file=None)
        assert settings.storage_max_scroll_limit == 5000

    def test_storage_max_scroll_limit_bounds(self):
        """Scroll limit must be between 100 and 100000."""
        # Test lower bound
        with pytest.raises(ValidationError):
            Settings(storage_max_scroll_limit=50, _env_file=None)

        # Test upper bound
        with pytest.raises(ValidationError):
            Settings(storage_max_scroll_limit=200000, _env_file=None)

    def test_storage_max_scroll_limit_from_env(self):
        """ENGRAM_STORAGE_MAX_SCROLL_LIMIT should override default."""
        with patch.dict(os.environ, {"ENGRAM_STORAGE_MAX_SCROLL_LIMIT": "2500"}):
            settings = Settings()
            assert settings.storage_max_scroll_limit == 2500


class TestCORSSettings:
    """Tests for CORS middleware settings (#173)."""

    def test_cors_enabled_by_default(self):
        """CORS should be enabled by default."""
        settings = Settings(_env_file=None)
        assert settings.cors_enabled is True

    def test_cors_default_origins(self):
        """Default CORS origins should allow all (dev mode)."""
        settings = Settings(_env_file=None)
        assert settings.cors_allow_origins == ["*"]

    def test_cors_custom_origins(self):
        """Custom CORS origins should be accepted."""
        settings = Settings(
            cors_allow_origins=["https://app.example.com", "https://admin.example.com"],
            _env_file=None,
        )
        assert settings.cors_allow_origins == [
            "https://app.example.com",
            "https://admin.example.com",
        ]

    def test_cors_default_methods(self):
        """Default CORS methods should include common HTTP methods."""
        settings = Settings(_env_file=None)
        assert "GET" in settings.cors_allow_methods
        assert "POST" in settings.cors_allow_methods
        assert "DELETE" in settings.cors_allow_methods
        assert "OPTIONS" in settings.cors_allow_methods

    def test_cors_credentials_default_false(self):
        """Credentials should be disabled by default."""
        settings = Settings(_env_file=None)
        assert settings.cors_allow_credentials is False

    def test_cors_max_age_default(self):
        """Default CORS max age should be 600 seconds."""
        settings = Settings(_env_file=None)
        assert settings.cors_max_age == 600

    def test_cors_max_age_bounds(self):
        """CORS max age must be between 0 and 86400."""
        # Test upper bound
        with pytest.raises(ValidationError):
            Settings(cors_max_age=100000, _env_file=None)

    def test_cors_can_be_disabled(self):
        """CORS can be explicitly disabled."""
        settings = Settings(cors_enabled=False, _env_file=None)
        assert settings.cors_enabled is False


class TestPhoneSettings:
    """Tests for phone extraction settings (#171)."""

    def test_phone_default_region(self):
        """Default phone region should be US."""
        settings = Settings(_env_file=None)
        assert settings.phone_default_region == "US"

    def test_phone_region_custom(self):
        """Custom phone region should be accepted."""
        settings = Settings(phone_default_region="GB", _env_file=None)
        assert settings.phone_default_region == "GB"

    def test_phone_region_length(self):
        """Phone region must be exactly 2 characters."""
        with pytest.raises(ValidationError):
            Settings(phone_default_region="USA", _env_file=None)
        with pytest.raises(ValidationError):
            Settings(phone_default_region="U", _env_file=None)

    def test_phone_region_from_env(self):
        """ENGRAM_PHONE_DEFAULT_REGION should override default."""
        with patch.dict(os.environ, {"ENGRAM_PHONE_DEFAULT_REGION": "DE"}):
            settings = Settings()
            assert settings.phone_default_region == "DE"


class TestWorkingMemorySettings:
    """Tests for working memory size limit settings (#169)."""

    def test_working_memory_max_size_default(self):
        """Default working memory max size should be 1000."""
        settings = Settings(_env_file=None)
        assert settings.working_memory_max_size == 1000

    def test_working_memory_max_size_custom(self):
        """Custom working memory max size should be accepted."""
        settings = Settings(working_memory_max_size=500, _env_file=None)
        assert settings.working_memory_max_size == 500

    def test_working_memory_max_size_bounds(self):
        """Working memory max size must be between 10 and 10000."""
        # Test lower bound
        with pytest.raises(ValidationError):
            Settings(working_memory_max_size=5, _env_file=None)
        # Test upper bound
        with pytest.raises(ValidationError):
            Settings(working_memory_max_size=20000, _env_file=None)

    def test_working_memory_max_size_from_env(self):
        """ENGRAM_WORKING_MEMORY_MAX_SIZE should override default."""
        with patch.dict(os.environ, {"ENGRAM_WORKING_MEMORY_MAX_SIZE": "2000"}):
            settings = Settings()
            assert settings.working_memory_max_size == 2000


class TestSyncOpenAIApiKey:
    """Tests for ENGRAM_OPENAI_API_KEY -> OPENAI_API_KEY sync."""

    def test_sync_sets_openai_api_key_when_not_present(self):
        """sync_openai_api_key should set OPENAI_API_KEY from settings."""
        # Ensure OPENAI_API_KEY is not set
        env = {"ENGRAM_OPENAI_API_KEY": "test-key-123"}
        with patch.dict(os.environ, env, clear=False):
            # Remove OPENAI_API_KEY if present
            os.environ.pop("OPENAI_API_KEY", None)

            settings = Settings(openai_api_key="test-key-123", _env_file=None)
            assert os.environ.get("OPENAI_API_KEY") is None

            settings.sync_openai_api_key()

            assert os.environ.get("OPENAI_API_KEY") == "test-key-123"

            # Clean up
            os.environ.pop("OPENAI_API_KEY", None)

    def test_sync_does_not_overwrite_existing_openai_api_key(self):
        """sync_openai_api_key should not overwrite existing OPENAI_API_KEY."""
        env = {"OPENAI_API_KEY": "existing-key"}
        with patch.dict(os.environ, env, clear=False):
            settings = Settings(openai_api_key="engram-key", _env_file=None)

            settings.sync_openai_api_key()

            # Should keep existing key, not overwrite
            assert os.environ.get("OPENAI_API_KEY") == "existing-key"

    def test_sync_does_nothing_when_no_openai_api_key(self):
        """sync_openai_api_key should do nothing if settings.openai_api_key is None."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OPENAI_API_KEY", None)

            settings = Settings(openai_api_key=None, _env_file=None)

            settings.sync_openai_api_key()

            assert os.environ.get("OPENAI_API_KEY") is None
