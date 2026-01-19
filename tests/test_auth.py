"""Tests for authentication and rate limiting."""

import time

import pytest

from engram.api.auth import (
    AuthenticatedUser,
    RateLimiter,
    RateLimitInfo,
    TokenValidator,
    reset_rate_limiter,
)
from engram.exceptions import AuthenticationError, RateLimitError


class TestTokenValidator:
    """Tests for token creation and validation."""

    @pytest.fixture
    def validator(self):
        """Create a token validator with test secret."""
        return TokenValidator("test-secret-key")

    def test_create_token(self, validator):
        """Should create a signed token."""
        token = validator.create_token("user_123")
        assert token is not None
        assert "user_123" in token
        # Token format: user_id:org_id:expires_at:signature
        parts = token.split(":")
        assert len(parts) == 4

    def test_create_token_with_org(self, validator):
        """Should include org_id in token."""
        token = validator.create_token("user_123", org_id="org_456")
        assert "org_456" in token

    def test_validate_token_success(self, validator):
        """Should validate a valid token."""
        token = validator.create_token("user_123", org_id="org_456")
        user = validator.validate_token(token)

        assert isinstance(user, AuthenticatedUser)
        assert user.user_id == "user_123"
        assert user.org_id == "org_456"

    def test_validate_token_no_org(self, validator):
        """Should handle token without org_id."""
        token = validator.create_token("user_123")
        user = validator.validate_token(token)

        assert user.user_id == "user_123"
        assert user.org_id is None

    def test_validate_token_invalid_format(self, validator):
        """Should reject invalid token format."""
        with pytest.raises(AuthenticationError, match="Invalid token format"):
            validator.validate_token("invalid-token")

    def test_validate_token_invalid_signature(self, validator):
        """Should reject token with invalid signature."""
        token = validator.create_token("user_123")
        # Tamper with the token
        parts = token.split(":")
        parts[3] = "invalid_signature"
        tampered = ":".join(parts)

        with pytest.raises(AuthenticationError, match="Invalid token signature"):
            validator.validate_token(tampered)

    def test_validate_token_expired(self, validator):
        """Should reject expired token."""
        # Create token with 0 minutes expiry (already expired)
        token = validator.create_token("user_123", expire_minutes=0)

        # Wait a tiny bit to ensure expiration
        time.sleep(0.1)

        with pytest.raises(AuthenticationError, match="Token has expired"):
            validator.validate_token(token)

    def test_different_secrets_produce_different_tokens(self):
        """Tokens from different secrets should be incompatible."""
        validator1 = TokenValidator("secret1")
        validator2 = TokenValidator("secret2")

        token = validator1.create_token("user_123")

        with pytest.raises(AuthenticationError):
            validator2.validate_token(token)


class TestRateLimiter:
    """Tests for rate limiting."""

    @pytest.fixture
    def limiter(self):
        """Create a rate limiter with short window for testing."""
        return RateLimiter(window_seconds=1)

    def test_allows_requests_under_limit(self, limiter):
        """Should allow requests under the limit."""
        for i in range(5):
            info = limiter.check_rate_limit("user_1", "endpoint", limit=10)
            assert info.remaining == 10 - i - 1

    def test_rejects_requests_over_limit(self, limiter):
        """Should reject requests over the limit."""
        # Make 5 requests (the limit)
        for _ in range(5):
            limiter.check_rate_limit("user_1", "endpoint", limit=5)

        # Next request should fail
        with pytest.raises(RateLimitError) as exc_info:
            limiter.check_rate_limit("user_1", "endpoint", limit=5)

        # retry_after can be 0 or 1 due to rounding with 1-second window
        assert exc_info.value.retry_after >= 0

    def test_different_users_have_separate_limits(self, limiter):
        """Each user should have their own limit."""
        # User 1 hits limit
        for _ in range(5):
            limiter.check_rate_limit("user_1", "endpoint", limit=5)

        # User 2 should still be able to make requests
        info = limiter.check_rate_limit("user_2", "endpoint", limit=5)
        assert info.remaining == 4

    def test_different_endpoints_have_separate_limits(self, limiter):
        """Each endpoint should have its own limit per user."""
        # Hit limit on endpoint1
        for _ in range(5):
            limiter.check_rate_limit("user_1", "endpoint1", limit=5)

        # endpoint2 should still work
        info = limiter.check_rate_limit("user_1", "endpoint2", limit=5)
        assert info.remaining == 4

    def test_window_resets_after_time(self, limiter):
        """Rate limit should reset after window expires."""
        # Hit the limit
        for _ in range(5):
            limiter.check_rate_limit("user_1", "endpoint", limit=5)

        # Wait for window to expire
        time.sleep(1.1)

        # Should be able to make requests again
        info = limiter.check_rate_limit("user_1", "endpoint", limit=5)
        assert info.remaining == 4

    def test_rate_limit_info_fields(self, limiter):
        """RateLimitInfo should have correct fields."""
        info = limiter.check_rate_limit("user_1", "endpoint", limit=100)

        assert isinstance(info, RateLimitInfo)
        assert info.limit == 100
        assert info.remaining == 99
        assert info.reset_at > int(time.time())


class TestAuthenticatedUser:
    """Tests for AuthenticatedUser model."""

    def test_user_with_all_fields(self):
        """Should create user with all fields."""
        user = AuthenticatedUser(
            user_id="user_123",
            org_id="org_456",
            scopes=["read", "write"],
        )
        assert user.user_id == "user_123"
        assert user.org_id == "org_456"
        assert user.scopes == ["read", "write"]

    def test_user_with_defaults(self):
        """Should create user with default values."""
        user = AuthenticatedUser(user_id="user_123")
        assert user.user_id == "user_123"
        assert user.org_id is None
        assert user.scopes == []


class TestGlobalSingletons:
    """Tests for global singleton behavior."""

    def setup_method(self):
        """Reset singletons before each test."""
        reset_rate_limiter()

    def test_reset_rate_limiter(self):
        """reset_rate_limiter should clear the singleton."""
        from engram.api.auth import get_rate_limiter

        limiter1 = get_rate_limiter()
        reset_rate_limiter()
        limiter2 = get_rate_limiter()

        # Should be different instances
        assert limiter1 is not limiter2
