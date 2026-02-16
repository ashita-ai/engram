"""Tests for authentication and rate limiting."""

import time
from unittest.mock import MagicMock

import pytest

from engram.api.auth import (
    AuthenticatedUser,
    InMemoryRateLimiter,
    RateLimitInfo,
    TokenValidator,
    extract_client_ip,
    reset_auth_singletons,
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
    """Tests for in-memory rate limiting."""

    @pytest.fixture
    def limiter(self):
        """Create an in-memory rate limiter with short window for testing."""
        return InMemoryRateLimiter(window_seconds=1)

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
    """Tests for singleton behavior using lru_cache."""

    def setup_method(self):
        """Reset singletons before each test."""
        reset_auth_singletons()

    def test_reset_auth_singletons(self):
        """reset_auth_singletons should clear the cached instances."""
        from engram.api.auth import get_rate_limiter

        limiter1 = get_rate_limiter()
        reset_auth_singletons()
        limiter2 = get_rate_limiter()

        # Should be different instances after cache clear
        assert limiter1 is not limiter2

    def test_token_validator_singleton(self):
        """get_token_validator should return same instance for same key."""
        from engram.api.auth import get_token_validator

        validator1 = get_token_validator("secret-key")
        validator2 = get_token_validator("secret-key")

        # Should be same instance (cached)
        assert validator1 is validator2

    def test_token_validator_different_keys(self):
        """get_token_validator should return different instances for different keys."""
        from engram.api.auth import get_token_validator

        # Clear cache first
        reset_auth_singletons()

        validator1 = get_token_validator("secret-key-1")
        validator2 = get_token_validator("secret-key-2")

        # Should be different instances (different cache keys)
        assert validator1 is not validator2

    def test_get_rate_limiter_returns_inmemory_by_default(self):
        """get_rate_limiter should return InMemoryRateLimiter when no redis_url."""
        from engram.api.auth import get_rate_limiter

        reset_auth_singletons()
        limiter = get_rate_limiter()

        assert isinstance(limiter, InMemoryRateLimiter)

    def test_get_rate_limiter_returns_redis_when_url_provided(self):
        """get_rate_limiter should return RedisRateLimiter when redis_url is provided."""
        from engram.api.auth import REDIS_AVAILABLE, RedisRateLimiter, get_rate_limiter

        if not REDIS_AVAILABLE:
            pytest.skip("Redis not installed")

        import redis as redis_lib

        reset_auth_singletons()

        # This will fail if Redis isn't running, but tests the type selection
        try:
            limiter = get_rate_limiter(redis_url="redis://localhost:6379")
            assert isinstance(limiter, RedisRateLimiter)
        except (RuntimeError, redis_lib.exceptions.ConnectionError) as e:
            if "Failed to connect to Redis" in str(e) or "Connection refused" in str(e):
                pytest.skip("Redis server not running")
            raise


class TestRedisRateLimiter:
    """Tests for Redis-based rate limiting.

    These tests require Redis to be running. They are skipped if Redis
    is not available or not running.
    """

    @pytest.fixture
    def redis_limiter(self):
        """Create a Redis rate limiter with short window for testing."""
        from engram.api.auth import REDIS_AVAILABLE, RedisRateLimiter

        if not REDIS_AVAILABLE:
            pytest.skip("Redis not installed")

        import redis as redis_lib

        try:
            limiter = RedisRateLimiter(redis_url="redis://localhost:6379", window_seconds=1)
            # Clean up any existing test keys
            limiter._redis.delete(limiter._get_key("test_user", "endpoint"))
            return limiter
        except (RuntimeError, redis_lib.exceptions.ConnectionError) as e:
            if "Failed to connect to Redis" in str(e) or "Connection refused" in str(e):
                pytest.skip("Redis server not running")
            raise

    def test_redis_allows_requests_under_limit(self, redis_limiter):
        """Should allow requests under the limit."""
        for i in range(5):
            info = redis_limiter.check_rate_limit("test_user", "endpoint", limit=10)
            assert info.remaining == 10 - i - 1

    def test_redis_rejects_requests_over_limit(self, redis_limiter):
        """Should reject requests over the limit."""
        # Make 5 requests (the limit)
        for _ in range(5):
            redis_limiter.check_rate_limit("test_user", "endpoint", limit=5)

        # Next request should fail
        with pytest.raises(RateLimitError) as exc_info:
            redis_limiter.check_rate_limit("test_user", "endpoint", limit=5)

        assert exc_info.value.retry_after >= 0

    def test_redis_different_users_have_separate_limits(self, redis_limiter):
        """Each user should have their own limit."""
        # Clean up keys
        redis_limiter._redis.delete(redis_limiter._get_key("user_1", "endpoint"))
        redis_limiter._redis.delete(redis_limiter._get_key("user_2", "endpoint"))

        # User 1 hits limit
        for _ in range(5):
            redis_limiter.check_rate_limit("user_1", "endpoint", limit=5)

        # User 2 should still be able to make requests
        info = redis_limiter.check_rate_limit("user_2", "endpoint", limit=5)
        assert info.remaining == 4

    def test_redis_window_resets_after_time(self, redis_limiter):
        """Rate limit should reset after window expires."""
        # Clean up key
        redis_limiter._redis.delete(redis_limiter._get_key("test_user", "endpoint"))

        # Hit the limit
        for _ in range(5):
            redis_limiter.check_rate_limit("test_user", "endpoint", limit=5)

        # Wait for window to expire
        time.sleep(1.1)

        # Should be able to make requests again
        info = redis_limiter.check_rate_limit("test_user", "endpoint", limit=5)
        assert info.remaining == 4


def _make_request(
    *,
    client_host: str | None = "127.0.0.1",
    headers: dict[str, str] | None = None,
) -> MagicMock:
    """Build a mock FastAPI Request with configurable client and headers."""
    request = MagicMock()
    if client_host is not None:
        request.client.host = client_host
    else:
        request.client = None
    request.headers = headers or {}
    return request


class TestExtractClientIp:
    """Tests for extract_client_ip — proxy-aware IP extraction."""

    def test_direct_connection_ip(self) -> None:
        """Uses request.client.host when proxy headers are not trusted."""
        request = _make_request(client_host="10.0.0.5")
        assert extract_client_ip(request) == "ip:10.0.0.5"

    def test_no_client_returns_unknown(self) -> None:
        """Falls back to 'ip:unknown' when request.client is None."""
        request = _make_request(client_host=None)
        assert extract_client_ip(request) == "ip:unknown"

    def test_ignores_proxy_headers_when_not_trusted(self) -> None:
        """X-Forwarded-For is ignored when trust_proxy_headers=False."""
        request = _make_request(
            client_host="10.0.0.1",
            headers={"x-forwarded-for": "203.0.113.50, 10.0.0.1"},
        )
        result = extract_client_ip(request, trust_proxy_headers=False)
        assert result == "ip:10.0.0.1"

    def test_x_forwarded_for_trusted(self) -> None:
        """Uses leftmost X-Forwarded-For entry when trusted."""
        request = _make_request(
            client_host="10.0.0.1",
            headers={"x-forwarded-for": "203.0.113.50, 10.0.0.1"},
        )
        result = extract_client_ip(request, trust_proxy_headers=True)
        assert result == "ip:203.0.113.50"

    def test_x_forwarded_for_single_entry(self) -> None:
        """Handles single-entry X-Forwarded-For correctly."""
        request = _make_request(
            client_host="10.0.0.1",
            headers={"x-forwarded-for": "198.51.100.7"},
        )
        result = extract_client_ip(request, trust_proxy_headers=True)
        assert result == "ip:198.51.100.7"

    def test_x_real_ip_trusted(self) -> None:
        """Falls back to X-Real-IP when X-Forwarded-For is absent."""
        request = _make_request(
            client_host="10.0.0.1",
            headers={"x-real-ip": "203.0.113.99"},
        )
        result = extract_client_ip(request, trust_proxy_headers=True)
        assert result == "ip:203.0.113.99"

    def test_x_forwarded_for_takes_priority_over_x_real_ip(self) -> None:
        """X-Forwarded-For is preferred over X-Real-IP."""
        request = _make_request(
            client_host="10.0.0.1",
            headers={
                "x-forwarded-for": "203.0.113.50",
                "x-real-ip": "203.0.113.99",
            },
        )
        result = extract_client_ip(request, trust_proxy_headers=True)
        assert result == "ip:203.0.113.50"

    def test_empty_forwarded_for_falls_through(self) -> None:
        """Empty X-Forwarded-For falls back to next option."""
        request = _make_request(
            client_host="10.0.0.1",
            headers={"x-forwarded-for": ""},
        )
        result = extract_client_ip(request, trust_proxy_headers=True)
        assert result == "ip:10.0.0.1"

    def test_whitespace_only_x_real_ip_falls_through(self) -> None:
        """Whitespace-only X-Real-IP falls back to client.host."""
        request = _make_request(
            client_host="10.0.0.1",
            headers={"x-real-ip": "   "},
        )
        result = extract_client_ip(request, trust_proxy_headers=True)
        assert result == "ip:10.0.0.1"

    def test_ip_prefix_prevents_collision_with_user_ids(self) -> None:
        """The 'ip:' prefix ensures IP-based keys don't collide with user_id keys."""
        request = _make_request(client_host="user_123")
        result = extract_client_ip(request)
        assert result == "ip:user_123"
        assert result != "user_123"

    def test_trusted_fallback_to_client_host(self) -> None:
        """When proxy headers trusted but absent, falls back to client.host."""
        request = _make_request(client_host="192.168.1.100")
        result = extract_client_ip(request, trust_proxy_headers=True)
        assert result == "ip:192.168.1.100"

    def test_trusted_no_headers_no_client(self) -> None:
        """Returns 'ip:unknown' when everything is absent."""
        request = _make_request(client_host=None)
        result = extract_client_ip(request, trust_proxy_headers=True)
        assert result == "ip:unknown"
