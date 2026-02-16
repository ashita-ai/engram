"""Authentication and rate limiting for Engram API.

Provides:
- Bearer token authentication with JWT
- Per-user rate limiting with in-memory or Redis-based tracking
- FastAPI dependencies for route protection
"""

from __future__ import annotations

import hashlib
import hmac
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from functools import lru_cache
from typing import TYPE_CHECKING, Annotated

from fastapi import Depends, Request, Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict, Field

from engram.exceptions import AuthenticationError, RateLimitError
from engram.logging import get_logger

if TYPE_CHECKING:
    from engram.config import Settings

logger = get_logger(__name__)

# Track if Redis is available (optional dependency)
try:
    import redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

# Security scheme for OpenAPI docs
security = HTTPBearer(auto_error=False)


class AuthenticatedUser(BaseModel):
    """Represents an authenticated user.

    Attributes:
        user_id: Unique identifier for the user.
        org_id: Optional organization ID.
        scopes: List of permission scopes.
    """

    model_config = ConfigDict(extra="forbid")

    user_id: str = Field(description="Unique identifier for the user")
    org_id: str | None = Field(default=None, description="Optional organization ID")
    scopes: list[str] = Field(default_factory=list, description="Permission scopes")


class RateLimitInfo(BaseModel):
    """Rate limit tracking for a user/endpoint combination.

    Attributes:
        limit: Maximum requests allowed per window.
        remaining: Requests remaining in current window.
        reset_at: Unix timestamp when the window resets.
    """

    model_config = ConfigDict(extra="forbid")

    limit: int = Field(ge=0, description="Maximum requests allowed per window")
    remaining: int = Field(ge=0, description="Requests remaining in current window")
    reset_at: int = Field(description="Unix timestamp when the window resets")


class TokenValidator:
    """Validates Bearer tokens using HMAC-SHA256.

    Simple token format: user_id:timestamp:signature
    where signature = HMAC(secret, user_id:timestamp)

    For production, replace with proper JWT validation.
    """

    def __init__(self, secret_key: str) -> None:
        self.secret_key = secret_key.encode()

    def create_token(
        self,
        user_id: str,
        org_id: str | None = None,
        expire_minutes: int = 60,
    ) -> str:
        """Create a signed token for a user.

        Args:
            user_id: User identifier.
            org_id: Optional organization ID (encoded in token).
            expire_minutes: Token validity in minutes.

        Returns:
            Signed token string.
        """
        expires_at = int(time.time()) + (expire_minutes * 60)
        # Include org_id in payload if present
        payload = f"{user_id}:{org_id or ''}:{expires_at}"
        signature = hmac.new(
            self.secret_key,
            payload.encode(),
            hashlib.sha256,
        ).hexdigest()
        return f"{payload}:{signature}"

    def validate_token(self, token: str) -> AuthenticatedUser:
        """Validate a token and return the authenticated user.

        Args:
            token: The token to validate.

        Returns:
            AuthenticatedUser with extracted user info.

        Raises:
            AuthenticationError: If token is invalid or expired.
        """
        try:
            parts = token.split(":")
            if len(parts) != 4:
                raise AuthenticationError("Invalid token format")

            user_id, org_id, expires_at_str, signature = parts
            payload = f"{user_id}:{org_id}:{expires_at_str}"

            # Verify signature
            expected_sig = hmac.new(
                self.secret_key,
                payload.encode(),
                hashlib.sha256,
            ).hexdigest()

            if not hmac.compare_digest(signature, expected_sig):
                raise AuthenticationError("Invalid token signature")

            # Check expiration
            expires_at = int(expires_at_str)
            if time.time() > expires_at:
                raise AuthenticationError("Token has expired")

            return AuthenticatedUser(
                user_id=user_id,
                org_id=org_id if org_id else None,
            )

        except (ValueError, IndexError) as e:
            raise AuthenticationError(f"Invalid token: {e}") from e


class RateLimiter(ABC):
    """Abstract base class for rate limiters.

    All rate limiters must implement check_rate_limit() which tracks
    request counts per user/endpoint with automatic window reset.
    """

    @abstractmethod
    def check_rate_limit(
        self,
        user_id: str,
        endpoint: str,
        limit: int,
    ) -> RateLimitInfo:
        """Check if request is within rate limit.

        Args:
            user_id: User making the request.
            endpoint: API endpoint being accessed.
            limit: Maximum requests per window.

        Returns:
            RateLimitInfo with current status.

        Raises:
            RateLimitError: If rate limit exceeded.
        """
        ...


class InMemoryRateLimiter(RateLimiter):
    """In-memory rate limiter using sliding window.

    Tracks request counts per user/endpoint with automatic window reset.
    Not suitable for multi-instance deployments - use RedisRateLimiter instead.
    """

    def __init__(self, window_seconds: int = 60) -> None:
        self.window_seconds = window_seconds
        # Structure: {user_id: {endpoint: [(timestamp, count)]}}
        self._requests: dict[str, dict[str, list[tuple[float, int]]]] = defaultdict(
            lambda: defaultdict(list)
        )

    def check_rate_limit(
        self,
        user_id: str,
        endpoint: str,
        limit: int,
    ) -> RateLimitInfo:
        """Check if request is within rate limit.

        Args:
            user_id: User making the request.
            endpoint: API endpoint being accessed.
            limit: Maximum requests per window.

        Returns:
            RateLimitInfo with current status.

        Raises:
            RateLimitError: If rate limit exceeded.
        """
        now = time.time()
        window_start = now - self.window_seconds
        reset_at = int(now) + self.window_seconds

        # Get and clean old entries
        user_requests = self._requests[user_id][endpoint]
        # Filter to only requests within the window
        user_requests[:] = [(ts, c) for ts, c in user_requests if ts > window_start]

        # Count requests in window
        request_count = sum(c for _, c in user_requests)

        if request_count >= limit:
            retry_after = int(reset_at - now)
            logger.warning(
                "Rate limit exceeded",
                user_id=user_id,
                endpoint=endpoint,
                limit=limit,
                retry_after=retry_after,
            )
            raise RateLimitError(retry_after)

        # Record this request
        user_requests.append((now, 1))

        return RateLimitInfo(
            limit=limit,
            remaining=limit - request_count - 1,
            reset_at=reset_at,
        )


class RedisRateLimiter(RateLimiter):
    """Redis-based rate limiter using sliding window with atomic Lua script.

    Uses Redis for distributed rate limiting across multiple instances.
    Requires the 'redis' extra: uv sync --extra redis

    Uses a sorted set per user/endpoint with timestamps as scores.
    A Lua script ensures atomic check-and-increment to prevent race conditions.
    """

    # Lua script for atomic rate limit check-and-increment
    # This prevents the TOCTOU race condition that existed with separate pipeline calls
    _RATE_LIMIT_SCRIPT = """
    local key = KEYS[1]
    local limit = tonumber(ARGV[1])
    local window_start = tonumber(ARGV[2])
    local now = tonumber(ARGV[3])
    local member = ARGV[4]
    local ttl = tonumber(ARGV[5])

    -- Remove entries outside the window
    redis.call('ZREMRANGEBYSCORE', key, 0, window_start)

    -- Count current entries
    local count = redis.call('ZCARD', key)

    -- Check if limit exceeded BEFORE adding
    if count >= limit then
        return {0, count}  -- Rejected: 0 = not allowed
    end

    -- Add the new request atomically
    redis.call('ZADD', key, now, member)
    redis.call('EXPIRE', key, ttl)

    return {1, count}  -- Accepted: 1 = allowed, count = previous count
    """

    def __init__(self, redis_url: str, window_seconds: int = 60) -> None:
        if not REDIS_AVAILABLE:
            raise ImportError("Redis is not installed. Install with: uv sync --extra redis")

        self.window_seconds = window_seconds
        self._redis = redis.Redis.from_url(redis_url, decode_responses=True)
        self._key_prefix = "engram:ratelimit:"

        # Register the Lua script for atomic execution
        # Cast to str to satisfy mypy (script_load returns str for sync client)
        self._rate_limit_sha: str = str(self._redis.script_load(self._RATE_LIMIT_SCRIPT))

        # Test connection
        try:
            self._redis.ping()
        except redis.ConnectionError as e:
            raise RuntimeError(f"Failed to connect to Redis at {redis_url}: {e}") from e

        logger.info("Redis rate limiter initialized", redis_url=redis_url)

    def _get_key(self, user_id: str, endpoint: str) -> str:
        """Generate Redis key for user/endpoint combination."""
        return f"{self._key_prefix}{user_id}:{endpoint}"

    def check_rate_limit(
        self,
        user_id: str,
        endpoint: str,
        limit: int,
    ) -> RateLimitInfo:
        """Check if request is within rate limit using Redis.

        Uses a Lua script for atomic check-and-increment to prevent race conditions.
        The script:
        1. Removes entries outside the sliding window
        2. Counts current entries
        3. If under limit, adds the new request atomically
        4. Returns whether the request was allowed

        Args:
            user_id: User making the request.
            endpoint: API endpoint being accessed.
            limit: Maximum requests per window.

        Returns:
            RateLimitInfo with current status.

        Raises:
            RateLimitError: If rate limit exceeded.
        """
        now = time.time()
        window_start = now - self.window_seconds
        reset_at = int(now) + self.window_seconds
        key = self._get_key(user_id, endpoint)

        # Create unique member for this request
        member = f"{now}:{id(self)}:{hash(time.time_ns())}"
        ttl = self.window_seconds + 10  # TTL slightly longer than window

        # Execute atomic Lua script
        try:
            result = self._redis.evalsha(
                self._rate_limit_sha,
                1,  # Number of keys
                key,  # KEYS[1]
                limit,  # ARGV[1]
                window_start,  # ARGV[2]
                now,  # ARGV[3]
                member,  # ARGV[4]
                ttl,  # ARGV[5]
            )
        except redis.exceptions.NoScriptError:
            # Script was flushed from cache, reload it
            self._rate_limit_sha = str(self._redis.script_load(self._RATE_LIMIT_SCRIPT))
            result = self._redis.evalsha(
                self._rate_limit_sha,
                1,
                key,
                limit,
                window_start,
                now,
                member,
                ttl,
            )

        # Result is a list of [allowed (0/1), count]
        result_list = list(result)  # type: ignore[arg-type]
        allowed, request_count = int(result_list[0]), int(result_list[1])

        if not allowed:
            retry_after = int(reset_at - now)
            logger.warning(
                "Rate limit exceeded",
                user_id=user_id,
                endpoint=endpoint,
                limit=limit,
                retry_after=retry_after,
                backend="redis",
            )
            raise RateLimitError(retry_after)

        return RateLimitInfo(
            limit=limit,
            remaining=limit - request_count - 1,
            reset_at=reset_at,
        )


# Singleton instances using functools.lru_cache for thread-safe caching
# This approach avoids global mutable state while maintaining singleton behavior


@lru_cache(maxsize=1)
def get_token_validator(secret_key: str) -> TokenValidator:
    """Get or create the token validator singleton.

    Uses lru_cache for thread-safe singleton behavior.
    The secret_key parameter ensures a new validator is created if the key changes.

    Args:
        secret_key: The secret key for token validation.

    Returns:
        TokenValidator instance.
    """
    return TokenValidator(secret_key)


@lru_cache(maxsize=1)
def get_rate_limiter(redis_url: str | None = None, window_seconds: int = 60) -> RateLimiter:
    """Get or create the rate limiter singleton.

    Uses lru_cache for thread-safe singleton behavior.
    If redis_url is provided, uses RedisRateLimiter for distributed rate limiting.
    Otherwise, uses InMemoryRateLimiter (not suitable for multi-instance deployments).

    Args:
        redis_url: Optional Redis URL for distributed rate limiting.
        window_seconds: The sliding window size in seconds.

    Returns:
        RateLimiter instance (either InMemoryRateLimiter or RedisRateLimiter).
    """
    if redis_url:
        logger.info("Using Redis rate limiter", redis_url=redis_url)
        return RedisRateLimiter(redis_url, window_seconds)
    else:
        logger.info("Using in-memory rate limiter (not distributed)")
        return InMemoryRateLimiter(window_seconds)


def reset_auth_singletons() -> None:
    """Reset all auth singletons (for testing).

    Clears the lru_cache for both token validator and rate limiter,
    allowing new instances to be created with fresh state.
    """
    get_token_validator.cache_clear()
    get_rate_limiter.cache_clear()


class AuthDependency:
    """FastAPI dependency for authentication.

    Usage:
        @router.post("/encode")
        async def encode(
            request: EncodeRequest,
            user: AuthenticatedUser = Depends(AuthDependency(settings)),
        ):
            ...
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def __call__(
        self,
        request: Request,
        credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(security)],
    ) -> AuthenticatedUser | None:
        """Validate credentials and return authenticated user.

        If auth is disabled, returns None (endpoints use user_id from request body).
        If auth is enabled, validates the Bearer token.
        """
        if not self.settings.is_auth_enabled:
            return None

        if credentials is None:
            raise AuthenticationError("Missing authentication credentials")

        validator = get_token_validator(self.settings.effective_auth_secret_key)
        user = validator.validate_token(credentials.credentials)

        logger.debug(
            "User authenticated",
            user_id=user.user_id,
            org_id=user.org_id,
        )

        return user


def extract_client_ip(request: Request, *, trust_proxy_headers: bool = False) -> str:
    """Extract the best available client IP for rate limiting.

    When behind a reverse proxy, ``request.client.host`` is the proxy's IP,
    not the real client.  Enable *trust_proxy_headers* only when you control
    the proxy that sets ``X-Forwarded-For`` / ``X-Real-IP``.

    Priority (when trust_proxy_headers is True):
        1. X-Forwarded-For — leftmost entry (original client)
        2. X-Real-IP — typically set by nginx
        3. request.client.host — direct connection IP

    When trust_proxy_headers is False, only request.client.host is used.

    Returns:
        A string suitable as a rate-limit key, prefixed with ``ip:`` to
        prevent collisions with authenticated user_id values.
    """
    if trust_proxy_headers:
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            # Format: "client, proxy1, proxy2" — first entry is the original client
            client_ip = forwarded_for.split(",")[0].strip()
            if client_ip:
                return f"ip:{client_ip}"

        real_ip = request.headers.get("x-real-ip")
        if real_ip and real_ip.strip():
            return f"ip:{real_ip.strip()}"

    # Direct connection
    if request.client and request.client.host:
        return f"ip:{request.client.host}"

    return "ip:unknown"


class RateLimitDependency:
    """FastAPI dependency for rate limiting.

    Usage:
        @router.post("/encode")
        async def encode(
            request: EncodeRequest,
            rate_info: RateLimitInfo = Depends(RateLimitDependency(settings, "encode", 100)),
            user: AuthenticatedUser | None = Depends(auth),
        ):
            ...
    """

    def __init__(self, settings: Settings, endpoint: str, limit: int | None = None) -> None:
        self.settings = settings
        self.endpoint = endpoint
        self.limit = limit

    async def __call__(
        self,
        request: Request,
        user: Annotated[AuthenticatedUser | None, Depends(AuthDependency)],
    ) -> RateLimitInfo | None:
        """Check rate limit for the current request.

        If rate limiting is disabled, returns None.
        Uses authenticated user_id if available, otherwise extracts client IP.
        """
        if not self.settings.rate_limit_enabled:
            return None

        # Get user_id from auth or extract client IP
        if user is not None:
            user_id = user.user_id
        else:
            user_id = extract_client_ip(
                request,
                trust_proxy_headers=self.settings.rate_limit_trust_proxy_headers,
            )

        # Use configured limit or endpoint-specific default
        limit = self.limit
        if limit is None:
            if self.endpoint == "encode":
                limit = self.settings.rate_limit_encode
            elif self.endpoint == "recall":
                limit = self.settings.rate_limit_recall
            else:
                limit = self.settings.rate_limit_default

        limiter = get_rate_limiter(redis_url=self.settings.rate_limit_redis_url)
        return limiter.check_rate_limit(user_id, self.endpoint, limit)


def add_rate_limit_headers(
    response: Response,
    rate_info: RateLimitInfo | None,
) -> None:
    """Add rate limit headers to response.

    Args:
        response: FastAPI Response object.
        rate_info: Rate limit info (None if disabled).
    """
    if rate_info is None:
        return

    response.headers["X-RateLimit-Limit"] = str(rate_info.limit)
    response.headers["X-RateLimit-Remaining"] = str(rate_info.remaining)
    response.headers["X-RateLimit-Reset"] = str(rate_info.reset_at)
