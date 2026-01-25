"""FastAPI application for Engram."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from engram.config import Settings
from engram.exceptions import (
    AuthenticationError,
    AuthorizationError,
    EngramError,
    NotFoundError,
    RateLimitError,
    ValidationError,
)
from engram.logging import configure_logging, get_logger
from engram.service import EngramService
from engram.workflows import init_workflows, shutdown_workflows

from .router import router, set_service

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage application lifespan.

    Initializes the EngramService and DBOS workflows on startup,
    and cleans up on shutdown.
    """
    settings = Settings()

    # Configure structured logging
    configure_logging(level=settings.log_level, format=settings.log_format)
    logger.info("Starting Engram API", log_level=settings.log_level, log_format=settings.log_format)

    service = EngramService.create(settings)

    # Initialize storage and set service
    await service.initialize()
    set_service(service)

    # Initialize durable workflows (DBOS or Temporal based on config)
    try:
        init_workflows()  # Sync call - configures and launches backend
        logger.info("Durable workflows initialized")
    except Exception as e:
        logger.warning(f"Failed to initialize workflows: {e}")
        # Continue without workflows - they're optional for basic encode/recall

    yield

    # Cleanup
    shutdown_workflows()  # Sync call
    await service.close()
    set_service(None)  # type: ignore[arg-type]


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create a FastAPI application.

    Args:
        settings: Optional settings. Uses environment if None.

    Returns:
        Configured FastAPI application.

    Example:
        ```python
        from engram.api import create_app

        app = create_app()
        # Run with: uvicorn engram.api:app --reload
        ```
    """
    if settings is None:
        settings = Settings()

    app = FastAPI(
        title="Engram",
        description="Memory you can trust. A memory system for AI applications.",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Add CORS middleware if enabled
    if settings.cors_enabled:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_allow_origins,
            allow_credentials=settings.cors_allow_credentials,
            allow_methods=settings.cors_allow_methods,
            allow_headers=settings.cors_allow_headers,
            max_age=settings.cors_max_age,
        )

    # Register exception handlers
    @app.exception_handler(ValidationError)
    async def validation_error_handler(request: Request, exc: ValidationError) -> JSONResponse:
        """Handle validation errors with 400 status."""
        logger.warning(
            "Validation error", field=exc.field, error=exc.message, path=str(request.url)
        )
        return JSONResponse(status_code=400, content=exc.to_dict())

    @app.exception_handler(NotFoundError)
    async def not_found_error_handler(request: Request, exc: NotFoundError) -> JSONResponse:
        """Handle not found errors with 404 status."""
        logger.info(
            "Resource not found",
            resource_type=exc.resource_type,
            resource_id=exc.resource_id,
            path=str(request.url),
        )
        return JSONResponse(status_code=404, content=exc.to_dict())

    @app.exception_handler(AuthenticationError)
    async def authentication_error_handler(
        request: Request, exc: AuthenticationError
    ) -> JSONResponse:
        """Handle authentication errors with 401 status."""
        logger.warning("Authentication failed", error=exc.message, path=str(request.url))
        return JSONResponse(status_code=401, content=exc.to_dict())

    @app.exception_handler(AuthorizationError)
    async def authorization_error_handler(
        request: Request, exc: AuthorizationError
    ) -> JSONResponse:
        """Handle authorization errors with 403 status."""
        logger.warning("Authorization failed", error=exc.message, path=str(request.url))
        return JSONResponse(status_code=403, content=exc.to_dict())

    @app.exception_handler(RateLimitError)
    async def rate_limit_error_handler(request: Request, exc: RateLimitError) -> JSONResponse:
        """Handle rate limit errors with 429 status."""
        logger.warning("Rate limit exceeded", retry_after=exc.retry_after, path=str(request.url))
        return JSONResponse(
            status_code=429,
            content=exc.to_dict(),
            headers={"Retry-After": str(exc.retry_after)},
        )

    @app.exception_handler(EngramError)
    async def engram_error_handler(request: Request, exc: EngramError) -> JSONResponse:
        """Handle all other Engram errors with 500 status."""
        logger.error("Engram error", error=exc.message, code=exc.code, path=str(request.url))
        return JSONResponse(status_code=500, content=exc.to_dict())

    app.include_router(router, prefix="/api/v1")

    return app


# Default app instance for uvicorn
app = create_app()
