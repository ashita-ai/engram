"""FastAPI application for Engram."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from engram.config import Settings
from engram.service import EngramService

from .router import router, set_service


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage application lifespan.

    Initializes the EngramService on startup and cleans up on shutdown.
    """
    settings = Settings()
    service = EngramService.create(settings)

    # Initialize storage and set service
    await service.initialize()
    set_service(service)

    yield

    # Cleanup
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
    app = FastAPI(
        title="Engram",
        description="Memory you can trust. A memory system for AI applications.",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.include_router(router, prefix="/api/v1")

    return app


# Default app instance for uvicorn
app = create_app()
