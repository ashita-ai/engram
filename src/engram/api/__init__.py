"""FastAPI REST API for Engram.

This module provides the REST API layer for Engram memory operations.

Example:
    ```python
    import uvicorn
    from engram.api import create_app

    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)
    ```

Or run directly:
    ```bash
    uvicorn engram.api:app --reload
    ```
"""

from .app import app, create_app
from .router import router

__all__ = [
    "app",
    "create_app",
    "router",
]
