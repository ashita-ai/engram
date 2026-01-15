"""Engram service layer.

Provides the high-level EngramService for encoding and recalling memories.

Example:
    ```python
    from engram.service import EngramService

    async with EngramService.create() as engram:
        result = await engram.encode(
            content="My email is user@example.com",
            role="user",
            user_id="user_123",
        )
        memories = await engram.recall("email", user_id="user_123")
    ```
"""

from .base import EngramService
from .helpers import IMPORTANCE_KEYWORDS, calculate_importance, cosine_similarity
from .models import (
    EncodeResult,
    RecallResult,
    SourceEpisodeSummary,
    VerificationResult,
)

# Backwards compatibility alias
_calculate_importance = calculate_importance

__all__ = [
    "IMPORTANCE_KEYWORDS",
    "EncodeResult",
    "EngramService",
    "RecallResult",
    "SourceEpisodeSummary",
    "VerificationResult",
    "_calculate_importance",
    "calculate_importance",
    "cosine_similarity",
]
