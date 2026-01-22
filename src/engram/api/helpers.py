"""API helper functions to reduce code duplication.

Provides:
- Memory type detection from ID prefixes
- Response object builders
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from engram.models import Episode

from .schemas import EpisodeResponse

# Memory ID prefix to type mapping
MEMORY_ID_PREFIXES: dict[str, str] = {
    "ep_": "episodic",
    "struct_": "structured",
    "sem_": "semantic",
    "proc_": "procedural",
}

MemoryTypeStr = Literal["episodic", "structured", "semantic", "procedural"]


def get_memory_type_from_id(memory_id: str) -> MemoryTypeStr | None:
    """Determine memory type from ID prefix.

    Args:
        memory_id: Memory identifier with type prefix.

    Returns:
        Memory type string, or None if prefix not recognized.

    Examples:
        >>> get_memory_type_from_id("ep_abc123")
        'episodic'
        >>> get_memory_type_from_id("sem_xyz789")
        'semantic'
        >>> get_memory_type_from_id("unknown_123")
        None
    """
    for prefix, memory_type in MEMORY_ID_PREFIXES.items():
        if memory_id.startswith(prefix):
            return memory_type  # type: ignore[return-value]
    return None


def episode_to_response(episode: Episode) -> EpisodeResponse:
    """Convert an Episode model to an EpisodeResponse.

    Args:
        episode: The Episode model instance.

    Returns:
        EpisodeResponse with all fields mapped.
    """
    return EpisodeResponse(
        id=episode.id,
        content=episode.content,
        role=episode.role,
        user_id=episode.user_id,
        org_id=episode.org_id,
        session_id=episode.session_id,
        importance=episode.importance,
        created_at=episode.timestamp.isoformat(),
    )
