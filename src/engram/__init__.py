"""Engram: Memory you can trust.

A memory system for AI applications that preserves ground truth,
tracks confidence, and prevents hallucinations.

Quick Start:
    from engram import Engram

    memory = Engram(user_id="user_123")

    # Store an interaction
    result = await memory.encode(
        content="My email is john@example.com",
        role="user",
    )

    # Retrieve relevant memories
    memories = await memory.recall(
        query="What's the user's email?",
        min_confidence=0.7,
    )

Memory Types:
    - Episode: Immutable ground truth (raw interactions)
    - Fact: Pattern-extracted facts (emails, phones, dates)
    - SemanticMemory: LLM-inferred knowledge
    - ProceduralMemory: Behavioral patterns
    - InhibitoryFact: What is NOT true (negations)
    - Working: Current session context (in-memory)

For more information, see: https://github.com/ashita-ai/engram
"""

__version__ = "0.1.0"

# Configuration
from .config import ConfidenceWeights, Settings, settings

# Models
from .models import (
    AuditEntry,
    ConfidenceScore,
    Episode,
    ExtractionMethod,
    Fact,
    InhibitoryFact,
    ProceduralMemory,
    SemanticMemory,
)

__all__ = [
    # Version
    "__version__",
    # Configuration
    "Settings",
    "ConfidenceWeights",
    "settings",
    # Models
    "ConfidenceScore",
    "ExtractionMethod",
    "Episode",
    "Fact",
    "SemanticMemory",
    "ProceduralMemory",
    "InhibitoryFact",
    "AuditEntry",
]
