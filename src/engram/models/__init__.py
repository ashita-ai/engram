"""Memory type models for Engram.

This module exports all memory models and related types:

Memory Types:
    - Episode: Immutable ground truth storage
    - StructuredMemory: Per-episode structured extraction (NEW)
    - SemanticMemory: Cross-episode LLM-inferred knowledge
    - ProceduralMemory: Behavioral patterns
    - Fact: Deterministically extracted facts (DEPRECATED - use StructuredMemory)
    - NegationFact: What is NOT true (DEPRECATED - use StructuredMemory.negations)

Supporting Types:
    - ConfidenceScore: Composite confidence with auditability
    - ExtractionMethod: How memory was extracted
    - QuickExtracts: Immediate regex extractions on Episode
    - ResolvedDate, Person, Preference, Negation: Structured extraction sub-types
    - AuditEntry: Operation logging
"""

from .audit import AuditEntry
from .base import ConfidenceScore, ExtractionMethod, MemoryBase, Staleness, generate_id
from .episode import Episode, QuickExtracts
from .fact import Fact
from .negation import NegationFact
from .procedural import ProceduralMemory
from .semantic import EvolutionEntry, SemanticMemory
from .structured import Negation, Person, Preference, ResolvedDate, StructuredMemory

__all__ = [
    # Base types
    "ConfidenceScore",
    "ExtractionMethod",
    "MemoryBase",
    "Staleness",
    "generate_id",
    # Memory types
    "Episode",
    "QuickExtracts",
    "StructuredMemory",
    "SemanticMemory",
    "ProceduralMemory",
    # Structured sub-types
    "ResolvedDate",
    "Person",
    "Preference",
    "Negation",
    # Deprecated (use StructuredMemory instead)
    "Fact",
    "NegationFact",
    # Supporting types
    "EvolutionEntry",
    # Audit
    "AuditEntry",
]
