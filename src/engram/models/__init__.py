"""Memory type models for Engram.

This module exports all memory models and related types:

Memory Types:
    - Episode: Immutable ground truth storage
    - StructuredMemory: Per-episode structured extraction
    - SemanticMemory: Cross-episode LLM-inferred knowledge
    - ProceduralMemory: Behavioral patterns

Supporting Types:
    - ConfidenceScore: Composite confidence with auditability
    - ExtractionMethod: How memory was extracted
    - QuickExtracts: Immediate regex extractions on Episode
    - ResolvedDate, Person, Preference, Negation: Structured extraction sub-types
    - ProvenanceChain, ProvenanceEvent: Derivation tracking
    - AuditEntry: Operation logging
    - WebhookConfig, WebhookEvent, WebhookDelivery: Webhook notifications
"""

from .audit import AuditEntry
from .base import ConfidenceScore, ExtractionMethod, MemoryBase, Staleness, generate_id
from .episode import Episode, QuickExtracts
from .history import ChangeType, HistoryEntry, TriggerType
from .procedural import ProceduralMemory
from .provenance import ProvenanceChain, ProvenanceEvent
from .semantic import EvolutionEntry, SemanticMemory
from .structured import Negation, Person, Preference, ResolvedDate, StructuredMemory
from .webhook import (
    ALL_EVENT_TYPES,
    DeliveryStatus,
    EventType,
    WebhookConfig,
    WebhookDelivery,
    WebhookEvent,
)

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
    # Supporting types
    "EvolutionEntry",
    # Provenance
    "ProvenanceChain",
    "ProvenanceEvent",
    # Audit
    "AuditEntry",
    # History
    "HistoryEntry",
    "ChangeType",
    "TriggerType",
    # Webhooks
    "WebhookConfig",
    "WebhookEvent",
    "WebhookDelivery",
    "EventType",
    "DeliveryStatus",
    "ALL_EVENT_TYPES",
]
