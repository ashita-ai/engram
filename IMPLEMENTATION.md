# Implementation Plan

Phased implementation plan for Engram. Take time to polish.

> **Approach**: Build foundation first, layer capabilities, ship when solid.

## Overview

```
Phase 1: Foundation     → Models, config, project structure
Phase 2: Storage        → Qdrant client, collections, audit log
Phase 3: Extraction     → Pattern matchers (email, phone, URL, date)
Phase 4: Core API       → encode(), recall(), working memory
Phase 5: Embeddings     → OpenAI + FastEmbed providers
Phase 6: Background     → DBOS setup, consolidation workflow
Phase 7: Semantics      → LLM extraction, linking, selectivity
Phase 8: Server         → FastAPI REST API
Phase 9: Polish         → SDK convenience, errors, logging
Phase 10: Demo          → Marimo interactive notebook
Phase 11: Benchmarks    → HaluMem accuracy evaluation
```

---

## Phase 1: Foundation

**Goal**: Project structure, models, configuration.

### 1.1 Project Structure

```
engram/
├── src/
│   └── engram/
│       ├── __init__.py
│       ├── client.py           # Main Engram class
│       ├── config.py           # Settings, ConfidenceWeights
│       ├── models/
│       │   ├── __init__.py
│       │   ├── base.py         # Base model, ConfidenceScore
│       │   ├── episode.py
│       │   ├── fact.py
│       │   ├── semantic.py
│       │   ├── procedural.py
│       │   ├── inhibitory.py
│       │   └── audit.py        # Audit log entry
│       ├── extraction/
│       │   ├── __init__.py
│       │   ├── base.py         # Extractor interface
│       │   ├── email.py
│       │   ├── phone.py
│       │   ├── url.py
│       │   └── date.py
│       ├── storage/
│       │   ├── __init__.py
│       │   └── qdrant.py
│       ├── embeddings/
│       │   ├── __init__.py
│       │   ├── base.py         # Embedder interface
│       │   ├── openai.py
│       │   └── fastembed.py
│       ├── workflows/
│       │   ├── __init__.py
│       │   ├── consolidation.py
│       │   └── decay.py
│       └── api/
│           ├── __init__.py
│           ├── app.py          # FastAPI app
│           ├── routes/
│           └── middleware/
├── tests/
│   ├── unit/
│   ├── integration/
│   └── benchmarks/
├── demo/
│   └── marimo/
├── pyproject.toml
├── Dockerfile
└── docker-compose.yml
```

### 1.2 Models

```python
# src/engram/models/base.py
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum

class ExtractionMethod(str, Enum):
    VERBATIM = "verbatim"
    EXTRACTED = "extracted"
    INFERRED = "inferred"

class ConfidenceScore(BaseModel):
    value: float = Field(ge=0.0, le=1.0)
    extraction_method: ExtractionMethod
    extraction_base: float
    supporting_episodes: int = 1
    last_confirmed: datetime = Field(default_factory=datetime.utcnow)
    contradictions: int = 0

    def explain(self) -> str:
        """Human-readable explanation of confidence score."""
        return (
            f"{self.value:.2f}: {self.extraction_method.value}, "
            f"{self.supporting_episodes} sources, "
            f"confirmed {self._time_ago(self.last_confirmed)}"
        )
```

```python
# src/engram/models/episode.py
class Episode(BaseModel):
    id: str = Field(default_factory=lambda: f"ep_{uuid4().hex[:12]}")
    content: str
    role: str  # "user" | "assistant" | "system"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    session_id: str | None = None
    user_id: str
    org_id: str | None = None
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    embedding: list[float] | None = None
```

### 1.3 Configuration

```python
# src/engram/config.py
from pydantic_settings import BaseSettings

class ConfidenceWeights(BaseModel):
    extraction: float = 0.50
    corroboration: float = 0.25
    recency: float = 0.15
    verification: float = 0.10
    decay_half_life_days: int = 365

class Settings(BaseSettings):
    # Storage
    qdrant_url: str = "http://localhost:6333"
    collection_prefix: str = "engram"

    # Embeddings
    embedding_provider: str = "openai"  # or "fastembed"
    embedding_model: str = "text-embedding-3-small"
    openai_api_key: str | None = None

    # LLM (for consolidation)
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"

    # Confidence
    confidence_weights: ConfidenceWeights = ConfidenceWeights()

    # Background
    durable_backend: str = "dbos"  # or "temporal"

    class Config:
        env_prefix = "ENGRAM_"
        env_file = ".env"
```

### 1.4 Tasks

- [ ] Initialize pyproject.toml with dependencies
- [ ] Create src/engram directory structure
- [ ] Implement ConfidenceScore model with explain()
- [ ] Implement Episode model
- [ ] Implement Fact model
- [ ] Implement SemanticMemory model
- [ ] Implement ProceduralMemory model
- [ ] Implement InhibitoryFact model
- [ ] Implement AuditEntry model
- [ ] Implement Settings with pydantic-settings
- [ ] Implement ConfidenceWeights
- [ ] Write unit tests for models

---

## Phase 2: Storage

**Goal**: Qdrant client, collection management, audit logging.

### 2.1 Qdrant Client

```python
# src/engram/storage/qdrant.py
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

class EngramStorage:
    COLLECTIONS = [
        "episode",
        "fact",
        "semantic",
        "procedural",
        "negation",
        "audit",
    ]

    def __init__(self, settings: Settings):
        self.client = QdrantClient(url=settings.qdrant_url)
        self.prefix = settings.collection_prefix
        self._ensure_collections()

    def _collection_name(self, memory_type: str) -> str:
        return f"{self.prefix}_{memory_type}"

    def _ensure_collections(self):
        """Create collections if they don't exist."""
        for coll in self.COLLECTIONS:
            name = self._collection_name(coll)
            if not self.client.collection_exists(name):
                self.client.create_collection(
                    collection_name=name,
                    vectors_config=VectorParams(
                        size=1536,  # OpenAI embedding size
                        distance=Distance.COSINE,
                    ),
                )

    async def store_episode(self, episode: Episode) -> str:
        """Store episode and return ID."""
        ...

    async def store_fact(self, fact: Fact) -> str:
        """Store fact and return ID."""
        ...

    async def search(
        self,
        collection: str,
        query_vector: list[float],
        limit: int = 10,
        filters: dict | None = None,
    ) -> list[dict]:
        """Vector search with optional filters."""
        ...

    async def log_audit(self, entry: AuditEntry):
        """Append to audit log."""
        ...
```

### 2.2 Multi-Tenancy Keys

```python
def _build_key(org_id: str | None, user_id: str, memory_id: str) -> str:
    """Build storage key with tenant isolation."""
    if org_id:
        return f"{org_id}/{user_id}/{memory_id}"
    return f"personal/{user_id}/{memory_id}"
```

### 2.3 Tasks

- [ ] Implement EngramStorage class
- [ ] Implement collection initialization
- [ ] Implement store_episode()
- [ ] Implement store_fact()
- [ ] Implement store_semantic()
- [ ] Implement store_procedural()
- [ ] Implement store_inhibitory()
- [ ] Implement search() with filters
- [ ] Implement log_audit()
- [ ] Implement get_by_id()
- [ ] Implement delete()
- [ ] Write integration tests against real Qdrant

---

## Phase 3: Extraction

**Goal**: Deterministic pattern extractors.

### 3.1 Extractor Interface

```python
# src/engram/extraction/base.py
from abc import ABC, abstractmethod

class Extractor(ABC):
    """Base class for pattern extractors."""

    @property
    @abstractmethod
    def category(self) -> str:
        """Category name (email, phone, url, date)."""
        ...

    @abstractmethod
    def extract(self, text: str) -> list[ExtractedFact]:
        """Extract facts from text."""
        ...

class ExtractedFact(BaseModel):
    content: str
    category: str
    span: tuple[int, int]  # Start, end position in text
    raw_match: str
```

### 3.2 Extractors

```python
# src/engram/extraction/email.py
import re
from email_validator import validate_email, EmailNotValidError

class EmailExtractor(Extractor):
    category = "email"
    PATTERN = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'

    def extract(self, text: str) -> list[ExtractedFact]:
        facts = []
        for match in re.finditer(self.PATTERN, text):
            email = match.group()
            try:
                validate_email(email, check_deliverability=False)
                facts.append(ExtractedFact(
                    content=f"email={email}",
                    category=self.category,
                    span=(match.start(), match.end()),
                    raw_match=email,
                ))
            except EmailNotValidError:
                pass
        return facts
```

```python
# src/engram/extraction/phone.py
import phonenumbers

class PhoneExtractor(Extractor):
    category = "phone"

    def extract(self, text: str, region: str = "US") -> list[ExtractedFact]:
        facts = []
        for match in phonenumbers.PhoneNumberMatcher(text, region):
            formatted = phonenumbers.format_number(
                match.number,
                phonenumbers.PhoneNumberFormat.E164
            )
            facts.append(ExtractedFact(
                content=f"phone={formatted}",
                category=self.category,
                span=(match.start, match.end),
                raw_match=match.raw_string,
            ))
        return facts
```

```python
# src/engram/extraction/date.py
import dateparser

class DateExtractor(Extractor):
    category = "date"

    def extract(self, text: str) -> list[ExtractedFact]:
        # dateparser.search finds dates in text
        results = dateparser.search.search_dates(text) or []
        facts = []
        for raw, parsed in results:
            facts.append(ExtractedFact(
                content=f"date={parsed.isoformat()}",
                category=self.category,
                span=self._find_span(text, raw),
                raw_match=raw,
            ))
        return facts
```

### 3.3 Extraction Pipeline

```python
# src/engram/extraction/__init__.py
class ExtractionPipeline:
    def __init__(self):
        self.extractors = [
            EmailExtractor(),
            PhoneExtractor(),
            UrlExtractor(),
            DateExtractor(),
        ]

    def extract_all(self, text: str) -> list[ExtractedFact]:
        """Run all extractors and return combined results."""
        facts = []
        for extractor in self.extractors:
            facts.extend(extractor.extract(text))
        return facts
```

### 3.4 Tasks

- [ ] Implement Extractor base class
- [ ] Implement ExtractedFact model
- [ ] Implement EmailExtractor
- [ ] Implement PhoneExtractor
- [ ] Implement UrlExtractor
- [ ] Implement DateExtractor
- [ ] Implement ExtractionPipeline
- [ ] Write unit tests for each extractor
- [ ] Test edge cases (malformed inputs, unicode, etc.)

---

## Phase 4: Core API

**Goal**: encode(), recall(), working memory.

### 4.1 Main Client

```python
# src/engram/client.py
class Engram:
    def __init__(
        self,
        user_id: str,
        org_id: str | None = None,
        session_id: str | None = None,
        settings: Settings | None = None,
    ):
        self.user_id = user_id
        self.org_id = org_id
        self.session_id = session_id
        self.settings = settings or Settings()

        self._storage = EngramStorage(self.settings)
        self._embedder = get_embedder(self.settings)
        self._extractor = ExtractionPipeline()
        self._working_memory: list[Episode] = []

    async def encode(
        self,
        content: str,
        role: str = "user",
        importance: float = 0.5,
        extract_facts: bool = True,
    ) -> EncodeResult:
        """
        Store an interaction.

        - Creates Episode (sync)
        - Extracts Facts via patterns (sync)
        - Queues consolidation (async, background)
        """
        # 1. Create episode
        episode = Episode(
            content=content,
            role=role,
            user_id=self.user_id,
            org_id=self.org_id,
            session_id=self.session_id,
            importance=importance,
            embedding=await self._embedder.embed(content),
        )

        # 2. Store episode
        await self._storage.store_episode(episode)

        # 3. Add to working memory
        self._working_memory.append(episode)

        # 4. Extract facts (deterministic)
        facts = []
        if extract_facts:
            extracted = self._extractor.extract_all(content)
            for ext in extracted:
                fact = Fact(
                    content=ext.content,
                    category=ext.category,
                    source_episode_id=episode.id,
                    user_id=self.user_id,
                    org_id=self.org_id,
                    confidence=ConfidenceScore(
                        value=0.9,
                        extraction_method=ExtractionMethod.EXTRACTED,
                        extraction_base=0.9,
                    ),
                    embedding=await self._embedder.embed(ext.content),
                )
                await self._storage.store_fact(fact)
                facts.append(fact)

        # 5. Audit log
        await self._storage.log_audit(AuditEntry(
            event="encode",
            user_id=self.user_id,
            org_id=self.org_id,
            session_id=self.session_id,
            details={"episode_id": episode.id, "facts_count": len(facts)},
        ))

        return EncodeResult(episode=episode, facts=facts)

    async def recall(
        self,
        query: str,
        memory_types: list[str] | None = None,
        min_confidence: float = 0.0,
        limit: int = 10,
        include_working: bool = True,
    ) -> list[Memory]:
        """
        Retrieve relevant memories.

        Searches across specified memory types,
        filters by confidence, returns ranked results.
        """
        memory_types = memory_types or ["episodic", "factual", "semantic"]
        query_embedding = await self._embedder.embed(query)

        results = []

        # Search each memory type
        for mem_type in memory_types:
            hits = await self._storage.search(
                collection=mem_type,
                query_vector=query_embedding,
                limit=limit,
                filters={
                    "user_id": self.user_id,
                    "org_id": self.org_id,
                },
            )
            results.extend(hits)

        # Include working memory if requested
        if include_working and "episodic" in memory_types:
            for ep in self._working_memory:
                # Simple relevance check
                results.append({"memory": ep, "score": 0.5})

        # Filter by confidence
        results = [r for r in results if r.get("confidence", 1.0) >= min_confidence]

        # Sort by relevance score
        results.sort(key=lambda x: x["score"], reverse=True)

        # Audit log
        await self._storage.log_audit(AuditEntry(
            event="recall",
            user_id=self.user_id,
            org_id=self.org_id,
            details={"query_hash": hash(query), "results_count": len(results)},
        ))

        return results[:limit]

    def get_working_memory(self) -> list[Episode]:
        """Get current session's working memory."""
        return self._working_memory.copy()

    def clear_working_memory(self):
        """Clear working memory (end of session)."""
        self._working_memory.clear()
```

### 4.2 Tasks

- [ ] Implement Engram client class
- [ ] Implement encode() method
- [ ] Implement recall() method
- [ ] Implement get_working_memory()
- [ ] Implement clear_working_memory()
- [ ] Implement EncodeResult model
- [ ] Implement Memory result model
- [ ] Write integration tests for encode/recall
- [ ] Test multi-tenancy isolation

---

## Phase 5: Embeddings

**Goal**: Pluggable embedding providers.

### 5.1 Embedder Interface

```python
# src/engram/embeddings/base.py
from abc import ABC, abstractmethod

class Embedder(ABC):
    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """Generate embedding for text."""
        ...

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        ...

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Embedding vector dimensions."""
        ...
```

### 5.2 OpenAI Provider

```python
# src/engram/embeddings/openai.py
from openai import AsyncOpenAI

class OpenAIEmbedder(Embedder):
    def __init__(self, model: str = "text-embedding-3-small", api_key: str | None = None):
        self.model = model
        self.client = AsyncOpenAI(api_key=api_key)
        self._dimensions = 1536  # text-embedding-3-small

    async def embed(self, text: str) -> list[float]:
        response = await self.client.embeddings.create(
            model=self.model,
            input=text,
        )
        return response.data[0].embedding

    @property
    def dimensions(self) -> int:
        return self._dimensions
```

### 5.3 FastEmbed Provider (Local, Free)

```python
# src/engram/embeddings/fastembed.py
from fastembed import TextEmbedding

class FastEmbedEmbedder(Embedder):
    def __init__(self, model: str = "BAAI/bge-small-en-v1.5"):
        self.model = TextEmbedding(model)
        self._dimensions = 384  # bge-small

    async def embed(self, text: str) -> list[float]:
        # FastEmbed is sync, wrap for async interface
        embeddings = list(self.model.embed([text]))
        return embeddings[0].tolist()

    @property
    def dimensions(self) -> int:
        return self._dimensions
```

### 5.4 Provider Factory

```python
# src/engram/embeddings/__init__.py
def get_embedder(settings: Settings) -> Embedder:
    if settings.embedding_provider == "openai":
        return OpenAIEmbedder(
            model=settings.embedding_model,
            api_key=settings.openai_api_key,
        )
    elif settings.embedding_provider == "fastembed":
        return FastEmbedEmbedder(model=settings.embedding_model)
    else:
        raise ValueError(f"Unknown embedding provider: {settings.embedding_provider}")
```

### 5.5 Tasks

- [ ] Implement Embedder base class
- [ ] Implement OpenAIEmbedder
- [ ] Implement FastEmbedEmbedder
- [ ] Implement get_embedder() factory
- [ ] Handle dimension mismatches (different models = different sizes)
- [ ] Write tests for each embedder

---

## Phase 6: Background Processing

**Goal**: DBOS setup, consolidation and decay workflows.

### 6.1 DBOS Setup

```python
# src/engram/workflows/__init__.py
from dbos import DBOS

dbos = DBOS()

def init_dbos():
    """Initialize DBOS for durable execution."""
    dbos.launch()
```

### 6.2 Consolidation Workflow

```python
# src/engram/workflows/consolidation.py
from dbos import DBOS
from pydantic_ai import Agent

consolidation_agent = Agent(
    "openai:gpt-4o-mini",
    result_type=ConsolidationResult,
    system_prompt="""
    Extract semantic facts from these conversation episodes.
    Identify relationships between memories.
    Flag contradictions.
    Be conservative - only extract high-confidence facts.
    """,
)

@DBOS.workflow()
async def consolidate_episodes(
    user_id: str,
    org_id: str | None,
    episode_ids: list[str],
) -> ConsolidationResult:
    """
    Durable consolidation workflow.
    Survives crashes, retries on failure.
    """
    # 1. Fetch episodes
    episodes = await fetch_episodes(episode_ids)

    # 2. Run LLM extraction
    result = await consolidation_agent.run(
        format_episodes(episodes)
    )

    # 3. Store semantic memories
    for fact in result.data.facts:
        await store_semantic_memory(
            content=fact,
            source_episode_ids=episode_ids,
            user_id=user_id,
            org_id=org_id,
        )

    # 4. Build links
    for mem_id, related_id in result.data.links:
        await create_link(mem_id, related_id)

    return result.data
```

### 6.3 Decay Workflow

```python
# src/engram/workflows/decay.py
from math import exp

@DBOS.workflow()
async def decay_memories(user_id: str, org_id: str | None):
    """
    Update confidence scores based on time decay.
    Archive or delete low-confidence memories.
    """
    memories = await get_all_memories(user_id, org_id)

    for memory in memories:
        days_since = (datetime.utcnow() - memory.confidence.last_confirmed).days
        decay_factor = exp(-days_since / settings.confidence_weights.decay_half_life_days)

        new_confidence = memory.confidence.value * decay_factor

        if new_confidence < 0.1:
            await archive_memory(memory.id)
        else:
            await update_confidence(memory.id, new_confidence)
```

### 6.4 Tasks

- [ ] Set up DBOS in project
- [ ] Implement consolidation workflow
- [ ] Implement ConsolidationResult model
- [ ] Implement decay workflow
- [ ] Add consolidate() method to Engram client
- [ ] Add decay() method to Engram client
- [ ] Write tests for workflows (mock LLM)
- [ ] Test crash recovery

---

## Phase 7: Semantic Extraction

**Goal**: LLM-based extraction, linking, selectivity.

### 7.1 Pydantic AI Agent

```python
# src/engram/agents/consolidation.py
from pydantic import BaseModel
from pydantic_ai import Agent

class ExtractedSemanticFact(BaseModel):
    content: str
    confidence: float
    reasoning: str

class IdentifiedLink(BaseModel):
    source_content: str
    target_content: str
    relationship: str

class ConsolidationResult(BaseModel):
    facts: list[ExtractedSemanticFact]
    links: list[IdentifiedLink]
    contradictions: list[str]

consolidation_agent = Agent(
    "openai:gpt-4o-mini",
    result_type=ConsolidationResult,
    system_prompt="""
    You are analyzing conversation episodes to extract lasting knowledge.

    Extract facts that are:
    - Likely to be relevant in future conversations
    - Not just transient details
    - Stated with reasonable confidence by the user

    Identify relationships between concepts.
    Flag any contradictions with previously known facts.

    Be conservative. When uncertain, don't extract.
    """,
)
```

### 7.2 Link Building

```python
async def build_links(
    new_memory: SemanticMemory,
    existing_memories: list[SemanticMemory],
    embedder: Embedder,
) -> list[str]:
    """Find and create links to related memories."""
    new_embedding = new_memory.embedding

    related_ids = []
    for existing in existing_memories:
        similarity = cosine_similarity(new_embedding, existing.embedding)
        if similarity > 0.7:  # Threshold for linking
            related_ids.append(existing.id)

    return related_ids
```

### 7.3 Selectivity Scoring

```python
async def update_selectivity(memory_id: str, consolidated: bool):
    """
    Update selectivity score based on consolidation result.
    Memories that survive consolidation become more selective.
    """
    memory = await get_memory(memory_id)

    if consolidated:
        # Survived consolidation - increase selectivity
        new_score = min(1.0, memory.selectivity_score + 0.1)
    else:
        # Pruned - decrease selectivity
        new_score = max(0.0, memory.selectivity_score - 0.1)

    await update_memory(memory_id, selectivity_score=new_score)
```

### 7.4 Tasks

- [ ] Implement ConsolidationResult model
- [ ] Implement consolidation_agent with Pydantic AI
- [ ] Implement build_links()
- [ ] Implement update_selectivity()
- [ ] Implement inhibitory fact detection
- [ ] Write tests with mocked LLM responses

---

## Phase 8: Server

**Goal**: FastAPI REST API.

### 8.1 FastAPI App

```python
# src/engram/api/app.py
from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import HTTPBearer

app = FastAPI(
    title="Engram API",
    description="Memory you can trust",
    version="0.1.0",
)

security = HTTPBearer()

@app.post("/v1/encode")
async def encode(
    request: EncodeRequest,
    token: str = Depends(security),
):
    memory = get_memory_client(token)
    result = await memory.encode(
        content=request.content,
        role=request.role,
        importance=request.importance,
    )
    return EncodeResponse(
        success=True,
        episode_id=result.episode.id,
        facts_extracted=[f.model_dump() for f in result.facts],
    )

@app.post("/v1/recall")
async def recall(
    request: RecallRequest,
    token: str = Depends(security),
):
    memory = get_memory_client(token)
    results = await memory.recall(
        query=request.query,
        memory_types=request.memory_types,
        min_confidence=request.min_confidence,
        limit=request.limit,
    )
    return RecallResponse(
        success=True,
        memories=[m.model_dump() for m in results],
    )

@app.post("/v1/consolidate")
async def consolidate(token: str = Depends(security)):
    memory = get_memory_client(token)
    result = await memory.consolidate()
    return {"success": True, "result": result}

@app.get("/v1/health")
async def health():
    return {"status": "healthy"}
```

### 8.2 Request/Response Models

```python
# src/engram/api/models.py
class EncodeRequest(BaseModel):
    user_id: str
    content: str
    role: str = "user"
    session_id: str | None = None
    importance: float = 0.5

class RecallRequest(BaseModel):
    user_id: str
    query: str
    memory_types: list[str] | None = None
    min_confidence: float = 0.0
    limit: int = 10
```

### 8.3 Tasks

- [ ] Implement FastAPI app
- [ ] Implement /v1/encode endpoint
- [ ] Implement /v1/recall endpoint
- [ ] Implement /v1/consolidate endpoint
- [ ] Implement /v1/decay endpoint
- [ ] Implement /v1/health endpoint
- [ ] Implement authentication middleware
- [ ] Implement rate limiting
- [ ] Write API tests
- [ ] Update docker-compose.yml with API service

---

## Phase 9: Polish

**Goal**: SDK convenience, error handling, logging.

### 9.1 Context Manager

```python
# src/engram/context.py
from contextvars import ContextVar

_memory_context: ContextVar[Engram | None] = ContextVar("memory", default=None)

@asynccontextmanager
async def memory_context(
    user_id: str,
    org_id: str | None = None,
    session_id: str | None = None,
    settings: Settings | None = None,
):
    """
    Context manager for memory operations.
    Auto-commits on exit.
    """
    memory = Engram(
        user_id=user_id,
        org_id=org_id,
        session_id=session_id,
        settings=settings,
    )
    token = _memory_context.set(memory)
    try:
        yield memory
    finally:
        _memory_context.reset(token)
        # Cleanup if needed
```

### 9.2 Error Handling

```python
# src/engram/exceptions.py
class EngramError(Exception):
    """Base exception for Engram."""
    pass

class ValidationError(EngramError):
    """Invalid input."""
    def __init__(self, field: str, message: str):
        self.field = field
        self.message = message
        super().__init__(f"{field}: {message}")

class StorageError(EngramError):
    """Storage operation failed."""
    pass

class EmbeddingError(EngramError):
    """Embedding generation failed."""
    pass

class RateLimitError(EngramError):
    """Rate limit exceeded."""
    def __init__(self, retry_after: int):
        self.retry_after = retry_after
        super().__init__(f"Rate limited. Retry after {retry_after}s")
```

### 9.3 Logging

```python
# src/engram/logging.py
import structlog

def configure_logging(level: str = "INFO"):
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
    )

logger = structlog.get_logger()
```

### 9.4 Tasks

- [ ] Implement memory_context() context manager
- [ ] Implement exception classes
- [ ] Implement structured logging
- [ ] Add logging throughout codebase
- [ ] Add retry logic for transient failures
- [ ] Polish public API surface
- [ ] Write docstrings for all public methods
- [ ] Generate API documentation

---

## Phase 10: Demo

**Goal**: Marimo interactive notebook.

### 10.1 Marimo App

```python
# demo/marimo/app.py
import marimo as mo

app = mo.App()

@app.cell
def setup():
    from engram import Engram
    memory = Engram(user_id="demo_user")
    return memory

@app.cell
def encode_ui(memory):
    message = mo.ui.text_area(label="Message to encode")
    encode_btn = mo.ui.button(label="Encode")

    if encode_btn.value:
        result = memory.encode(message.value)
        mo.output.append(f"Stored episode: {result.episode.id}")
        mo.output.append(f"Facts extracted: {len(result.facts)}")
        for fact in result.facts:
            mo.output.append(f"  - {fact.content} (confidence: {fact.confidence.value:.2f})")

    return message, encode_btn

@app.cell
def recall_ui(memory):
    query = mo.ui.text(label="Query")
    recall_btn = mo.ui.button(label="Recall")

    if recall_btn.value:
        results = memory.recall(query.value)
        for mem in results:
            mo.output.append(f"- {mem.content}")
            mo.output.append(f"  Confidence: {mem.confidence.explain()}")
            mo.output.append(f"  Source: {mem.source_episode_id}")
```

### 10.2 Tasks

- [ ] Set up Marimo in project
- [ ] Create interactive encode demo
- [ ] Create interactive recall demo
- [ ] Show confidence scores visually
- [ ] Show source tracing ("why does it think this?")
- [ ] Create "full conversation" demo
- [ ] Test demo works with FastEmbed (no API key)
- [ ] Package as Docker demo

---

## Phase 11: Benchmarks

**Goal**: HaluMem accuracy evaluation.

### 11.1 HaluMem Setup

```python
# tests/benchmarks/halumem.py
"""
HaluMem benchmark implementation.
Based on: https://arxiv.org/html/2511.03506

Metrics:
- Answer Accuracy: % of questions answered correctly
- Hallucination Rate: % of answers with fabricated info
- Omission Rate: % of answers missing key info
"""

class HaluMemBenchmark:
    def __init__(self, memory: Engram):
        self.memory = memory
        self.results = []

    async def run_single(self, conversation: list[dict], question: str, expected: str):
        """Run single benchmark case."""
        # 1. Encode conversation
        for msg in conversation:
            await self.memory.encode(content=msg["content"], role=msg["role"])

        # 2. Recall for question
        context = await self.memory.recall(query=question, limit=10)

        # 3. Evaluate
        result = self.evaluate(context, question, expected)
        self.results.append(result)

        return result

    def evaluate(self, context, question, expected) -> BenchmarkResult:
        """Evaluate retrieval quality."""
        # Check if expected info is in retrieved context
        # Check for hallucinated info
        # Check for omissions
        ...

    def report(self) -> BenchmarkReport:
        """Generate benchmark report."""
        return BenchmarkReport(
            accuracy=self.calc_accuracy(),
            hallucination_rate=self.calc_hallucination_rate(),
            omission_rate=self.calc_omission_rate(),
        )
```

### 11.2 Tasks

- [ ] Download/create HaluMem test dataset
- [ ] Implement HaluMemBenchmark class
- [ ] Implement evaluation metrics
- [ ] Run benchmark against Engram
- [ ] Compare with baseline (raw retrieval)
- [ ] Document results
- [ ] Set up CI benchmark runs

---

## Implementation Order

```
Week 1-2: Phase 1 (Foundation) + Phase 2 (Storage)
Week 3:   Phase 3 (Extraction) + Phase 4 (Core API)
Week 4:   Phase 5 (Embeddings) + Phase 6 (Background)
Week 5:   Phase 7 (Semantics) + Phase 8 (Server)
Week 6:   Phase 9 (Polish) + Phase 10 (Demo)
Week 7:   Phase 11 (Benchmarks) + Documentation
Week 8:   Testing, bug fixes, v0.1 release
```

---

## Dependencies

```toml
# pyproject.toml
[project]
dependencies = [
    # Core
    "pydantic>=2.0",
    "pydantic-settings>=2.0",

    # Storage
    "qdrant-client>=1.7",

    # Embeddings
    "openai>=1.0",
    "fastembed>=0.2",

    # Extraction
    "email-validator>=2.0",
    "phonenumbers>=8.0",
    "python-dateutil>=2.8",
    "dateparser>=1.0",
    "validators>=0.20",

    # LLM
    "pydantic-ai>=0.0.20",

    # Background
    "dbos>=0.9",

    # Server
    "fastapi>=0.100",
    "uvicorn>=0.20",

    # Utils
    "structlog>=24.0",
    "httpx>=0.25",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-asyncio>=0.21",
    "pytest-cov>=4.0",
    "ruff>=0.1",
    "mypy>=1.0",
]
demo = [
    "marimo>=0.1",
]
```

---

## Success Criteria

### v0.1 Release Criteria

- [ ] encode() works with all 6 memory types
- [ ] recall() returns confidence-scored results
- [ ] Pattern extraction works for email, phone, URL, date
- [ ] Consolidation produces semantic memories
- [ ] REST API is functional
- [ ] Demo runs without API key (FastEmbed)
- [ ] HaluMem benchmark shows improvement over baseline
- [ ] Documentation is complete
- [ ] Tests pass with >80% coverage
