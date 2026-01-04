# Design Notes

Technical decisions and rationale for the Engram stack.

## Stack Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     Application Layer                        │
├─────────────────────────────────────────────────────────────┤
│                      Engram API                              │
│         encode() / recall() / consolidate() / decay()        │
├─────────────────────────────────────────────────────────────┤
│  Pydantic          │  Pydantic AI       │  python-dotenv    │
│  Data models       │  LLM agents        │  Configuration    │
├─────────────────────────────────────────────────────────────┤
│                        Qdrant                                │
│              Vector storage for all memory types             │
├─────────────────────────────────────────────────────────────┤
│                   Docker / Docker Compose                    │
│              Local dev and production deployment             │
└─────────────────────────────────────────────────────────────┘
```

## Core Dependencies

### Pydantic v2

**What**: Data validation and settings management.

**Why**:
- Type-safe memory models with runtime validation
- `ConfigDict` for strict validation (`extra="forbid"`)
- `computed_field` for derived properties
- Native JSON serialization for Qdrant payloads

**Usage**:
```python
from pydantic import BaseModel, ConfigDict, Field
from datetime import datetime

class Episode(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    id: str
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    importance: float = Field(ge=0.0, le=1.0)
    session_id: str
```

### Pydantic AI

**What**: LLM agents with structured outputs.

**Why**:
- Type-safe LLM responses (no parsing JSON strings)
- Provider-agnostic (OpenAI, Anthropic, Google, Groq)
- Built on Pydantic models
- Streaming support
- Tool/function calling

**Usage**:
```python
from pydantic import BaseModel
from pydantic_ai import Agent

class ExtractedFacts(BaseModel):
    facts: list[str]
    confidence: float
    reasoning: str

consolidation_agent = Agent(
    "openai:gpt-4o-mini",
    result_type=ExtractedFacts,
    system_prompt="Extract facts from conversation episodes.",
)

result = await consolidation_agent.run(episodes_text)
# result.data is ExtractedFacts, fully typed
```

### python-dotenv

**What**: Load environment variables from `.env` files.

**Why**:
- Keep secrets out of code
- Different configs for dev/staging/prod
- Standard pattern, works with Docker

**Usage**:
```python
from dotenv import load_dotenv
import os

load_dotenv()  # Load .env file

qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
openai_key = os.getenv("OPENAI_API_KEY")
```

### Qdrant

**What**: Vector database for semantic search.

**Why**:
- Purpose-built for embeddings
- Payload filtering (filter by memory type, confidence, time)
- HNSW indexing for fast similarity search
- Docker-native, easy local dev
- Managed cloud option for production

**Collections**:
```
engram_episodic    # Raw interactions, immutable
engram_factual     # Pattern-extracted facts
engram_semantic    # LLM-inferred knowledge
engram_procedural  # Behavioral patterns
engram_inhibitory  # What is NOT true (negations)
```

### Durable Execution: DBOS (Local) + Temporal (Production)

Engram's `consolidate()` operation runs LLM extraction in the background. This needs durable execution to handle crashes, retries, and long-running workflows.

**Two-tier approach**:

| Environment | Backend | Why |
|-------------|---------|-----|
| **Local dev / examples** | DBOS | In-process, just needs SQLite, zero infrastructure |
| **Production** | Temporal | Distributed, battle-tested, full visibility |

**DBOS (Local)**:
```python
from dbos import DBOS

@DBOS.workflow()
async def consolidate_workflow(episodes: list[str]) -> list[Fact]:
    """Durable consolidation - survives crashes."""
    result = await consolidation_agent.run(episodes)
    return result.data.facts
```

**Temporal (Production)**:
```python
from temporalio import workflow, activity

@activity.defn
async def extract_facts(episodes: list[str]) -> list[Fact]:
    result = await consolidation_agent.run(episodes)
    return result.data.facts

@workflow.defn
class ConsolidateWorkflow:
    @workflow.run
    async def run(self, episodes: list[str]) -> list[Fact]:
        return await workflow.execute_activity(
            extract_facts,
            episodes,
            start_to_close_timeout=timedelta(minutes=5),
        )
```

**Configuration** (`.env`):
```bash
# Local dev
DURABLE_BACKEND=dbos

# Production
DURABLE_BACKEND=temporal
TEMPORAL_ADDRESS=temporal:7233
```

**The code abstraction**:
```python
# src/engram/durability.py
async def run_durable(fn, *args):
    if settings.durable_backend == "dbos":
        return await dbos_run(fn, *args)
    else:
        return await temporal_run(fn, *args)
```

## Configuration

### Environment Variables

All configuration via environment variables. See `.env.example` for full list.

**Required**:
- `QDRANT_URL` - Qdrant connection string
- At least one LLM API key for consolidation

**Optional**:
- `EMBEDDING_PROVIDER` / `EMBEDDING_MODEL` - Embedding config
- `CONSOLIDATION_MODEL` - Which LLM for semantic extraction
- `DECAY_*` - Decay scheduling and thresholds

### Settings Pattern

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    qdrant_url: str = "http://localhost:6333"
    embedding_provider: str = "openai"
    embedding_model: str = "text-embedding-3-small"
    consolidation_model: str = "openai:gpt-4o-mini"

    class Config:
        env_file = ".env"

settings = Settings()
```

## Docker Setup

Docker Compose is self-contained. Infrastructure env vars are set in the compose file.
Only LLM API keys need to be provided externally.

### What Docker Sets Automatically

```yaml
# These are hardcoded in docker-compose.yml - no config needed
QDRANT_URL: http://qdrant:6333
DURABLE_BACKEND: temporal
TEMPORAL_ADDRESS: temporal:7233
```

### What You Provide

```bash
# Only secrets - export or put in .env
export OPENAI_API_KEY=sk-...
# or
echo "OPENAI_API_KEY=sk-..." >> .env
```

### Commands

```bash
# Infrastructure only (Qdrant + Temporal)
docker compose up -d

# Infrastructure + Engram app
docker compose --profile app up -d

# Just Qdrant (for local DBOS dev)
docker compose up -d qdrant

# Verify
curl http://localhost:6333/health  # Qdrant
curl http://localhost:6366/health  # Engram API
open http://localhost:7280         # Temporal UI
```

### Service Ports

**MEMO = 6366** (M=6, E=3, M=6, O=6 on phone keypad)

| Service | Port | URL |
|---------|------|-----|
| Qdrant REST | 6333 | http://localhost:6333 |
| Qdrant gRPC | 6334 | - |
| **Engram API** | **6366** | http://localhost:6366 |
| Temporal gRPC | 7233 | - |
| Temporal UI | 7280 | http://localhost:7280 |

## Project Structure

```
engram/
├── src/
│   └── engram/
│       ├── __init__.py
│       ├── api.py              # Public API (encode, recall, etc.)
│       ├── models/
│       │   ├── __init__.py
│       │   ├── episode.py      # Episodic memory model
│       │   ├── fact.py         # Factual memory model
│       │   ├── semantic.py     # Semantic memory model
│       │   ├── procedural.py   # Procedural memory model
│       │   └── inhibitory.py   # Inhibitory memory model (negations)
│       ├── extraction/
│       │   ├── __init__.py
│       │   ├── patterns.py     # Deterministic extraction (regex)
│       │   └── semantic.py     # LLM-based extraction
│       ├── storage/
│       │   ├── __init__.py
│       │   └── qdrant.py       # Qdrant client wrapper
│       ├── agents/
│       │   ├── __init__.py
│       │   └── consolidation.py # Pydantic AI consolidation agent
│       └── config.py           # Settings management
├── tests/
│   ├── unit/
│   └── integration/
├── docs/
│   ├── architecture.md
│   └── design.md
├── research/
├── .env.example
├── .gitignore
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml
└── README.md
```

## Development Workflow

### Setup

```bash
# Clone
git clone https://github.com/ashita-ai/engram.git
cd engram

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e ".[dev]"

# Copy environment template
cp .env.example .env
# Edit .env with your API keys

# Start Qdrant
docker compose up -d qdrant
```

### Testing

```bash
# Run tests
pytest

# With coverage
pytest --cov=src/engram --cov-report=term-missing

# Type checking
mypy src/engram

# Linting
ruff check src/engram

# Format
black src/engram
```

### Pre-commit Checklist

```bash
# Format
black src/engram tests/

# Lint
ruff check src/engram tests/

# Type check
mypy src/engram

# Test
pytest

# No secrets
grep -r "sk-" src/ tests/ && echo "SECRETS FOUND" || echo "OK"
```

## LLM Provider Configuration

Engram uses Pydantic AI for LLM interactions. Configure via environment:

```bash
# Use OpenAI (default)
CONSOLIDATION_MODEL=openai:gpt-4o-mini

# Use Anthropic
CONSOLIDATION_MODEL=anthropic:claude-3-haiku-20240307

# Use Groq (fast, cheap)
CONSOLIDATION_MODEL=groq:llama-3.1-8b-instant

# Use Google
CONSOLIDATION_MODEL=google-gla:gemini-1.5-flash
```

Pydantic AI handles the provider abstraction. Just set the appropriate API key.

## Embedding Configuration

```bash
# OpenAI embeddings (recommended)
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small

# Sentence Transformers (local, free)
EMBEDDING_PROVIDER=sentence-transformers
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Cohere
EMBEDDING_PROVIDER=cohere
EMBEDDING_MODEL=embed-english-v3.0
```

## Security Notes

1. **Never commit `.env`** - It's in `.gitignore`
2. **Use `.env.example`** - Document required vars without values
3. **Rotate keys** - If accidentally committed, rotate immediately
4. **Least privilege** - Use read-only API keys where possible
5. **Validate inputs** - Pydantic models reject unknown fields
