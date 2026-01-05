# Engram Examples

This directory contains example scripts demonstrating Engram's key features.

## Directory Structure

```
examples/
├── local/          # No external dependencies (no Qdrant, no API keys)
│   ├── extraction_demo.py
│   └── confidence_demo.py
└── external/       # Requires Qdrant and/or API keys
    ├── quickstart.py
    ├── multi_tenant.py
    ├── api_client.py
    └── consolidation_demo.py  # Requires OpenAI API key
```

## Quick Start

### Local Examples (No Setup Required)

These examples run without any external dependencies:

```bash
# Pattern extraction demo
python examples/local/extraction_demo.py

# Confidence scoring demo
python examples/local/confidence_demo.py
```

### External Examples (Requires Qdrant)

Most external examples require Qdrant running locally:

```bash
# Start Qdrant
docker run -p 6333:6333 qdrant/qdrant

# Run examples with FastEmbed (free, local embeddings)
ENGRAM_EMBEDDING_PROVIDER=fastembed python examples/external/quickstart.py
ENGRAM_EMBEDDING_PROVIDER=fastembed python examples/external/multi_tenant.py
```

### LLM-Powered Examples (Requires OpenAI API Key)

The consolidation demo requires an OpenAI API key:

```bash
ENGRAM_OPENAI_API_KEY=sk-... \
ENGRAM_EMBEDDING_PROVIDER=fastembed \
python examples/external/consolidation_demo.py
```

---

## Local Examples

### 🔍 Extraction Pipeline
**`local/extraction_demo.py`** - All 8 extractors in action

Demonstrates:
- EmailExtractor (email-validator)
- PhoneExtractor (phonenumbers)
- URLExtractor (validators)
- DateExtractor (dateparser)
- QuantityExtractor (Pint)
- LanguageExtractor (langdetect)
- NameExtractor (nameparser)
- IDExtractor (python-stdnum)

### 📊 Confidence Scoring
**`local/confidence_demo.py`** - Understanding confidence levels

Explains:
- Extraction methods (VERBATIM, EXTRACTED, INFERRED)
- Confidence calculation formula
- Decay over time

---

## External Examples

### 🚀 Quickstart
**`external/quickstart.py`** - Basic encode/recall workflow

Shows:
- Initializing EngramService
- Storing memories with `encode()`
- Semantic search with `recall()`
- Automatic fact extraction

### 🔒 Multi-Tenancy
**`external/multi_tenant.py`** - Data isolation between users and orgs

Shows:
- User-level isolation (`user_id`)
- Organization-level isolation (`org_id`)
- Same user in multiple organizations

### 🌐 REST API Client
**`external/api_client.py`** - Using the HTTP API

```bash
# Terminal 1: Start the server
ENGRAM_EMBEDDING_PROVIDER=fastembed uvicorn engram.api:app --reload

# Terminal 2: Run the client
python examples/external/api_client.py
```

Demonstrates:
- Health check endpoint
- Encode via HTTP POST
- Recall via HTTP POST

### 🧠 Consolidation (LLM-Powered)
**`external/consolidation_demo.py`** - Semantic knowledge extraction

**Requires OpenAI API key!** This is the only example that makes external LLM calls.

```bash
ENGRAM_OPENAI_API_KEY=sk-... \
ENGRAM_EMBEDDING_PROVIDER=fastembed \
python examples/external/consolidation_demo.py
```

Shows:
- Storing raw conversation episodes
- Running LLM consolidation with GPT-4o-mini
- Extracting semantic memories from episodes
- Querying consolidated knowledge

---

## Requirements Summary

| Example | Qdrant | Embeddings | LLM API |
|---------|--------|------------|---------|
| local/extraction_demo.py | - | - | - |
| local/confidence_demo.py | - | - | - |
| external/quickstart.py | ✅ | FastEmbed or OpenAI | - |
| external/multi_tenant.py | ✅ | FastEmbed or OpenAI | - |
| external/api_client.py | ✅ | FastEmbed or OpenAI | - |
| external/consolidation_demo.py | ✅ | FastEmbed or OpenAI | ✅ OpenAI |

## Using FastEmbed (Free Local Embeddings)

For examples that require embeddings but you don't have an OpenAI key:

```bash
export ENGRAM_EMBEDDING_PROVIDER=fastembed
python examples/external/quickstart.py
```

FastEmbed downloads a small model (~50MB) on first use and runs locally.
