# Engram Examples

This directory contains example scripts demonstrating Engram's key features.

## Prerequisites

Most examples require:
1. **Qdrant** running locally:
   ```bash
   docker run -p 6333:6333 qdrant/qdrant
   ```

2. **Embeddings** - either:
   - OpenAI API key: `export OPENAI_API_KEY=sk-...`
   - Or FastEmbed (local, free): `export ENGRAM_EMBEDDING_PROVIDER=fastembed`

## Examples

### 🚀 Quickstart
**`quickstart.py`** - Basic encode/recall workflow

```bash
python examples/quickstart.py
```

Shows:
- Initializing EngramService
- Storing memories with `encode()`
- Semantic search with `recall()`
- Automatic fact extraction

### 🔍 Extraction Pipeline
**`extraction_demo.py`** - All 8 extractors in action

```bash
python examples/extraction_demo.py
```

Demonstrates:
- EmailExtractor (email-validator)
- PhoneExtractor (phonenumbers)
- URLExtractor (validators)
- DateExtractor (dateparser)
- QuantityExtractor (Pint)
- LanguageExtractor (langdetect)
- NameExtractor (nameparser)
- IDExtractor (python-stdnum)

### 🔒 Multi-Tenancy
**`multi_tenant.py`** - Data isolation between users and orgs

```bash
python examples/multi_tenant.py
```

Shows:
- User-level isolation (`user_id`)
- Organization-level isolation (`org_id`)
- Same user in multiple organizations

### 🌐 REST API Client
**`api_client.py`** - Using the HTTP API

```bash
# Terminal 1: Start the server
uvicorn engram.api:app --reload

# Terminal 2: Run the client
python examples/api_client.py
```

Demonstrates:
- Health check endpoint
- Encode via HTTP POST
- Recall via HTTP POST

### 📊 Confidence Scoring
**`confidence_demo.py`** - Understanding confidence levels

```bash
python examples/confidence_demo.py
```

Explains:
- Extraction methods (VERBATIM, EXTRACTED, INFERRED)
- Confidence calculation formula
- Decay over time

## Quick Reference

| Example | Requires Qdrant | Requires API Key | Topic |
|---------|----------------|------------------|-------|
| quickstart.py | ✅ | ✅ or FastEmbed | Core workflow |
| extraction_demo.py | ❌ | ❌ | Pattern extraction |
| multi_tenant.py | ✅ | ✅ or FastEmbed | Data isolation |
| api_client.py | ✅ | ✅ or FastEmbed | REST API |
| confidence_demo.py | ❌ | ❌ | Confidence scoring |

## Using FastEmbed (No API Key)

For examples that require embeddings but you don't have an OpenAI key:

```bash
export ENGRAM_EMBEDDING_PROVIDER=fastembed
python examples/quickstart.py
```

FastEmbed downloads a small model (~50MB) on first use and runs locally.
