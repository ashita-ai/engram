# Engram Development Guidelines

## Security (CRITICAL)

**NEVER commit secrets to the repository.**

### Prohibited:
- `.env` files (use `.env.example` with placeholders only)
- API keys, tokens, passwords in ANY file
- Credentials in code, docs, examples, tests, or config
- Hardcoded URLs with auth tokens

### Required:
- All secrets via environment variables
- `.env` is in `.gitignore` (verified)
- `.env.example` has placeholder values only
- Pre-commit check: `grep -r "sk-\|sk-ant-\|AIza" src/ tests/ && echo "SECRETS FOUND" || echo "OK"`

### If secrets are accidentally committed:
1. Do NOT just delete and commit again (still in git history)
2. Rotate the exposed credentials immediately
3. Use `git filter-branch` or BFG to remove from history
4. Force push (with team coordination)

## Scientific Claims

Be careful about claims regarding cognitive science foundations. Engram is **inspired by** cognitive science, not a strict implementation of it.

### What we can claim:
- Multiple memory types are a useful engineering abstraction
- Ground truth preservation solves a real problem (LLM extraction errors)
- Confidence tracking distinguishes certain from uncertain
- Deferred processing reduces cost and latency
- Some form of forgetting is necessary for relevance and performance

### What we should NOT claim:
- That our architecture mirrors how the brain actually works
- That episodic/semantic are cleanly separable (they're not — it's a continuum)
- That consolidation works exactly as we model it (heavily debated)
- That Ebbinghaus decay is the definitive model of forgetting (it's a simplification)
- That Atkinson-Shiffrin is current (Baddeley's Working Memory Model superseded it)

### When writing docs:
- Use "inspired by" not "based on" when referencing cognitive science
- Acknowledge limitations and ongoing debates in the field
- Focus on the practical engineering problems we solve
- Don't overstate the scientific backing for specific design choices

## Memory Types

The six memory types are engineering constructs:
- **Working, Episodic, Semantic, Procedural** — Inspired by cognitive science
- **Factual** — Engineering subdivision (verbatim vs inferred) not from cognitive science
- **Inhibitory** — Inspired by CCK+ interneurons in memory selectivity (Tomé et al.)

Be explicit about which are science-inspired and which are engineering additions.
