# Procedural Memory

Why Engram tracks behavioral preferences.

## The Argument

Users have implicit preferences they may never explicitly state:
- "I prefer concise responses"
- "I like code examples"
- "I work better with bullet points"

These are learned through repeated interactions, not extracted from single statements. Procedural memory captures this.

## What is Procedural Memory?

Procedural memory stores "knowing how" rather than "knowing that":

> "Procedural memory is a type of long-term implicit memory that stores information about how to perform tasks. Unlike declarative memory, it operates without conscious awareness."
> — [Squire, 2004](https://pubmed.ncbi.nlm.nih.gov/15450156/)

Key characteristics:
- **Implicit** — Operates without conscious awareness
- **Automatic** — Expressed through behavior, not verbal recall
- **Slow to acquire** — Requires repetition and practice
- **Resistant to forgetting** — Once learned, highly stable

## Evidence: The H.M. Case

Patient H.M. (Henry Molaison) had bilateral hippocampal damage:
- Could not form new episodic memories
- Could not learn new facts
- **But** could learn new motor skills (mirror drawing)
- Each session, he improved despite having no memory of practicing

> "The distinction was first demonstrated by Milner (1962), showing that the amnesic patient H.M. could learn motor skills despite having no memory of practicing them."
> — [Milner, 1962](https://www.sciencedirect.com/topics/neuroscience/implicit-memory)

This proves procedural memory is a distinct system.

## Declarative vs Procedural

| Declarative | Procedural |
|-------------|------------|
| "Knowing that" | "Knowing how" |
| Explicit, conscious | Implicit, automatic |
| Fast to acquire | Slow to acquire |
| Easy to verbalize | Difficult to verbalize |
| Flexible | Inflexible |

Human examples:
- Riding a bicycle
- Typing on a keyboard
- Speaking grammatically (vs knowing grammar rules)

## Engram Application

Procedural memory captures learned behavioral preferences:

```python
# Examples of procedural memories
{
    "type": "procedural",
    "pattern": "response_style",
    "preference": "concise",
    "confidence": 0.85,
    "evidence_count": 23,  # Built from many interactions
    "last_updated": "2024-01-15"
}

{
    "type": "procedural",
    "pattern": "code_examples",
    "preference": "include_examples",
    "confidence": 0.92,
    "evidence_count": 47
}
```

**How procedural memories are created**:
1. Not from single statements
2. Inferred from repeated patterns across episodes
3. Built up slowly over many interactions
4. High threshold for creation (requires consistent evidence)

**How they're used**:
1. Activated by context matching, not direct query
2. Applied automatically during response generation
3. Influence response style, format, depth

## Procedural vs Semantic Preferences

| Semantic Preference | Procedural Preference |
|--------------------|----------------------|
| "User said they like Python" | User consistently chooses Python examples |
| Explicit statement | Inferred from behavior |
| Created from one episode | Built from many episodes |
| Can be wrong (user changed mind) | Reflects actual behavior |

## Decay Characteristics

Procedural memories are highly stable:
- **Slow to form** — Require consistent evidence
- **Slow to decay** — Once established, resist forgetting
- **Can be overwritten** — New patterns eventually replace old

This matches biological procedural memory: skills acquired slowly are retained for years.

## References

- [Squire (2004). Memory systems of the brain](https://pubmed.ncbi.nlm.nih.gov/15450156/)
- [Milner (1962). Implicit memory in H.M.](https://www.sciencedirect.com/topics/neuroscience/implicit-memory)
- [Anderson (1982). Acquisition of cognitive skill](https://psycnet.apa.org/record/1983-24235-001)
- [PMC: Phases of procedural learning and memory](https://pmc.ncbi.nlm.nih.gov/articles/PMC8048153/)
