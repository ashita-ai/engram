# Engram: Memory That Actually Works

**January 2026**

---

## The Accuracy Problem

AI memory systems exist. The problem is they're wrong more than half the time.

The HaluMem benchmark tested leading memory systems and found accuracy below 56%, with omission rates exceeding 50%. This isn't edge cases—it's the baseline.

Why? Most systems follow a pattern: conversation happens, LLM extracts "facts," facts go into a database. The original words get thrown away. Every extraction pass can introduce mistakes. Every summarization loses detail. After a few iterations, you're querying a game of telephone.

## What Engram Does Differently

Engram never throws away what you actually said.

Think of it like your own memory. You remember the conversation with your friend (the episode), but you also remember the takeaway (the meaning). If someone questions your takeaway, you can go back to what was actually said. That's the approach.

```
What you said  →  Stored forever, untouched
       ↓
What it means  →  Extracted, but always traceable back
```

## How Memories Get Smarter

Here's where it gets interesting. Engram isn't just storage—it's a system that learns.

**Memories that matter stick around.** Every time a memory gets used—retrieved, linked to something new, confirmed by new information—it gets stronger. Just like how you remember things better when you actually use them. (This is real science: the Testing Effect is one of the most robust findings in memory research.)

**Related memories find each other.** When you say "prefer Python for scripts," that automatically links to the memory about "using Python at work." No manual organization needed. The system figures out what goes together.

**Irrelevant stuff fades away.** That random detail from six months ago that never came up again? It slowly fades, just like real memory. This isn't a bug—it's how memory is supposed to work. It keeps the important stuff front and center.

**Retrieval has consequences.** When you search for something and get results back, the memories that *almost* matched but didn't make the cut get slightly weaker. This naturally prunes redundant or outdated information. (Psychologists call this Retrieval-Induced Forgetting, and yes, your brain does this too.)

## The Result

Put it all together and you get a memory system where:

- **Frequently-used facts** rise to the top automatically
- **Stale information** fades without you doing anything
- **Wrong extractions** can be traced back and fixed
- **Related knowledge** connects itself

You don't manage it. It manages itself.

## Why This Matters

Every AI agent needs memory to be useful. But memory that hallucinates is worse than no memory at all—it confidently tells you wrong things.

Engram exists because AI memory shouldn't:
- Forget what you literally just told it
- Confidently state things you never said
- Treat a wild guess the same as a direct quote
- Grow forever until it's useless

Memory should work like memory. It should remember what matters, forget what doesn't, and always let you check the receipts.

That's Engram.

---

*Want the technical details? See [architecture.md](../docs/architecture.md) and [research foundations](overview.md).*
