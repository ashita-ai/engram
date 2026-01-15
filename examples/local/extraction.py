#!/usr/bin/env python3
"""Pattern extraction demo.

Demonstrates Engram's deterministic extraction pipeline:
- 3 pattern extractors (email, phone, URL)
- How extraction feeds into StructuredMemory
- Rich mode with LLM extraction

No external dependencies required - runs entirely locally.
"""

from engram.extraction import (
    EmailExtractor,
    PhoneExtractor,
    URLExtractor,
)
from engram.models import Episode


def main() -> None:
    print("=" * 70)
    print("Engram Pattern Extraction Demo")
    print("=" * 70)

    # =========================================================================
    # Part 1: Individual Extractors
    # =========================================================================
    print("\n1. PATTERN EXTRACTORS")
    print("-" * 70)
    print("  Deterministic regex extractors - no LLM, no hallucinations.\n")

    extractors = [
        ("Email", EmailExtractor(), "Contact me at alice@example.com or bob@corp.io"),
        ("Phone", PhoneExtractor(), "Call me at (415) 555-1234 or +1-800-555-0199"),
        ("URL", URLExtractor(), "Check out https://github.com/ashita-ai/engram"),
    ]

    for name, extractor, text in extractors:
        print(f"  {name}Extractor")
        print(f'  Input: "{text}"')
        episode = Episode(content=text, role="user", user_id="demo", org_id="demo_org")
        results = extractor.extract(episode)
        if results:
            for item in results:
                print(f"  -> {item}")
        else:
            print("  -> (no matches)")
        print()

    # =========================================================================
    # Part 2: Combined Extraction
    # =========================================================================
    print("\n2. COMBINED EXTRACTION")
    print("-" * 70)

    complex_message = """
    Hi, I'm Alex Johnson and my email is alex.j@techcorp.com.
    You can reach me at +1-415-555-6543.
    More info at https://techcorp.com/projects.
    """

    print(f"  Input: {complex_message.strip()}\n")
    episode = Episode(content=complex_message, role="user", user_id="demo", org_id="demo_org")

    emails = EmailExtractor().extract(episode)
    phones = PhoneExtractor().extract(episode)
    urls = URLExtractor().extract(episode)

    print("  Extracted:")
    print(f"    Emails: {emails}")
    print(f"    Phones: {phones}")
    print(f"    URLs:   {urls}")

    # =========================================================================
    # Part 3: How Extraction Feeds Into StructuredMemory
    # =========================================================================
    print("\n\n3. HOW EXTRACTION FEEDS INTO STRUCTURED MEMORY")
    print("-" * 70)
    print("""
  When you call engram.encode(), extraction happens automatically:

  Episode (raw text)
      │
      ├── Regex extractors run immediately
      │   └── emails, phones, URLs → StructuredMemory
      │
      └── If enrich=True, LLM extraction runs
          └── dates, people, preferences, negations → StructuredMemory

  Example:
  ```python
  # Fast mode (default) - regex only
  result = await engram.encode(content="My email is x@y.com", ...)
  print(result.structured.emails)  # ["x@y.com"]

  # Rich mode - regex + LLM
  result = await engram.encode(content="...", enrich=True)
  print(result.structured.people)  # [Person(name="Alice", role="manager")]
  ```
    """)

    # =========================================================================
    # Part 4: Why Pattern Extraction Matters
    # =========================================================================
    print("\n4. WHY PATTERN EXTRACTION MATTERS")
    print("-" * 70)
    print("""
  Engram extracts patterns BEFORE using LLMs:

  | Approach           | Confidence | Hallucination Risk |
  |--------------------|------------|-------------------|
  | Pattern extraction | 90%        | Zero              |
  | LLM extraction     | 80%        | Possible          |

  Pattern-matched extracts are:
  - Deterministic (same input = same output)
  - Auditable (regex patterns are inspectable)
  - Fast (no API calls needed)
  - Trustworthy (no hallucination possible)

  LLM enrichment runs optionally for richer extraction:
  - People with roles
  - Preferences with sentiment
  - Dates resolved to ISO format
  - Negations with context
    """)

    print("=" * 70)
    print("Demo complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
