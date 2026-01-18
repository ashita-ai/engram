"""Specificity scoring for confidence calculations.

Specificity measures the information density of text. Specific statements
with concrete details should have higher confidence than vague statements.

Examples:
- Low specificity: "I like programming" (generic)
- High specificity: "I use Python 3.12 with FastAPI for REST APIs" (concrete)

This module provides pattern-based specificity scoring as part of Phase 1
of intelligent confidence scoring (#136).

Example:
    >>> from engram.confidence.specificity import calculate_specificity
    >>> result = calculate_specificity("I use Python 3.12 with FastAPI")
    >>> result.score  # Higher = more specific
    0.75
"""

from __future__ import annotations

import re
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field


class SpecificityResult(BaseModel):
    """Result of specificity analysis.

    Attributes:
        score: Specificity score 0.0-1.0 (higher = more specific).
        word_count: Number of words in text.
        specific_count: Number of specific elements found.
        vague_count: Number of vague elements found.
        details: Breakdown of what contributed to the score.
    """

    model_config = ConfigDict(extra="forbid")

    score: float = Field(
        ge=0.0,
        le=1.0,
        description="Specificity score (higher = more specific)",
    )
    word_count: int = Field(
        ge=0,
        description="Number of words in analyzed text",
    )
    specific_count: int = Field(
        default=0,
        ge=0,
        description="Number of specific elements found",
    )
    vague_count: int = Field(
        default=0,
        ge=0,
        description="Number of vague elements found",
    )
    details: dict[str, int] = Field(
        default_factory=dict,
        description="Breakdown by category (numbers, dates, etc.)",
    )


class SpecificityScorer:
    """Score text specificity based on information density.

    Scoring factors:
    - Positive: Numbers, dates, proper nouns, technical terms, version numbers
    - Negative: Vague pronouns, generic words, filler phrases

    The final score is normalized to 0.0-1.0 where:
    - 0.0-0.3: Very vague
    - 0.3-0.5: Somewhat vague
    - 0.5-0.7: Moderately specific
    - 0.7-1.0: Highly specific
    """

    # Patterns that indicate specificity (boost score)
    NUMBER_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"\b\d+(?:\.\d+)?(?:\s*(?:GB|MB|KB|TB|GHz|MHz|ms|seconds?|minutes?|hours?|days?|weeks?|months?|years?|%|dollars?|euros?|pounds?|\$|€|£))?\b",
        re.IGNORECASE,
    )

    # Version numbers like 3.12, v2.1.0, Python 3.12
    VERSION_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"\b(?:v(?:ersion)?\.?\s*)?\d+\.\d+(?:\.\d+)?(?:[-.]?\w+)?\b",
        re.IGNORECASE,
    )

    # Date patterns (various formats)
    DATE_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"\b(?:"
        r"\d{1,2}[-/]\d{1,2}[-/]\d{2,4}"  # 01/15/2024
        r"|\d{4}[-/]\d{1,2}[-/]\d{1,2}"  # 2024-01-15
        r"|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+\d{4})?"  # Jan 15, 2024
        r"|\d{1,2}(?:st|nd|rd|th)?\s+(?:of\s+)?(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?(?:,?\s+\d{4})?"  # 15th of January
        r")\b",
        re.IGNORECASE,
    )

    # Technical terms (common in software/engineering context)
    TECH_TERMS: ClassVar[set[str]] = {
        # Programming languages
        "python",
        "javascript",
        "typescript",
        "java",
        "kotlin",
        "swift",
        "rust",
        "go",
        "golang",
        "ruby",
        "php",
        "scala",
        "haskell",
        "elixir",
        "clojure",
        # Frameworks
        "fastapi",
        "django",
        "flask",
        "express",
        "react",
        "vue",
        "angular",
        "nextjs",
        "nuxt",
        "svelte",
        "rails",
        "spring",
        "laravel",
        "phoenix",
        # Databases
        "postgresql",
        "postgres",
        "mysql",
        "mongodb",
        "redis",
        "elasticsearch",
        "sqlite",
        "cassandra",
        "dynamodb",
        "qdrant",
        "pinecone",
        "weaviate",
        # Cloud/infra
        "aws",
        "gcp",
        "azure",
        "kubernetes",
        "docker",
        "terraform",
        "ansible",
        "jenkins",
        "github",
        "gitlab",
        "circleci",
        "vercel",
        "heroku",
        "netlify",
        # AI/ML
        "openai",
        "anthropic",
        "claude",
        "gpt",
        "llama",
        "mistral",
        "langchain",
        "pytorch",
        "tensorflow",
        "huggingface",
        "transformer",
        "embedding",
        # Protocols/formats
        "rest",
        "graphql",
        "grpc",
        "websocket",
        "http",
        "https",
        "json",
        "yaml",
        "protobuf",
        "avro",
        "parquet",
        "csv",
        "xml",
    }

    # Email pattern
    EMAIL_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    )

    # URL pattern
    URL_PATTERN: ClassVar[re.Pattern[str]] = re.compile(r"\bhttps?://[^\s<>\"{}|\\^`\[\]]+\b")

    # Vague words/phrases (reduce score)
    VAGUE_WORDS: ClassVar[set[str]] = {
        "something",
        "things",
        "stuff",
        "everything",
        "anything",
        "nothing",
        "someone",
        "somebody",
        "everyone",
        "anybody",
        "nobody",
        "somewhere",
        "anywhere",
        "everywhere",
        "nowhere",
        "sometime",
        "sometimes",
        "always",
        "never",
        "often",
        "it",
        "they",
        "them",
        "those",
        "these",
        "that",
        "this",
        "etc",
        "whatever",
        "whichever",
        "whoever",
        "good",
        "bad",
        "nice",
        "great",
        "fine",
        "okay",
        "ok",
        "interesting",
        "useful",
        "helpful",
        "different",
    }

    # Filler phrases (reduce score)
    FILLER_PATTERNS: ClassVar[list[re.Pattern[str]]] = [
        re.compile(r"\byou know\b", re.IGNORECASE),
        re.compile(r"\bkind of\b", re.IGNORECASE),
        re.compile(r"\bsort of\b", re.IGNORECASE),
        re.compile(r"\blike\s+(?:really|very|so)\b", re.IGNORECASE),
        re.compile(r"\bbasically\b", re.IGNORECASE),
        re.compile(r"\bessentially\b", re.IGNORECASE),
        re.compile(r"\bactually\b", re.IGNORECASE),
        re.compile(r"\bjust\b", re.IGNORECASE),
    ]

    # Weights for different specificity signals
    NUMBER_WEIGHT: ClassVar[float] = 0.15
    VERSION_WEIGHT: ClassVar[float] = 0.20
    DATE_WEIGHT: ClassVar[float] = 0.15
    TECH_WEIGHT: ClassVar[float] = 0.10
    EMAIL_WEIGHT: ClassVar[float] = 0.15
    URL_WEIGHT: ClassVar[float] = 0.15
    VAGUE_PENALTY: ClassVar[float] = -0.05
    FILLER_PENALTY: ClassVar[float] = -0.03

    def score(self, text: str) -> SpecificityResult:
        """Calculate specificity score for the given text.

        Analyzes text for specific and vague elements, then computes
        a normalized score.

        Args:
            text: Source text to analyze.

        Returns:
            SpecificityResult with score and details.

        Example:
            >>> scorer = SpecificityScorer()
            >>> result = scorer.score("I use Python 3.12 with FastAPI")
            >>> result.score > 0.5
            True
        """
        if not text or not text.strip():
            return SpecificityResult(score=0.5, word_count=0)

        text_lower = text.lower()
        words = text.split()
        word_count = len(words)

        # Start with base score
        base_score = 0.5
        adjustments = 0.0
        details: dict[str, int] = {}
        specific_count = 0
        vague_count = 0

        # Count numbers
        numbers = self.NUMBER_PATTERN.findall(text)
        if numbers:
            count = len(numbers)
            details["numbers"] = count
            adjustments += min(count * self.NUMBER_WEIGHT, 0.3)
            specific_count += count

        # Count version numbers (separate from general numbers)
        versions = self.VERSION_PATTERN.findall(text)
        if versions:
            count = len(versions)
            details["versions"] = count
            adjustments += min(count * self.VERSION_WEIGHT, 0.2)
            specific_count += count

        # Count dates
        dates = self.DATE_PATTERN.findall(text)
        if dates:
            count = len(dates)
            details["dates"] = count
            adjustments += min(count * self.DATE_WEIGHT, 0.2)
            specific_count += count

        # Count technical terms
        tech_count = sum(1 for word in text_lower.split() if word in self.TECH_TERMS)
        if tech_count:
            details["tech_terms"] = tech_count
            adjustments += min(tech_count * self.TECH_WEIGHT, 0.3)
            specific_count += tech_count

        # Count emails
        emails = self.EMAIL_PATTERN.findall(text)
        if emails:
            count = len(emails)
            details["emails"] = count
            adjustments += min(count * self.EMAIL_WEIGHT, 0.2)
            specific_count += count

        # Count URLs
        urls = self.URL_PATTERN.findall(text)
        if urls:
            count = len(urls)
            details["urls"] = count
            adjustments += min(count * self.URL_WEIGHT, 0.2)
            specific_count += count

        # Count vague words
        vague_word_count = sum(
            1 for word in text_lower.split() if word.strip(".,!?;:") in self.VAGUE_WORDS
        )
        if vague_word_count:
            details["vague_words"] = vague_word_count
            adjustments += max(vague_word_count * self.VAGUE_PENALTY, -0.2)
            vague_count += vague_word_count

        # Count filler phrases
        filler_count = sum(1 for pattern in self.FILLER_PATTERNS if pattern.search(text))
        if filler_count:
            details["fillers"] = filler_count
            adjustments += max(filler_count * self.FILLER_PENALTY, -0.1)
            vague_count += filler_count

        # Calculate final score with bounds
        final_score = min(1.0, max(0.0, base_score + adjustments))

        return SpecificityResult(
            score=final_score,
            word_count=word_count,
            specific_count=specific_count,
            vague_count=vague_count,
            details=details,
        )


# Module-level convenience instance
_default_scorer: SpecificityScorer | None = None


def calculate_specificity(text: str) -> SpecificityResult:
    """Calculate text specificity using default scorer.

    Convenience function that uses a module-level scorer instance.

    Args:
        text: Source text to analyze.

    Returns:
        SpecificityResult with score and details.

    Example:
        >>> result = calculate_specificity("Deploy to AWS us-east-1 by January 15")
        >>> result.score > 0.5
        True
    """
    global _default_scorer
    if _default_scorer is None:
        _default_scorer = SpecificityScorer()
    return _default_scorer.score(text)
