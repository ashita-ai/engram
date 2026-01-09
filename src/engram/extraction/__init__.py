"""Fact extraction from episodes.

This module provides deterministic pattern matching to extract
facts from episode content. All extractors produce Facts with
EXTRACTED confidence (0.9).

Example:
    ```python
    from engram.extraction import ExtractionPipeline, default_pipeline
    from engram.models import Episode

    # Use default pipeline with all extractors
    episode = Episode(content="Email me at user@example.com", ...)
    facts = default_pipeline().run(episode)

    # Or create custom pipeline
    from engram.extraction import EmailExtractor, PhoneExtractor

    pipeline = ExtractionPipeline([
        EmailExtractor(),
        PhoneExtractor(),
    ])
    facts = pipeline.run(episode)
    ```
"""

from .base import ExtractionPipeline, ExtractionResult, Extractor
from .date import DateExtractor
from .email import EmailExtractor
from .language import LanguageExtractor
from .name import NameExtractor
from .negation import NegationDetector, NegationMatch, create_negation_facts, detect_negations
from .phone import PhoneExtractor
from .quantity import QuantityExtractor
from .stdnum import IDExtractor
from .url import URLExtractor


def default_pipeline() -> ExtractionPipeline:
    """Create a pipeline with all default extractors.

    Returns:
        ExtractionPipeline configured with all 8 extractors.
    """
    return ExtractionPipeline(
        [
            EmailExtractor(),
            PhoneExtractor(),
            URLExtractor(),
            DateExtractor(),
            QuantityExtractor(),
            LanguageExtractor(),
            NameExtractor(),
            IDExtractor(),
        ]
    )


__all__ = [
    # Base classes
    "Extractor",
    "ExtractionPipeline",
    "ExtractionResult",
    # Extractors
    "EmailExtractor",
    "PhoneExtractor",
    "URLExtractor",
    "DateExtractor",
    "QuantityExtractor",
    "LanguageExtractor",
    "NameExtractor",
    "IDExtractor",
    # Negation detection
    "NegationDetector",
    "NegationMatch",
    "detect_negations",
    "create_negation_facts",
    # Factory
    "default_pipeline",
]
