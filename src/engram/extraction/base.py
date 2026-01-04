"""Base classes for fact extraction from episodes.

Extractors use deterministic pattern matching (regex, validators) to identify
facts in episode content. All extracted facts receive EXTRACTED confidence (0.9).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from engram.models import Episode, Fact
from engram.models.base import ConfidenceScore

if TYPE_CHECKING:
    from collections.abc import Sequence


class ExtractionResult(BaseModel):
    """Result from a single extractor run.

    Attributes:
        facts: List of extracted facts.
        extractor_name: Name of the extractor that produced these facts.
    """

    model_config = ConfigDict(extra="forbid")

    facts: list[Fact] = Field(default_factory=list)
    extractor_name: str = Field(description="Name of the extractor")


class Extractor(ABC):
    """Abstract base class for fact extractors.

    Extractors identify specific patterns in episode content and create
    Facts with EXTRACTED confidence (0.9). Each extractor focuses on
    one category of facts (emails, phones, dates, etc.).

    Example:
        ```python
        class EmailExtractor(Extractor):
            name = "email"
            category = "contact"

            def extract(self, episode: Episode) -> list[Fact]:
                # Find emails in episode.content
                ...
        ```
    """

    name: str = "base"
    category: str = "general"

    @abstractmethod
    def extract(self, episode: Episode) -> list[Fact]:
        """Extract facts from an episode.

        Args:
            episode: The episode to extract facts from.

        Returns:
            List of extracted Facts. Empty list if no matches found.
        """
        ...

    def _create_fact(
        self,
        content: str,
        episode: Episode,
        category: str | None = None,
    ) -> Fact:
        """Create a Fact from extracted content.

        Helper method that creates a properly configured Fact with
        EXTRACTED confidence and correct source linkage.

        Args:
            content: The extracted fact content.
            episode: Source episode.
            category: Optional category override (defaults to self.category).

        Returns:
            Configured Fact instance.
        """
        return Fact(
            content=content,
            category=category or self.category,
            source_episode_id=episode.id,
            user_id=episode.user_id,
            org_id=episode.org_id,
            confidence=ConfidenceScore.for_extracted(),
        )


class ExtractionPipeline:
    """Pipeline that runs multiple extractors on episodes.

    The pipeline coordinates multiple extractors and aggregates their
    results. Each extractor runs independently and all results are
    collected into a single list.

    Example:
        ```python
        pipeline = ExtractionPipeline([
            EmailExtractor(),
            PhoneExtractor(),
            DateExtractor(),
        ])

        facts = pipeline.run(episode)
        ```

    Attributes:
        extractors: List of extractors to run.
    """

    def __init__(self, extractors: Sequence[Extractor] | None = None) -> None:
        """Initialize the pipeline with extractors.

        Args:
            extractors: List of extractors. Defaults to empty list.
        """
        self._extractors: list[Extractor] = list(extractors) if extractors else []

    @property
    def extractors(self) -> list[Extractor]:
        """Get the list of extractors."""
        return self._extractors

    def add_extractor(self, extractor: Extractor) -> None:
        """Add an extractor to the pipeline.

        Args:
            extractor: Extractor to add.
        """
        self._extractors.append(extractor)

    def run(self, episode: Episode) -> list[Fact]:
        """Run all extractors on an episode.

        Args:
            episode: Episode to extract facts from.

        Returns:
            List of all facts extracted by all extractors.
        """
        all_facts: list[Fact] = []

        for extractor in self._extractors:
            facts = extractor.extract(episode)
            all_facts.extend(facts)

        return all_facts

    def run_with_results(self, episode: Episode) -> list[ExtractionResult]:
        """Run all extractors and return detailed results.

        Unlike run(), this method returns results grouped by extractor,
        which is useful for debugging and auditing.

        Args:
            episode: Episode to extract facts from.

        Returns:
            List of ExtractionResult, one per extractor.
        """
        results: list[ExtractionResult] = []

        for extractor in self._extractors:
            facts = extractor.extract(episode)
            results.append(
                ExtractionResult(
                    facts=facts,
                    extractor_name=extractor.name,
                )
            )

        return results
