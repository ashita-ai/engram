"""Language detection extractor.

Uses langdetect library for language identification.
"""

from __future__ import annotations

from langdetect import LangDetectException, detect_langs  # type: ignore[import-untyped]

from engram.models import Episode, Fact

from .base import Extractor


class LanguageExtractor(Extractor):
    """Detect language of episode content.

    Uses langdetect library for:
    - 55 language support (ISO 639-1 codes)
    - Probability-based detection
    - Short text handling

    Note:
        This extractor returns a single fact with the detected language code.
        It operates on the entire episode content, not substrings.

    Example:
        ```python
        extractor = LanguageExtractor()
        facts = extractor.extract(episode)
        # facts[0].content = "en"
        # facts[0].category = "language"
        ```
    """

    name: str = "language"
    category: str = "language"

    def __init__(self, min_confidence: float = 0.8) -> None:
        """Initialize language extractor.

        Args:
            min_confidence: Minimum confidence threshold for detection.
                           Only returns a result if confidence >= threshold.
        """
        self.min_confidence = min_confidence

    def extract(self, episode: Episode) -> list[Fact]:
        """Detect language of episode content.

        Args:
            episode: Episode containing text to analyze.

        Returns:
            List with single Fact containing language code, or empty list
            if detection failed or confidence is below threshold.
        """
        content = episode.content.strip()

        # Need sufficient text for detection
        if len(content) < 10:
            return []

        try:
            # Get language probabilities
            lang_probs = detect_langs(content)

            if not lang_probs:
                return []

            # Get top language
            top_lang = lang_probs[0]

            # Check confidence threshold
            if top_lang.prob < self.min_confidence:
                return []

            # Return language code (e.g., "en", "es", "fr")
            return [self._create_fact(top_lang.lang, episode)]

        except LangDetectException:
            # Detection failed (empty text, no features, etc.)
            return []
