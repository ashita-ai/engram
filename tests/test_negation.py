"""Unit tests for Engram negation detection.

Tests the 5 core negation patterns:
1. "I don't/do not use/like/want/need/prefer X"
2. "I'm not a/an X"
3. "I'm not interested in X"
4. "I never use/used X"
5. "I/We no longer use/support X"

Edge cases are handled by LLM during consolidation, not pattern matching.
"""

from engram.extraction.negation import (
    NegationDetector,
    NegationMatch,
    create_negation_facts,
    detect_negations,
)
from engram.models import Episode, NegationFact
from engram.models.base import ExtractionMethod


def make_episode(content: str) -> Episode:
    """Create a test episode with given content."""
    return Episode(
        content=content,
        role="user",
        user_id="user_123",
    )


class TestDetectNegations:
    """Tests for detect_negations function - core patterns only."""

    # Pattern 1: I don't/do not use/like/want/need/prefer X

    def test_dont_use_pattern(self):
        """Should detect 'I don't use X' patterns."""
        matches = detect_negations("I don't use MongoDB for this project")
        assert len(matches) == 1
        assert matches[0].negated_term == "mongodb"
        assert "don't use MongoDB" in matches[0].statement

    def test_do_not_use_pattern(self):
        """Should detect 'I do not use X' patterns."""
        matches = detect_negations("I do not use Java anymore")
        assert len(matches) == 1
        assert matches[0].negated_term == "java"

    def test_dont_like_pattern(self):
        """Should detect 'don't like' negations."""
        matches = detect_negations("I don't like MongoDB")
        assert len(matches) == 1
        assert matches[0].negated_term == "mongodb"

    def test_dont_want_pattern(self):
        """Should detect 'don't want' negations."""
        matches = detect_negations("I don't want JavaScript")
        assert len(matches) == 1
        assert matches[0].negated_term == "javascript"

    def test_dont_need_pattern(self):
        """Should detect 'don't need' negations."""
        matches = detect_negations("I don't need TypeScript")
        assert len(matches) == 1
        assert matches[0].negated_term == "typescript"

    def test_dont_prefer_pattern(self):
        """Should detect 'don't prefer' negations."""
        matches = detect_negations("I don't prefer MySQL")
        assert len(matches) == 1
        assert matches[0].negated_term == "mysql"

    # Pattern 2: I'm not a/an X

    def test_not_a_user_pattern(self):
        """Should detect 'I'm not a X' patterns."""
        matches = detect_negations("I'm not a Python developer")
        assert len(matches) == 1
        assert matches[0].negated_term == "python"

    def test_i_am_not_pattern(self):
        """Should detect 'I am not a X' patterns."""
        matches = detect_negations("I am not a Java programmer")
        assert len(matches) == 1
        assert matches[0].negated_term == "java"

    # Pattern 3: I'm not interested in X

    def test_not_interested_pattern(self):
        """Should detect 'I'm not interested in X' patterns."""
        matches = detect_negations("I'm not interested in blockchain")
        assert len(matches) == 1
        assert matches[0].negated_term == "blockchain"

    def test_i_am_not_interested_pattern(self):
        """Should detect 'I am not interested in X' patterns."""
        matches = detect_negations("I am not interested in crypto")
        assert len(matches) == 1
        assert matches[0].negated_term == "crypto"

    # Pattern 4: I never use/used X

    def test_never_use_pattern(self):
        """Should detect 'I never use X' patterns."""
        matches = detect_negations("I never use Windows")
        assert len(matches) == 1
        assert matches[0].negated_term == "windows"

    def test_never_used_pattern(self):
        """Should detect 'I never used X' patterns."""
        matches = detect_negations("I never used Ruby in production")
        assert len(matches) == 1
        assert matches[0].negated_term == "ruby"

    # Pattern 5: I/We no longer use/support X

    def test_no_longer_use_pattern(self):
        """Should detect 'I no longer use X' patterns."""
        matches = detect_negations("I no longer use Vim")
        assert len(matches) == 1
        assert matches[0].negated_term == "vim"

    def test_we_no_longer_support_pattern(self):
        """Should detect 'We no longer support X' patterns."""
        matches = detect_negations("We no longer support Python 2")
        assert len(matches) == 1
        assert matches[0].negated_term == "python 2"

    # Edge cases

    def test_multiple_negations(self):
        """Should detect multiple negations in text."""
        text = "I don't use MongoDB and I never used Oracle"
        matches = detect_negations(text)
        assert len(matches) == 2
        terms = {m.negated_term for m in matches}
        assert "mongodb" in terms
        assert "oracle" in terms

    def test_no_negations(self):
        """Should return empty list when no negations found."""
        matches = detect_negations("I love using Python for everything")
        assert matches == []

    def test_empty_text(self):
        """Should handle empty text."""
        matches = detect_negations("")
        assert matches == []

    def test_case_insensitive(self):
        """Should match patterns case-insensitively."""
        matches = detect_negations("i DON'T USE mongodb")
        assert len(matches) == 1
        assert matches[0].negated_term == "mongodb"

    def test_span_tracking(self):
        """Should track correct character positions."""
        text = "Some text. I don't use MongoDB. More text."
        matches = detect_negations(text)
        assert len(matches) == 1
        start, end = matches[0].span
        assert text[start:end] == matches[0].statement

    def test_overlapping_patterns_handled(self):
        """Should not produce overlapping matches."""
        text = "I don't like Python and I don't use Python"
        matches = detect_negations(text)
        # Check no overlapping spans
        spans = [m.span for m in matches]
        for i, span1 in enumerate(spans):
            for span2 in spans[i + 1 :]:
                assert span1[1] <= span2[0] or span2[1] <= span1[0], "Spans overlap"

    def test_hyphenated_term(self):
        """Should handle hyphenated terms."""
        matches = detect_negations("I don't use type-script")
        assert len(matches) == 1
        assert matches[0].negated_term == "type-script"

    def test_underscored_term(self):
        """Should handle underscored terms."""
        matches = detect_negations("I don't use my_sql")
        assert len(matches) == 1
        assert matches[0].negated_term == "my_sql"

    def test_numeric_suffix(self):
        """Should handle terms with numbers."""
        matches = detect_negations("I don't use Python3")
        assert len(matches) == 1
        assert matches[0].negated_term == "python3"

    def test_version_number(self):
        """Should handle version numbers in terms."""
        matches = detect_negations("We no longer support Python 2.7")
        assert len(matches) == 1
        assert "python 2" in matches[0].negated_term.lower()


class TestNegationMatch:
    """Tests for NegationMatch dataclass."""

    def test_negation_match_creation(self):
        """Should create NegationMatch with all fields."""
        match = NegationMatch(
            statement="I don't use MongoDB",
            negated_term="mongodb",
            span=(0, 19),
        )
        assert match.statement == "I don't use MongoDB"
        assert match.negated_term == "mongodb"
        assert match.span == (0, 19)


class TestCreateNegationFacts:
    """Tests for create_negation_facts function."""

    def test_creates_negation_facts(self):
        """Should create NegationFact objects from episode."""
        episode = make_episode("I don't use MongoDB")
        facts = create_negation_facts(episode)

        assert len(facts) == 1
        assert isinstance(facts[0], NegationFact)
        assert facts[0].content == "I don't use MongoDB"
        assert facts[0].negates_pattern == "mongodb"
        assert facts[0].user_id == "user_123"

    def test_links_source_episode(self):
        """Should link NegationFact to source episode."""
        episode = make_episode("I don't use MongoDB")
        facts = create_negation_facts(episode)

        assert len(facts) == 1
        assert episode.id in facts[0].source_episode_ids

    def test_sets_extracted_confidence(self):
        """Should set EXTRACTED confidence for pattern-matched negations."""
        episode = make_episode("I don't use MongoDB")
        facts = create_negation_facts(episode)

        assert len(facts) == 1
        assert facts[0].confidence.extraction_method == ExtractionMethod.EXTRACTED
        assert facts[0].confidence.extraction_base == 0.9

    def test_inherits_org_id(self):
        """Should inherit org_id from episode."""
        episode = Episode(
            content="I don't use MongoDB",
            role="user",
            user_id="user_123",
            org_id="org_456",
        )
        facts = create_negation_facts(episode)

        assert len(facts) == 1
        assert facts[0].org_id == "org_456"

    def test_uses_provided_matches(self):
        """Should use pre-computed matches if provided."""
        episode = make_episode("Some text")
        pre_matches = [
            NegationMatch(
                statement="I never use Java",
                negated_term="java",
                span=(0, 16),
            )
        ]
        facts = create_negation_facts(episode, matches=pre_matches)

        assert len(facts) == 1
        assert facts[0].content == "I never use Java"
        assert facts[0].negates_pattern == "java"

    def test_empty_when_no_negations(self):
        """Should return empty list when no negations found."""
        episode = make_episode("I love Python")
        facts = create_negation_facts(episode)
        assert facts == []

    def test_multiple_negation_facts(self):
        """Should create multiple facts for multiple negations."""
        episode = make_episode("I don't use MongoDB and I never used Oracle")
        facts = create_negation_facts(episode)

        assert len(facts) == 2
        patterns = {f.negates_pattern for f in facts}
        assert patterns == {"mongodb", "oracle"}


class TestNegationDetector:
    """Tests for NegationDetector class."""

    def test_detector_name(self):
        """Should have correct name."""
        detector = NegationDetector()
        assert detector.name == "negation"

    def test_detect_method(self):
        """Should detect negations via detect method."""
        detector = NegationDetector()
        episode = make_episode("I don't use MongoDB")
        facts = detector.detect(episode)

        assert len(facts) == 1
        assert isinstance(facts[0], NegationFact)
        assert facts[0].negates_pattern == "mongodb"

    def test_detect_multiple(self):
        """Should detect multiple negations."""
        detector = NegationDetector()
        episode = make_episode("I don't like Java and I never use Ruby")
        facts = detector.detect(episode)

        # Both patterns match: "I don't like Java" and "I never use Ruby"
        assert len(facts) == 2
        patterns = {f.negates_pattern for f in facts}
        assert patterns == {"java", "ruby"}

    def test_detect_empty(self):
        """Should return empty list for positive text."""
        detector = NegationDetector()
        episode = make_episode("I love programming in Python")
        facts = detector.detect(episode)

        assert facts == []


class TestShortTermNegations:
    """Tests for short programming language terms (R, C, Go, etc.)."""

    def test_r_language_negation(self):
        """Should detect 'I don't use R' for R language."""
        matches = detect_negations("I don't use R anymore")
        assert len(matches) == 1
        assert matches[0].negated_term == "r"

    def test_c_language_negation(self):
        """Should detect 'I don't use C' for C language."""
        matches = detect_negations("I don't use C for this")
        assert len(matches) == 1
        assert matches[0].negated_term == "c"

    def test_go_language_negation(self):
        """Should detect 'I don't use Go' for Go language."""
        matches = detect_negations("I don't use Go")
        assert len(matches) == 1
        assert matches[0].negated_term == "go"
