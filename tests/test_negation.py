"""Unit tests for Engram negation detection."""

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
    """Tests for detect_negations function."""

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

    def test_no_longer_use_pattern(self):
        """Should detect 'I no longer use X' patterns."""
        matches = detect_negations("I no longer use Vim")
        assert len(matches) == 1
        assert matches[0].negated_term == "vim"

    def test_actually_not_pattern(self):
        """Should detect 'Actually, not X' patterns."""
        matches = detect_negations("Actually, not Python but JavaScript")
        assert len(matches) == 1
        assert matches[0].negated_term == "python"

    def test_no_not_pattern(self):
        """Should detect 'No, not X' patterns."""
        matches = detect_negations("No, not TypeScript")
        assert len(matches) == 1
        assert matches[0].negated_term == "typescript"

    def test_is_wrong_pattern(self):
        """Should detect 'X is wrong' patterns."""
        matches = detect_negations("MongoDB is wrong for this use case")
        assert len(matches) == 1
        assert matches[0].negated_term == "mongodb"

    def test_is_incorrect_pattern(self):
        """Should detect 'X is incorrect' patterns."""
        matches = detect_negations("That assumption is incorrect")
        assert len(matches) == 1
        assert matches[0].negated_term == "assumption"

    def test_my_x_is_not_y_pattern(self):
        """Should detect 'my X is not Y' patterns."""
        matches = detect_negations("my email is not jane@example.com")
        assert len(matches) == 1
        assert matches[0].negated_term == "jane@example.com"

    def test_thats_not_my_pattern(self):
        """Should detect 'that's not my X' patterns."""
        matches = detect_negations("That's not my preference")
        assert len(matches) == 1
        assert matches[0].negated_term == "preference"

    def test_stopped_using_pattern(self):
        """Should detect 'I stopped using X' patterns."""
        matches = detect_negations("I stopped using React years ago")
        assert len(matches) == 1
        assert matches[0].negated_term == "react"

    def test_dont_x_anymore_pattern(self):
        """Should detect 'I don't X anymore' patterns."""
        matches = detect_negations("I don't code anymore")
        assert len(matches) == 1
        assert matches[0].negated_term == "code"

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
        # This text could potentially match multiple patterns at same location
        text = "Actually, not Python - I don't use Python"
        matches = detect_negations(text)
        # Check no overlapping spans
        spans = [m.span for m in matches]
        for i, span1 in enumerate(spans):
            for span2 in spans[i + 1 :]:
                assert span1[1] <= span2[0] or span2[1] <= span1[0], "Spans overlap"


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
        episode = make_episode("I don't like Java and I'm not a fan of Ruby")
        facts = detector.detect(episode)

        assert len(facts) == 2

    def test_detect_empty(self):
        """Should return empty list for positive text."""
        detector = NegationDetector()
        episode = make_episode("I love programming in Python")
        facts = detector.detect(episode)

        assert facts == []


class TestNegationPatternCoverage:
    """Tests verifying pattern coverage for various negation forms."""

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

    def test_dont_have_pattern(self):
        """Should detect 'don't have' negations."""
        matches = detect_negations("I don't have React experience")
        assert len(matches) == 1
        assert matches[0].negated_term == "react"

    def test_dont_prefer_pattern(self):
        """Should detect 'don't prefer' negations."""
        matches = detect_negations("I don't prefer MySQL")
        assert len(matches) == 1
        assert matches[0].negated_term == "mysql"

    def test_never_liked_pattern(self):
        """Should detect 'never liked' negations."""
        matches = detect_negations("I never liked Angular")
        assert len(matches) == 1
        assert matches[0].negated_term == "angular"

    def test_never_wanted_pattern(self):
        """Should detect 'never wanted' negations."""
        matches = detect_negations("I never wanted Perl")
        assert len(matches) == 1
        assert matches[0].negated_term == "perl"

    def test_never_had_pattern(self):
        """Should detect 'never had' negations."""
        matches = detect_negations("I never had AWS experience")
        assert len(matches) == 1
        assert matches[0].negated_term == "aws"

    def test_no_longer_like_pattern(self):
        """Should detect 'no longer like' negations."""
        matches = detect_negations("I no longer like PHP")
        assert len(matches) == 1
        assert matches[0].negated_term == "php"

    def test_no_longer_need_pattern(self):
        """Should detect 'no longer need' negations."""
        matches = detect_negations("I no longer need jQuery")
        assert len(matches) == 1
        assert matches[0].negated_term == "jquery"

    def test_is_not_right_pattern(self):
        """Should detect 'is not right' negations."""
        matches = detect_negations("That approach is not right")
        assert len(matches) == 1
        assert matches[0].negated_term == "approach"

    def test_is_not_correct_pattern(self):
        """Should detect 'is not correct' negations."""
        matches = detect_negations("That answer is not correct")
        assert len(matches) == 1
        assert matches[0].negated_term == "answer"

    def test_is_not_true_pattern(self):
        """Should detect 'is not true' negations."""
        matches = detect_negations("That statement is not true")
        assert len(matches) == 1
        assert matches[0].negated_term == "statement"

    def test_is_false_pattern(self):
        """Should detect 'is false' negations."""
        matches = detect_negations("That claim is false")
        assert len(matches) == 1
        assert matches[0].negated_term == "claim"

    def test_that_is_not_my_pattern(self):
        """Should detect 'that is not my X' patterns."""
        matches = detect_negations("That is not my style")
        assert len(matches) == 1
        assert matches[0].negated_term == "style"

    def test_nope_not_pattern(self):
        """Should detect 'nope, not X' patterns."""
        matches = detect_negations("Nope, not JavaScript")
        assert len(matches) == 1
        assert matches[0].negated_term == "javascript"

    def test_stopped_liking_pattern(self):
        """Should detect 'stopped liking' negations."""
        matches = detect_negations("I stopped liking Ruby")
        assert len(matches) == 1
        assert matches[0].negated_term == "ruby"

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
