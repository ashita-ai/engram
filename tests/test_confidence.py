"""Tests for intelligent confidence scoring (#136).

Tests:
- Phase 1: Hedging detection, specificity scoring, signal integration
- Phase 2: LLM certainty assessment (mocked)
- Phase 3: Bayesian confidence updating
"""

import pytest

from engram.confidence import (
    BayesianConfidence,
    HedgingDetector,
    SpecificityScorer,
    bayesian_update,
    calculate_specificity,
    combine_bayesian_confidences,
    compute_confidence_signals,
    detect_hedging,
)
from engram.confidence.llm_certainty import (
    CertaintyAssessment,
    LLMCertaintyAssessor,
    SynthesisCertaintyAssessment,
)
from engram.models.base import ConfidenceScore, ExtractionMethod


class TestHedgingDetection:
    """Tests for hedging language detection."""

    def test_no_hedging_returns_no_penalty(self) -> None:
        """Text without hedging should have penalty of 1.0."""
        result = detect_hedging("I prefer Python for web development")
        assert result.has_hedging is False
        assert result.penalty == 1.0
        assert result.severity == "none"

    def test_mild_hedging_detected(self) -> None:
        """Mild hedging words should result in 0.9 penalty."""
        result = detect_hedging("I generally prefer Python for web development")
        assert result.has_hedging is True
        assert result.penalty == 0.9
        assert result.severity == "mild"
        assert "generally" in result.matched_patterns

    def test_moderate_hedging_detected(self) -> None:
        """Moderate hedging words should result in 0.8 penalty."""
        result = detect_hedging("I think Python is probably the best choice")
        assert result.has_hedging is True
        assert result.penalty == 0.8
        assert result.severity == "moderate"
        assert any(p in result.matched_patterns for p in ["I think", "probably"])

    def test_strong_hedging_detected(self) -> None:
        """Strong hedging words should result in 0.7 penalty."""
        result = detect_hedging("I'm not sure, but I could be wrong about this")
        assert result.has_hedging is True
        assert result.penalty == 0.7
        assert result.severity == "strong"
        assert any(p in result.matched_patterns for p in ["not sure", "could be wrong"])

    def test_multiple_hedging_patterns(self) -> None:
        """Multiple hedging patterns should all be captured."""
        result = detect_hedging("I think maybe I might prefer Python")
        assert result.has_hedging is True
        assert len(result.matched_patterns) >= 2
        # Penalty should be the lowest (most severe)
        assert result.penalty == 0.8

    def test_case_insensitive(self) -> None:
        """Hedging detection should be case-insensitive."""
        result = detect_hedging("I THINK Maybe PROBABLY this is correct")
        assert result.has_hedging is True
        assert result.penalty == 0.8

    def test_empty_text(self) -> None:
        """Empty text should return no hedging."""
        result = detect_hedging("")
        assert result.has_hedging is False
        assert result.penalty == 1.0

    def test_word_boundaries(self) -> None:
        """Patterns should respect word boundaries."""
        # "something" should not match "some"
        result = detect_hedging("Sometimes I use Python")
        # "Sometimes" shouldn't match hedging patterns (it's different from "some")
        # It might match "generally" patterns if we're not careful
        assert "something" not in result.matched_patterns

    def test_custom_detector_instance(self) -> None:
        """HedgingDetector can be instantiated and reused."""
        detector = HedgingDetector()
        result1 = detector.detect("I think this is good")
        result2 = detector.detect("This is definitely good")
        assert result1.has_hedging is True
        assert result2.has_hedging is False


class TestSpecificityScoring:
    """Tests for specificity scoring."""

    def test_vague_text_low_score(self) -> None:
        """Vague text should have low specificity score."""
        result = calculate_specificity("I like programming and stuff")
        assert result.score < 0.5
        assert result.vague_count > 0
        assert "vague_words" in result.details

    def test_specific_text_high_score(self) -> None:
        """Specific text with details should have high score."""
        result = calculate_specificity("I use Python 3.12 with FastAPI to build REST APIs on AWS")
        assert result.score > 0.5
        assert result.specific_count > 0
        assert "tech_terms" in result.details

    def test_numbers_boost_specificity(self) -> None:
        """Numbers in text should boost specificity."""
        result = calculate_specificity("The server has 16GB RAM and 8 CPU cores")
        assert result.score > 0.5
        assert "numbers" in result.details

    def test_version_numbers_boost_specificity(self) -> None:
        """Version numbers should boost specificity."""
        result = calculate_specificity("Running Python 3.12.1 on Ubuntu 22.04")
        assert result.score > 0.5
        assert "versions" in result.details

    def test_dates_boost_specificity(self) -> None:
        """Dates should boost specificity."""
        result = calculate_specificity("The deadline is January 15, 2026")
        assert result.score >= 0.5
        # Dates might be captured as either "dates" or "numbers" depending on format
        assert result.specific_count > 0

    def test_emails_boost_specificity(self) -> None:
        """Email addresses should boost specificity."""
        result = calculate_specificity("Contact me at user@example.com")
        assert result.score > 0.5
        assert "emails" in result.details

    def test_urls_boost_specificity(self) -> None:
        """URLs should boost specificity."""
        result = calculate_specificity("Check https://docs.example.com/api for details")
        assert result.score > 0.5
        assert "urls" in result.details

    def test_tech_terms_boost_specificity(self) -> None:
        """Technical terms should boost specificity."""
        result = calculate_specificity("Using PostgreSQL with Redis for caching")
        assert result.score > 0.5
        assert "tech_terms" in result.details
        assert result.details["tech_terms"] >= 2

    def test_filler_phrases_reduce_specificity(self) -> None:
        """Filler phrases should reduce specificity."""
        result = calculate_specificity("Basically, it's like, you know, just a thing")
        assert result.score < 0.5
        assert "fillers" in result.details or result.vague_count > 0

    def test_empty_text_returns_baseline(self) -> None:
        """Empty text should return baseline score."""
        result = calculate_specificity("")
        assert result.score == 0.5
        assert result.word_count == 0

    def test_custom_scorer_instance(self) -> None:
        """SpecificityScorer can be instantiated and reused."""
        scorer = SpecificityScorer()
        result1 = scorer.score("I use Python 3.12")
        result2 = scorer.score("I like things")
        assert result1.score > result2.score


class TestConfidenceSignals:
    """Tests for unified confidence signals."""

    def test_signals_combine_hedging_and_specificity(self) -> None:
        """Signals should combine both hedging and specificity."""
        signals = compute_confidence_signals("I think I use Python 3.12 with FastAPI")
        assert signals.hedging_penalty < 1.0  # Has hedging
        assert signals.specificity_boost > 0  # Has tech terms

    def test_no_adjustments_for_neutral_text(self) -> None:
        """Neutral text should have minimal adjustments."""
        signals = compute_confidence_signals("The project uses a database")
        # No hedging, moderate specificity
        assert signals.hedging_penalty == 1.0

    def test_apply_to_confidence(self) -> None:
        """Signals should correctly adjust base confidence."""
        signals = compute_confidence_signals("I think maybe this is correct")
        base = 0.9
        adjusted = signals.apply_to_confidence(base)
        assert adjusted < base  # Hedging should reduce confidence
        assert adjusted > 0.0  # But not below zero

    def test_explanation_generated(self) -> None:
        """Signals should include human-readable explanation."""
        signals = compute_confidence_signals("I think I prefer Python 3.12")
        assert signals.explanation != ""
        # Should mention hedging since "I think" is present
        assert "hedging" in signals.explanation.lower() or signals.hedging_penalty < 1.0


class TestConfidenceScoreWithSignals:
    """Tests for ConfidenceScore integration with intelligent signals."""

    def test_new_fields_have_defaults(self) -> None:
        """New signal fields should have sensible defaults."""
        score = ConfidenceScore.for_inferred()
        assert score.hedging_penalty == 1.0
        assert score.specificity_boost == 0.0
        assert score.source_text_analyzed is False

    def test_analyze_source_text_updates_signals(self) -> None:
        """analyze_source_text should update hedging and specificity."""
        score = ConfidenceScore.for_inferred()
        score.analyze_source_text("I think I might prefer Python 3.12")
        assert score.hedging_penalty < 1.0
        assert score.source_text_analyzed is True

    def test_analyze_source_text_recomputes(self) -> None:
        """analyze_source_text should recompute the confidence value."""
        score = ConfidenceScore.for_inferred()
        original_value = score.value
        score.analyze_source_text("I'm not sure but maybe it's correct")
        # Hedging should reduce confidence
        assert score.value <= original_value

    def test_recompute_applies_hedging_penalty(self) -> None:
        """recompute should apply hedging penalty."""
        score = ConfidenceScore(
            value=0.8,
            extraction_method=ExtractionMethod.INFERRED,
            extraction_base=0.6,
            hedging_penalty=0.8,  # 20% penalty
        )
        score.recompute()
        # Base score is roughly 0.6 * 0.5 + 0.5 * 0.25 + ~0.15 = 0.575
        # With 0.8 hedging penalty: 0.575 * 0.8 = 0.46
        assert score.value < 0.6

    def test_recompute_applies_specificity_boost(self) -> None:
        """recompute should apply specificity boost."""
        score = ConfidenceScore(
            value=0.8,
            extraction_method=ExtractionMethod.INFERRED,
            extraction_base=0.6,
            specificity_boost=0.1,  # 10% boost
        )
        score.recompute()
        # Base score + 0.1 boost
        assert score.value > 0.5

    def test_explain_includes_signals_when_analyzed(self) -> None:
        """explain() should include signal info when source was analyzed."""
        score = ConfidenceScore.for_inferred()
        score.analyze_source_text("I think this might be correct")
        explanation = score.explain()
        # Should include hedging info
        assert "hedging" in explanation.lower() or score.hedging_penalty == 1.0

    def test_explain_omits_signals_when_not_analyzed(self) -> None:
        """explain() should not include signal info when source wasn't analyzed."""
        score = ConfidenceScore.for_inferred()
        explanation = score.explain()
        # Should not include hedging/specific info
        assert "hedging" not in explanation.lower()
        assert "specific" not in explanation.lower()

    def test_confident_specific_text_high_score(self) -> None:
        """Confident, specific text should have high final score."""
        score = ConfidenceScore.for_extracted()  # Base 0.9
        score.analyze_source_text("Use Python 3.12 with FastAPI on AWS us-east-1")
        # No hedging, high specificity
        assert score.value >= 0.8

    def test_uncertain_vague_text_low_score(self) -> None:
        """Uncertain, vague text should have lower final score."""
        score = ConfidenceScore.for_inferred()  # Base 0.6
        score.analyze_source_text("I think maybe I like something or other")
        # Has hedging, low specificity
        assert score.value < 0.5


class TestEdgeCases:
    """Edge case tests."""

    def test_none_like_text_handled(self) -> None:
        """Whitespace-only text should not crash."""
        result = detect_hedging("   ")
        assert result.has_hedging is False

        result = calculate_specificity("   ")
        assert result.word_count == 0

    def test_very_long_text_handled(self) -> None:
        """Long text should be handled efficiently."""
        long_text = "I think " * 1000 + "Python 3.12 " * 500
        result = detect_hedging(long_text)
        assert result.has_hedging is True

        result = calculate_specificity(long_text)
        assert result.word_count > 0

    def test_special_characters_handled(self) -> None:
        """Special characters should not crash analysis."""
        text = "I use Python™ 3.12 (beta) for $100/month @ work!"
        result = detect_hedging(text)
        assert result.penalty == 1.0  # No hedging

        result = calculate_specificity(text)
        assert result.score > 0  # Should still score

    def test_mixed_case_tech_terms(self) -> None:
        """Tech terms should be case-insensitive."""
        result = calculate_specificity("Using POSTGRESQL and REDIS on AWS")
        assert "tech_terms" in result.details
        assert result.details["tech_terms"] >= 2


class TestBayesianConfidence:
    """Tests for Bayesian confidence updating (Phase 3)."""

    def test_initial_confidence_from_prior(self) -> None:
        """Initial confidence should reflect the prior."""
        bc = BayesianConfidence.from_prior("weak")
        assert bc.confidence == 0.5  # Uniform weak prior

        bc_opt = BayesianConfidence.from_prior("optimistic")
        assert bc_opt.confidence == 0.75  # 3/(3+1)

        bc_pess = BayesianConfidence.from_prior("pessimistic")
        assert bc_pess.confidence == 0.25  # 1/(1+3)

    def test_update_increases_confidence_on_confirmation(self) -> None:
        """Confirming observations should increase confidence."""
        bc = BayesianConfidence.from_prior("weak")
        initial = bc.confidence

        bc.update(observed=True)
        assert bc.confidence > initial
        assert bc.confirmations == 1
        assert bc.observations == 1

    def test_update_decreases_confidence_on_contradiction(self) -> None:
        """Contradicting observations should decrease confidence."""
        bc = BayesianConfidence.from_prior("weak")
        initial = bc.confidence

        bc.update(observed=False)
        assert bc.confidence < initial
        assert bc.contradictions == 1

    def test_multiple_updates_converge(self) -> None:
        """Many confirmations should converge toward 1.0."""
        bc = BayesianConfidence.from_prior("weak")

        for _ in range(20):
            bc.update(observed=True)

        assert bc.confidence > 0.9
        assert bc.strength == "strong_positive"

    def test_mixed_evidence(self) -> None:
        """Mixed evidence should result in moderate confidence."""
        bc = BayesianConfidence.from_observations(
            confirmations=6,
            contradictions=4,
            prior="weak",
        )
        # (2+6) / (2+6+2+4) = 8/14 ≈ 0.57
        assert 0.5 < bc.confidence < 0.7

    def test_batch_update(self) -> None:
        """Batch updates should work correctly."""
        bc = BayesianConfidence.from_prior("weak")
        bc.update_batch(confirmations=5, contradictions=2)

        assert bc.observations == 7
        assert bc.confirmations == 5
        assert bc.contradictions == 2

    def test_decay_reduces_certainty(self) -> None:
        """Decay should make confidence more uncertain."""
        bc = BayesianConfidence.from_observations(10, 2, prior="weak")
        initial_variance = bc.variance

        bc.decay(factor=0.5)

        # Variance should increase (more uncertain)
        assert bc.variance >= initial_variance

    def test_credible_interval(self) -> None:
        """Credible interval should be reasonable."""
        bc = BayesianConfidence.from_observations(10, 5, prior="weak")
        low, high = bc.credible_interval_95

        assert low < bc.confidence < high
        assert 0.0 <= low
        assert high <= 1.0

    def test_explain_output(self) -> None:
        """Explain should produce readable output."""
        bc = BayesianConfidence.from_observations(8, 2)
        explanation = bc.explain()

        assert "8 confirmations" in explanation
        assert "2 contradictions" in explanation

    def test_from_initial_confidence(self) -> None:
        """Can create prior from desired initial confidence."""
        bc = BayesianConfidence.from_prior(initial_confidence=0.7)
        assert abs(bc.confidence - 0.7) < 0.01

    def test_strength_categories(self) -> None:
        """Strength should categorize correctly."""
        # Insufficient data
        bc = BayesianConfidence.from_prior("weak")
        assert bc.strength == "insufficient"

        # Strong positive
        bc = BayesianConfidence.from_observations(15, 1)
        assert bc.strength == "strong_positive"

        # Strong negative
        bc = BayesianConfidence.from_observations(1, 15)
        assert bc.strength == "strong_negative"


class TestBayesianUpdateFunction:
    """Tests for the convenience bayesian_update function."""

    def test_basic_update(self) -> None:
        """Basic Bayesian update should work."""
        result = bayesian_update(
            prior_confidence=0.5,
            observations=[True, True, True],
        )
        assert result > 0.5

    def test_conflicting_evidence(self) -> None:
        """Conflicting evidence should moderate confidence."""
        result = bayesian_update(
            prior_confidence=0.8,
            observations=[False, False, False],
        )
        assert result < 0.8


class TestCombineBayesianConfidences:
    """Tests for combining multiple Bayesian confidences."""

    def test_combine_empty_list(self) -> None:
        """Empty list should return default prior."""
        result = combine_bayesian_confidences([])
        assert result.confidence == 0.5

    def test_combine_multiple(self) -> None:
        """Combining should aggregate evidence."""
        bc1 = BayesianConfidence.from_observations(5, 1)
        bc2 = BayesianConfidence.from_observations(3, 2)

        combined = combine_bayesian_confidences([bc1, bc2])

        assert combined.observations == 11  # 5+1+3+2
        assert combined.confirmations == 8  # 5+3
        assert combined.contradictions == 3  # 1+2


class TestLLMCertaintyModels:
    """Tests for LLM certainty assessment models (mocked)."""

    def test_certainty_assessment_validation(self) -> None:
        """CertaintyAssessment should validate bounds."""
        # Valid assessment
        assessment = CertaintyAssessment(
            certainty=0.85,
            reasoning="Clearly stated preference",
            hedging_detected=[],
            clarity="clear",
        )
        assert assessment.certainty == 0.85

        # Invalid certainty should fail
        with pytest.raises(ValueError):
            CertaintyAssessment(
                certainty=1.5,  # Out of bounds
                reasoning="Invalid",
            )

    def test_synthesis_certainty_assessment_validation(self) -> None:
        """SynthesisCertaintyAssessment should validate bounds."""
        assessment = SynthesisCertaintyAssessment(
            certainty=0.7,
            reasoning="Good source agreement",
            source_agreement="strong",
            inference_strength="direct",
        )
        assert assessment.certainty == 0.7
        assert assessment.source_agreement == "strong"

    @pytest.mark.asyncio
    async def test_llm_assessor_disabled_mode(self) -> None:
        """LLMCertaintyAssessor with enabled=False should return fallbacks."""
        assessor = LLMCertaintyAssessor(enabled=False)

        result = await assessor.assess_extraction(
            extracted="User prefers Python",
            source_text="I love Python",
        )
        assert result.certainty == 0.7
        assert "disabled" in result.reasoning.lower()

        result2 = await assessor.assess_synthesis(
            synthesized="User is a developer",
            source_memories=["Uses Python", "Writes code"],
        )
        assert result2.certainty == 0.5

    @pytest.mark.asyncio
    async def test_llm_assessor_call_counting(self) -> None:
        """LLMCertaintyAssessor should count calls even when disabled."""
        assessor = LLMCertaintyAssessor(enabled=False)

        await assessor.assess_extraction("test", "test")
        await assessor.assess_extraction("test2", "test2")
        await assessor.assess_synthesis("test", ["a", "b"])

        assert assessor.extraction_calls == 2
        assert assessor.synthesis_calls == 1
