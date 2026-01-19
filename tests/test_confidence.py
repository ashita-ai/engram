"""Tests for intelligent confidence scoring (#136).

Architecture:
- Episodic: 1.0 always (verbatim)
- Structured: LLM assesses confidence during extraction
- Semantic: LLM assesses confidence during synthesis
- Procedural: Bayesian updating with accumulating evidence
"""

import pytest

from engram.confidence import (
    BayesianConfidence,
    bayesian_update,
    combine_bayesian_confidences,
)
from engram.confidence.llm_certainty import (
    CertaintyAssessment,
    LLMCertaintyAssessor,
    SynthesisCertaintyAssessment,
)
from engram.models.base import ConfidenceScore, ExtractionMethod


class TestConfidenceScore:
    """Tests for the ConfidenceScore model."""

    def test_verbatim_always_1(self) -> None:
        """Verbatim (episodic) memories should always have confidence 1.0."""
        score = ConfidenceScore.for_verbatim()
        assert score.value == 1.0
        assert score.extraction_method == ExtractionMethod.VERBATIM

    def test_extracted_default_0_9(self) -> None:
        """Pattern-extracted content defaults to 0.9."""
        score = ConfidenceScore.for_extracted()
        assert score.value == 0.9
        assert score.extraction_method == ExtractionMethod.EXTRACTED

    def test_inferred_with_llm_confidence(self) -> None:
        """LLM-inferred content should use LLM-assessed confidence."""
        score = ConfidenceScore.for_inferred(
            confidence=0.85,
            reasoning="Clearly stated preference",
        )
        assert score.value == 0.85
        assert score.extraction_base == 0.85
        assert score.llm_reasoning == "Clearly stated preference"

    def test_inferred_default_confidence(self) -> None:
        """Default inferred confidence is 0.6."""
        score = ConfidenceScore.for_inferred()
        assert score.value == 0.6
        assert score.llm_reasoning is None

    def test_explain_includes_llm_reasoning(self) -> None:
        """Explain should include LLM reasoning when present."""
        score = ConfidenceScore.for_inferred(
            confidence=0.8,
            reasoning="User explicitly stated this preference",
        )
        explanation = score.explain()
        assert "LLM:" in explanation

    def test_recompute_with_corroboration(self) -> None:
        """More corroborating sources should increase confidence."""
        score1 = ConfidenceScore.for_inferred(confidence=0.7, supporting_episodes=1)
        score2 = ConfidenceScore.for_inferred(confidence=0.7, supporting_episodes=5)

        score1.recompute()
        score2.recompute()

        assert score2.value > score1.value

    def test_recompute_with_contradictions(self) -> None:
        """Contradictions should reduce confidence."""
        score = ConfidenceScore.for_inferred(confidence=0.8)
        score.contradictions = 2
        score.recompute()

        assert score.value < 0.8


class TestBayesianConfidence:
    """Tests for Bayesian confidence updating (Procedural memories)."""

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
    """Tests for LLM certainty assessment models."""

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
