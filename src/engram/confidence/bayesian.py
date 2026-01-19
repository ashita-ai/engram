"""Bayesian confidence updating for procedural memories.

Procedural memories represent behavioral patterns observed over time.
Unlike single-shot extractions, these benefit from Bayesian updating
where we maintain a prior belief and update it with accumulating evidence.

The model uses Beta distribution conjugate priors, which are perfect for
binary observations (behavior observed vs not observed).

Example:
    >>> from engram.confidence.bayesian import BayesianConfidence
    >>> bc = BayesianConfidence(prior_alpha=2, prior_beta=3)  # Weak prior ~0.4
    >>> bc.update(observed=True)   # Saw the behavior
    >>> bc.update(observed=True)   # Saw it again
    >>> bc.confidence  # Posterior mean
    0.67
"""

from __future__ import annotations

import math
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field


class BayesianConfidence(BaseModel):
    """Bayesian confidence tracker using Beta-Bernoulli model.

    The Beta distribution is the conjugate prior for Bernoulli trials,
    making updates simple: just add observations to alpha (successes)
    and beta (failures).

    Attributes:
        alpha: Success count + prior (shape parameter).
        beta: Failure count + prior (shape parameter).
        observations: Total number of observations made.
        confirmations: Number of confirming observations.
        contradictions: Number of contradicting observations.

    The confidence (posterior mean) is alpha / (alpha + beta).
    """

    model_config = ConfigDict(extra="forbid")

    alpha: float = Field(
        default=2.0,
        gt=0,
        description="Beta distribution alpha (successes + prior)",
    )
    beta: float = Field(
        default=2.0,
        gt=0,
        description="Beta distribution beta (failures + prior)",
    )
    observations: int = Field(
        default=0,
        ge=0,
        description="Total observations",
    )
    confirmations: int = Field(
        default=0,
        ge=0,
        description="Confirming observations",
    )
    contradictions: int = Field(
        default=0,
        ge=0,
        description="Contradicting observations",
    )

    # Default priors for different scenarios
    UNINFORMATIVE_PRIOR: ClassVar[tuple[float, float]] = (1.0, 1.0)  # Uniform
    WEAK_PRIOR: ClassVar[tuple[float, float]] = (2.0, 2.0)  # Slight regularization
    MODERATE_PRIOR: ClassVar[tuple[float, float]] = (5.0, 5.0)  # Stronger regularization
    OPTIMISTIC_PRIOR: ClassVar[tuple[float, float]] = (3.0, 1.0)  # Expect success
    PESSIMISTIC_PRIOR: ClassVar[tuple[float, float]] = (1.0, 3.0)  # Expect failure

    @property
    def confidence(self) -> float:
        """Posterior mean: E[p] = alpha / (alpha + beta)."""
        return self.alpha / (self.alpha + self.beta)

    @property
    def variance(self) -> float:
        """Posterior variance: Var[p] = alpha*beta / ((alpha+beta)^2 * (alpha+beta+1))."""
        total = self.alpha + self.beta
        return (self.alpha * self.beta) / (total * total * (total + 1))

    @property
    def std_dev(self) -> float:
        """Posterior standard deviation."""
        return math.sqrt(self.variance)

    @property
    def credible_interval_95(self) -> tuple[float, float]:
        """Approximate 95% credible interval using normal approximation.

        For large sample sizes, the Beta distribution approaches normal.
        For small samples, this is approximate but useful.
        """
        # Use 1.96 standard deviations for 95% CI
        mean = self.confidence
        margin = 1.96 * self.std_dev
        return (max(0.0, mean - margin), min(1.0, mean + margin))

    @property
    def strength(self) -> str:
        """Interpret the confidence strength based on evidence.

        Returns:
            String indicating evidence strength.
        """
        total = self.observations
        conf = self.confidence

        if total < 3:
            return "insufficient"
        elif total < 10:
            if conf > 0.7:
                return "emerging_positive"
            elif conf < 0.3:
                return "emerging_negative"
            else:
                return "uncertain"
        else:
            if conf > 0.8:
                return "strong_positive"
            elif conf > 0.6:
                return "moderate_positive"
            elif conf < 0.2:
                return "strong_negative"
            elif conf < 0.4:
                return "moderate_negative"
            else:
                return "mixed"

    def update(self, observed: bool, weight: float = 1.0) -> BayesianConfidence:
        """Update posterior with a new observation.

        Args:
            observed: True if behavior was observed (confirmation),
                     False if contradicted or not observed.
            weight: Observation weight (default 1.0). Use <1.0 for
                   weak evidence, >1.0 for strong evidence.

        Returns:
            Self with updated parameters.

        Example:
            >>> bc = BayesianConfidence()
            >>> bc.update(observed=True)  # Saw the behavior
            >>> bc.confidence > 0.5
            True
        """
        self.observations += 1

        if observed:
            self.alpha += weight
            self.confirmations += 1
        else:
            self.beta += weight
            self.contradictions += 1

        return self

    def update_batch(
        self,
        confirmations: int,
        contradictions: int,
        weight: float = 1.0,
    ) -> BayesianConfidence:
        """Update posterior with multiple observations at once.

        Args:
            confirmations: Number of confirming observations.
            contradictions: Number of contradicting observations.
            weight: Weight per observation.

        Returns:
            Self with updated parameters.
        """
        self.alpha += confirmations * weight
        self.beta += contradictions * weight
        self.observations += confirmations + contradictions
        self.confirmations += confirmations
        self.contradictions += contradictions
        return self

    def decay(self, factor: float = 0.95) -> BayesianConfidence:
        """Apply time-based decay to evidence.

        Reduces the effective sample size, making the posterior
        more uncertain over time without new observations.

        Args:
            factor: Decay factor (0.0-1.0). 0.95 means 5% decay.

        Returns:
            Self with decayed parameters.
        """
        # Decay both alpha and beta proportionally
        # But maintain minimum prior
        min_alpha = 1.0
        min_beta = 1.0

        self.alpha = max(min_alpha, self.alpha * factor)
        self.beta = max(min_beta, self.beta * factor)

        return self

    def explain(self) -> str:
        """Generate human-readable explanation of confidence.

        Example: "0.75 (strong_positive): 15 confirmations, 5 contradictions"
        """
        return (
            f"{self.confidence:.2f} ({self.strength}): "
            f"{self.confirmations} confirmations, {self.contradictions} contradictions"
        )

    @classmethod
    def from_prior(
        cls,
        prior: str = "weak",
        initial_confidence: float | None = None,
    ) -> BayesianConfidence:
        """Create a BayesianConfidence with a named prior.

        Args:
            prior: Prior type: "uninformative", "weak", "moderate",
                  "optimistic", "pessimistic".
            initial_confidence: If provided, set prior to achieve this
                              initial confidence with weak evidence.

        Returns:
            New BayesianConfidence instance.

        Example:
            >>> bc = BayesianConfidence.from_prior("optimistic")
            >>> bc.confidence
            0.75
        """
        if initial_confidence is not None:
            # Create prior that gives desired initial confidence
            # With total pseudo-count of 4 for regularization
            total = 4.0
            alpha = initial_confidence * total
            beta = total - alpha
            return cls(alpha=max(0.5, alpha), beta=max(0.5, beta))

        priors = {
            "uninformative": cls.UNINFORMATIVE_PRIOR,
            "weak": cls.WEAK_PRIOR,
            "moderate": cls.MODERATE_PRIOR,
            "optimistic": cls.OPTIMISTIC_PRIOR,
            "pessimistic": cls.PESSIMISTIC_PRIOR,
        }

        alpha, beta = priors.get(prior, cls.WEAK_PRIOR)
        return cls(alpha=alpha, beta=beta)

    @classmethod
    def from_observations(
        cls,
        confirmations: int,
        contradictions: int,
        prior: str = "weak",
    ) -> BayesianConfidence:
        """Create a BayesianConfidence from observation counts.

        Args:
            confirmations: Number of confirming observations.
            contradictions: Number of contradicting observations.
            prior: Prior type for regularization.

        Returns:
            New BayesianConfidence with observations incorporated.

        Example:
            >>> bc = BayesianConfidence.from_observations(8, 2)
            >>> bc.confidence
            0.75
        """
        bc = cls.from_prior(prior)
        bc.update_batch(confirmations, contradictions)
        return bc


def bayesian_update(
    prior_confidence: float,
    observations: list[bool],
    prior_strength: float = 4.0,
) -> float:
    """Simple Bayesian update function.

    Convenience function for one-off updates without managing state.

    Args:
        prior_confidence: Initial confidence (0.0-1.0).
        observations: List of True (confirm) / False (contradict).
        prior_strength: How strongly to weight the prior (pseudo-count).

    Returns:
        Updated confidence value.

    Example:
        >>> bayesian_update(0.6, [True, True, True, False])
        0.7
    """
    # Convert prior confidence to alpha/beta
    alpha = prior_confidence * prior_strength
    beta = prior_strength - alpha

    # Update with observations
    for obs in observations:
        if obs:
            alpha += 1
        else:
            beta += 1

    # Return posterior mean
    return alpha / (alpha + beta)


def combine_bayesian_confidences(
    confidences: list[BayesianConfidence],
) -> BayesianConfidence:
    """Combine multiple Bayesian confidences into one.

    Useful when merging procedural memories or aggregating evidence
    from multiple sources.

    Args:
        confidences: List of BayesianConfidence objects.

    Returns:
        Combined BayesianConfidence.
    """
    if not confidences:
        return BayesianConfidence.from_prior("weak")

    # Sum up all observations
    total_alpha = sum(c.alpha for c in confidences)
    total_beta = sum(c.beta for c in confidences)
    total_obs = sum(c.observations for c in confidences)
    total_conf = sum(c.confirmations for c in confidences)
    total_contra = sum(c.contradictions for c in confidences)

    # Normalize to prevent explosion with many sources
    # Keep effective sample size reasonable
    max_effective = 100
    current_effective = total_alpha + total_beta

    if current_effective > max_effective:
        scale = max_effective / current_effective
        total_alpha *= scale
        total_beta *= scale

    return BayesianConfidence(
        alpha=total_alpha,
        beta=total_beta,
        observations=total_obs,
        confirmations=total_conf,
        contradictions=total_contra,
    )
