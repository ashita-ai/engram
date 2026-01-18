"""Confidence propagation between linked memories.

Implements PageRank-style iterative propagation where linked memories
can boost (or reduce) each other's confidence scores.

Design principles:
- Episodes are trusted seeds (immutable, confidence=1.0)
- Damping factor prevents runaway propagation
- Distrust propagation handles contradictions
- Convergence detection stops iteration

Example:
    ```python
    from engram.propagation import propagate_confidence, PropagationConfig

    # Configure propagation
    config = PropagationConfig(
        damping_factor=0.85,
        max_iterations=10,
    )

    # Run propagation
    result = await propagate_confidence(memories, config)
    print(f"Updated {result.memories_updated} memories in {result.iterations} iterations")
    ```

References:
- Pearl 1982: Belief propagation
- PageRank: Iterative ranking with damping
- TrustRank: Trust from seed nodes
- GBR: Good/Bad rank for distrust
"""

from .algorithms import (
    DEFAULT_CONVERGENCE_THRESHOLD,
    DEFAULT_DAMPING_FACTOR,
    DEFAULT_DISTRUST_PENALTY,
    DEFAULT_MAX_BOOST_PER_CYCLE,
    DEFAULT_MAX_ITERATIONS,
    PropagationConfig,
    PropagationResult,
    compute_link_strength,
    propagate_confidence,
    propagate_distrust,
    run_propagation_cycle,
)

__all__ = [
    # Config
    "PropagationConfig",
    "PropagationResult",
    # Functions
    "compute_link_strength",
    "propagate_confidence",
    "propagate_distrust",
    "run_propagation_cycle",
    # Constants
    "DEFAULT_CONVERGENCE_THRESHOLD",
    "DEFAULT_DAMPING_FACTOR",
    "DEFAULT_DISTRUST_PENALTY",
    "DEFAULT_MAX_BOOST_PER_CYCLE",
    "DEFAULT_MAX_ITERATIONS",
]
