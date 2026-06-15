"""Centralized random number generation for path simulation.

Every random draw in the project goes through PathRNG. Centralizing this
gives three things:

1. Reproducibility. Same seed always produces the same draws. This is
   essential for debugging stochastic code -- without it, you can't tell
   whether you fixed a bug or got a different random outcome.

2. Single point of control. When a test fails non-deterministically, there
   is exactly one place to check whether seeding got hooked up correctly.

3. Support for variance reduction and Greeks. Antithetic variates need to
   consume the same Z values negated. Finite-difference Greeks need the
   same draws across bumped and unbumped runs. The reset() method lets a
   caller replay a draw sequence from the beginning.

We use numpy's modern Generator interface (np.random.default_rng) rather
than the legacy module-level np.random.seed(). The legacy API mutates
global state, which is dangerous in library code.
"""

from __future__ import annotations

import numpy as np


class PathRNG:
    """Reproducible standard-normal generator for path simulation."""

    def __init__(self, seed: int | None = None):
        """Create a generator. If seed is None, draws are nondeterministic."""
        self._seed = seed
        self._rng = np.random.default_rng(seed)

    def standard_normal(self, shape: int | tuple[int, ...]) -> np.ndarray:
        """Draw standard normal random variables of the given shape."""
        return self._rng.standard_normal(shape)

    def reset(self) -> None:
        """Reset the generator to its initial seed.

        After reset(), subsequent draws will produce the same sequence as
        when the PathRNG was first constructed. Used by variance reduction
        (antithetic variates) and by Greek computation (shared draws across
        bumped runs).
        """
        self._rng = np.random.default_rng(self._seed)

    @property
    def seed(self) -> int | None:
        """The seed this generator was constructed with."""
        return self._seed
