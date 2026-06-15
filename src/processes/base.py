"""Abstract base class for one-dimensional Itô diffusions.

Every SDE in this project subclasses StochasticProcess. The base class
exposes a single public method, simulate(), which takes process-agnostic
inputs (number of paths, number of time steps, terminal time, seed) and
returns a 2D ndarray of sample paths.

Subclasses implement two methods:
    drift(t, x)      -- the deterministic rate mu(t, x) in dX = mu dt + sigma dW
    diffusion(t, x)  -- the noise scale sigma(t, x) in dX = mu dt + sigma dW

Both methods are called with t as a Python scalar and x as a 1D ndarray of
shape (n_paths,), and must return a scalar or a 1D ndarray of shape (n_paths,).
Numpy broadcasting handles the common case where drift/diffusion are simple
algebraic expressions in x.

The default simulation scheme is Euler-Maruyama. Subclasses with a closed-form
solution (e.g., GBM via the exponential representation) may additionally
override simulate_exact() to produce paths with no discretization error,
which is useful as a validation reference.

See docs/02-simulation-interface.md for the design rationale.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Union

import numpy as np

from src.utils.rng import PathRNG

ArrayLike = Union[float, np.ndarray]


class StochasticProcess(ABC):
    """Abstract base for one-dimensional Itô diffusions.

    Models SDEs of the form
        dX_t = drift(t, X_t) dt + diffusion(t, X_t) dW_t,    X_0 = x0
    where W is a standard Brownian motion.

    Subclasses must implement drift() and diffusion() and call super().__init__(x0).
    """

    def __init__(self, x0: float):
        if not np.isfinite(x0):
            raise ValueError(f"x0 must be finite, got {x0}")
        self.x0 = float(x0)

    # ------------------------------------------------------------------ #
    # Abstract interface                                                  #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def drift(self, t: float, x: ArrayLike) -> ArrayLike:
        """Drift coefficient mu(t, x) in dX = mu dt + sigma dW.

        Called with t scalar and x of shape (n_paths,). Should return a scalar
        or array of the same shape as x.
        """
        ...

    @abstractmethod
    def diffusion(self, t: float, x: ArrayLike) -> ArrayLike:
        """Diffusion coefficient sigma(t, x) in dX = mu dt + sigma dW.

        Called with t scalar and x of shape (n_paths,). Should return a scalar
        or array of the same shape as x.
        """
        ...

    # ------------------------------------------------------------------ #
    # Public simulation API                                               #
    # ------------------------------------------------------------------ #

    def simulate(
        self,
        n_paths: int,
        n_steps: int,
        T: float,
        seed: int | None = None,
        scheme: str = "euler",
    ) -> np.ndarray:
        """Simulate sample paths of the process.

        Parameters
        ----------
        n_paths : int
            Number of independent sample paths.
        n_steps : int
            Number of time steps. Output paths have n_steps + 1 columns
            (including the initial value at t = 0).
        T : float
            Terminal time. Time step is dt = T / n_steps.
        seed : int or None, optional
            Random seed for reproducibility. If None, draws are nondeterministic.
        scheme : str, optional
            Discretization scheme. One of:
              - 'euler' : Euler-Maruyama (default).
              - 'exact' : closed-form simulation, only if the subclass
                          implements simulate_exact().

        Returns
        -------
        paths : ndarray, shape (n_paths, n_steps + 1)
            paths[i, j] is X at time j*dt on path i.
            paths[:, 0] equals self.x0 for every path.
            paths[:, n_steps] is the terminal value X_T on each path.
        """
        self._validate_simulate_args(n_paths, n_steps, T)

        if scheme == "euler":
            return self._simulate_euler(n_paths, n_steps, T, seed)
        elif scheme == "exact":
            return self.simulate_exact(n_paths, n_steps, T, seed)
        else:
            raise ValueError(
                f"Unknown scheme {scheme!r}. Valid choices are 'euler' or 'exact'."
            )

    def simulate_exact(
        self,
        n_paths: int,
        n_steps: int,
        T: float,
        seed: int | None = None,
    ) -> np.ndarray:
        """Exact (closed-form) path simulation.

        Subclasses with closed-form solutions override this. The default
        raises NotImplementedError.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not provide exact simulation. "
            f"Use scheme='euler' instead."
        )

    def time_grid(self, n_steps: int, T: float) -> np.ndarray:
        """Return the time grid corresponding to a simulate() call.

        Useful for plotting: pair time_grid(n_steps, T) with each row of paths.
        """
        return np.linspace(0.0, T, n_steps + 1)

    # ------------------------------------------------------------------ #
    # Internal: Euler-Maruyama loop                                       #
    # ------------------------------------------------------------------ #

    def _simulate_euler(
        self, n_paths: int, n_steps: int, T: float, seed: int | None
    ) -> np.ndarray:
        """Euler-Maruyama scheme.

        For dX = mu(t, X) dt + sigma(t, X) dW, the discretization is
            X_{k+1} = X_k + mu(t_k, X_k) * dt + sigma(t_k, X_k) * sqrt(dt) * Z_k
        with Z_k ~ N(0, 1) iid.

        Strong order 0.5, weak order 1.0 for smooth coefficients (see
        docs/02-simulation-interface.md).
        """
        rng = PathRNG(seed)
        dt = T / n_steps
        sqrt_dt = np.sqrt(dt)

        paths = np.empty((n_paths, n_steps + 1), dtype=np.float64)
        paths[:, 0] = self.x0

        # Pre-draw all normal increments. Faster than calling rng inside the
        # loop and lets us inspect the draws when debugging.
        Z = rng.standard_normal((n_paths, n_steps))

        for k in range(n_steps):
            t_k = k * dt
            x_k = paths[:, k]
            mu_k = self.drift(t_k, x_k)
            sigma_k = self.diffusion(t_k, x_k)
            paths[:, k + 1] = x_k + mu_k * dt + sigma_k * sqrt_dt * Z[:, k]

        return paths

    # ------------------------------------------------------------------ #
    # Internal: argument validation                                       #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _validate_simulate_args(n_paths: int, n_steps: int, T: float) -> None:
        if not isinstance(n_paths, (int, np.integer)) or n_paths <= 0:
            raise ValueError(f"n_paths must be a positive integer, got {n_paths!r}")
        if not isinstance(n_steps, (int, np.integer)) or n_steps <= 0:
            raise ValueError(f"n_steps must be a positive integer, got {n_steps!r}")
        if not np.isfinite(T) or T <= 0:
            raise ValueError(f"T must be a positive finite number, got {T!r}")
