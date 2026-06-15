# 02 — Simulation Interface and Euler-Maruyama

This document specifies what a stochastic process simulator does, why the interface is shaped the way it is, and how Euler-Maruyama discretizes an Itô SDE into something a computer can run. The actual convergence theory — strong vs weak order, when each one matters — gets its own document later. What's here is the minimum needed to write `src/processes/base.py` without regretting it three weeks from now.

## 1. What the simulator must do

Given an SDE

$$dX_t = \mu(t, X_t) \, dt + \sigma(t, X_t) \, dW_t, \quad X_0 = x_0$$

the simulator returns a collection of independent sample paths discretized on a uniform time grid. Three things are non-negotiable:

**Vectorized over paths.** We need thousands to millions of independent paths per Monte Carlo run, and the inner loop has to use numpy broadcasting rather than Python `for path in paths`. The performance gap is roughly 100x, and at one million paths times one thousand time steps, that's the difference between two seconds and three minutes.

**Reproducible.** Same seed must produce identical paths bit-for-bit. Bugs in stochastic code are nearly impossible to track down without this — every run looks different and you can't tell whether you fixed the bug or got lucky. Reproducibility is also a hard requirement for variance reduction (antithetic variates need to know which $Z$ produced which path) and for finite-difference Greeks (the bumped and unbumped paths must use the same random draws to avoid swamping the sensitivity in MC noise).

**Process-agnostic from the outside.** The pricing layer takes a `StochasticProcess` object and never asks what kind it is. `price_european_call(process, K, T)` should work for GBM today, jump-diffusion tomorrow, and Heston in Phase 5 with zero changes to the pricer. This is the central architectural commitment of the project and the reason for the abstract base class.

## 2. The drift/diffusion abstraction

Every Itô diffusion of the form $dX_t = \mu \, dt + \sigma \, dW_t$ is fully specified by two scalar-valued functions: the drift $\mu(t, x)$ and the diffusion $\sigma(t, x)$. The base class makes these the only thing a subclass has to implement:

```python
class StochasticProcess(ABC):
    @abstractmethod
    def drift(self, t, x): ...
    @abstractmethod
    def diffusion(self, t, x): ...
```

For GBM, $\mu(t, x) = \mu \cdot x$ and $\sigma(t, x) = \sigma \cdot x$, both linear in $x$. For Ornstein-Uhlenbeck, $\mu(t, x) = \kappa(\theta - x)$ and $\sigma(t, x) = \sigma$, mean-reverting drift and constant diffusion. For jump-diffusion we'll extend the abstraction to include a jump intensity, but the diffusion part still fits this template.

The reason to put discretization *outside* of `drift` and `diffusion`: we want to compare schemes (Euler-Maruyama versus Milstein versus exact, where exact exists), and we want the same drift/diffusion code to feed all three. Tightly coupling the SDE to a scheme would force code duplication every time we added a new scheme or a new process.

## 3. Euler-Maruyama

The simplest discretization of an Itô SDE is **Euler-Maruyama**. On a uniform grid $0 = t_0 < t_1 < \cdots < t_n = T$ with step $\Delta t = T/n$:

$$X_{k+1} = X_k + \mu(t_k, X_k) \, \Delta t + \sigma(t_k, X_k) \, \sqrt{\Delta t} \, Z_k$$

where $Z_k \stackrel{\text{iid}}{\sim} \mathcal{N}(0, 1)$.

The scheme is the direct translation of $dX = \mu \, dt + \sigma \, dW$ into discrete time:

- The $dt$ term becomes $\mu \, \Delta t$, evaluated at the *left* endpoint of the interval. (Recall from doc 01 that the Itô integral is non-anticipative; the left-endpoint choice is what makes the discrete sum converge to the Itô integral, not Stratonovich.)
- The $dW$ term becomes $\sigma \, \Delta W_k$ where $\Delta W_k = W_{t_{k+1}} - W_{t_k} \sim \mathcal{N}(0, \Delta t)$, which we generate as $\sqrt{\Delta t} \, Z_k$ with $Z_k$ standard normal.

The scheme has **strong order 0.5** and **weak order 1.0** under mild smoothness conditions on $\mu$ and $\sigma$. Strong order measures path-by-path error: $\mathbb{E}|X_T^{\text{exact}} - X_T^{\text{Euler}}| = O(\Delta t^{0.5})$. Weak order measures error in expectations: $|\mathbb{E}f(X_T^{\text{exact}}) - \mathbb{E}f(X_T^{\text{Euler}})| = O(\Delta t)$ for smooth $f$.

For pricing, weak order is what matters — option prices are expectations. For path-dependent hedging or pathwise Greeks, strong order can matter too. Phase 1 ends with an experiment that measures both empirically and checks they match the theory.

### 3.1 Vectorization across paths

Done naively, the inner loop is

```
for k in range(n_steps):
    for i in range(n_paths):
        paths[i, k+1] = paths[i, k] + drift(t_k, paths[i, k]) * dt + ...
```

This is slow. With numpy, the path dimension vectorizes for free:

```
for k in range(n_steps):
    t_k = k * dt
    x_k = paths[:, k]                       # shape (n_paths,)
    mu_k = self.drift(t_k, x_k)              # shape (n_paths,) via broadcasting
    sigma_k = self.diffusion(t_k, x_k)       # shape (n_paths,)
    paths[:, k+1] = x_k + mu_k*dt + sigma_k*sqrt_dt*Z[:, k]
```

The outer loop over time steps stays in Python because each step depends on the previous one — we can't vectorize the time dimension. The path dimension is fully vectorized because paths are independent. This is the right shape for the algorithm: outer Python loop over $n$ steps, inner numpy loop over $n_{\text{paths}}$.

The expensive operation is the random number generation, which we do *once* before the loop by drawing all $n_{\text{paths}} \times n_{\text{steps}}$ standard normals at the start. This is faster than calling the RNG inside the loop and also makes the random draws inspectable for debugging.

## 4. The simulation interface

```python
simulate(n_paths: int, n_steps: int, T: float, seed: int | None) -> np.ndarray
```

returning an array of shape `(n_paths, n_steps + 1)`. The `+1` accounts for the initial value at $t = 0$: `paths[:, 0] == x0` for every path, and `paths[:, n_steps]` is the terminal value $X_T$.

The shape convention — paths as rows, time as columns — is the same one used by most quant libraries (Glasserman's book, QuantLib's outputs) and lines up with `paths.mean(axis=0)` giving the cross-sectional mean across paths at each time, which is what you want for plotting and for validating the moments.

Time is measured in years and rates are continuously compounded. So `T = 1.0` is one year, and a 5% annual rate is `r = 0.05`. Everywhere in the codebase.

## 5. RNG design

We wrap numpy's modern `np.random.Generator` interface in a small `PathRNG` class. The wrapper exists for three reasons:

A single seed produces a single deterministic stream. `np.random.default_rng(seed)` is the modern API (the legacy `np.random.seed()` mutates global state and shouldn't be used in library code).

The wrapper has a `reset()` method that returns the generator to its initial state, which we'll need for variance reduction and Greeks — antithetic variates need to consume the same draws twice (once positive, once negated), and finite-difference Greeks need to consume the same draws across bumped and unbumped runs.

All random draws in the project go through this one class. That way, when a test fails non-deterministically, there's exactly one place to check whether seeding got hooked up correctly.

We are *not* doing anything fancy with the RNG yet — no Sobol sequences, no antithetic sampling, no stratification. Those belong in `src/variance_reduction/`. The base RNG just gives us reproducible normals.

## 6. What this commits us to

After implementing `base.py` and `rng.py`, the contract is:

- Every new process inherits from `StochasticProcess` and implements `drift` and `diffusion`.
- Every Monte Carlo computation calls `process.simulate(...)` and operates on the returned `(n_paths, n_steps+1)` array. Nothing in `pricing/`, `variance_reduction/`, or `greeks/` reaches inside the process object.
- Reproducibility is a guarantee. Same seed, same paths.

The next document, `03-gbm-derivation.md`, will treat GBM in detail: the exact-simulation scheme (using the closed form from doc 01) versus the Euler-Maruyama scheme, the methodology for validating simulated moments against the analytical $\mathbb{E}[S_t] = S_0 e^{\mu t}$ and $\text{Var}(S_t) = S_0^2 e^{2\mu t}(e^{\sigma^2 t} - 1)$, and the Monte Carlo standard error formulas that tell us when the simulator passes or fails.

## References

- Glasserman, *Monte Carlo Methods in Financial Engineering*. §3.2 covers Euler-Maruyama and the discretization-vs-statistical-error tradeoff. §1.1 has the basic Monte Carlo setup.
- Kloeden and Platen, *Numerical Solution of Stochastic Differential Equations*. The reference for discretization theory; Ch. 9-10 cover Euler-Maruyama strong and weak convergence orders rigorously.
