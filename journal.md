# Research Journal — Monte Carlo Derivatives

## 2026-06-09 — Project kickoff

Started the Monte Carlo derivatives pricing project as a separate repo from the pairs trading work. Scope set to the full five-phase plan (foundations → European → exotics → variance reduction → Greeks) with Merton jump-diffusion as a stretch goal. Expected timeline 6-8 weeks of real evening/weekend work.

### Architecture decisions

- Separate repo (`monte-carlo-derivatives`) rather than a subdirectory of the pairs trading repo. The two projects share no code and have different runtime characteristics (this one is purely numerical, no market data). Keeping them separate makes the resume story cleaner: stat-arb and derivatives pricing as distinct demonstrations.
- `docs/` is first-class. Derivations precede code in every phase. The math is the project as much as the implementation is.
- `processes/base.py` will define a process-agnostic interface (`simulate(n_paths, n_steps, T, seed)`) so the pricing layer never has to know which SDE it's simulating. This means `price_european_call(process, strike, T)` works for GBM today and jump-diffusion later with no changes to the pricing code.
- Centralized RNG management in `utils/rng.py`. Reproducibility is non-negotiable for debugging, and variance reduction techniques (antithetic, control variates) and Greek computations (pathwise, finite difference) require sharing random draws across runs.

### Written today

- `docs/01-ito-and-sdes.md` — derivation of Itô's lemma from Brownian motion. Covers: the four defining properties of the Wiener process, sample path properties (continuous, nowhere differentiable, unbounded variation, finite quadratic variation), the $L^2$ proof of $[W,W]_T = T$, the multiplication rules $(dt)^2 = 0$ / $dt \cdot dW = 0$ / $(dW)^2 = dt$, sketch of the Itô integral construction with Itô isometry, statement and heuristic derivation of Itô's lemma via Taylor expansion, application to $\ln S_t$ to obtain the closed-form GBM solution $S_t = S_0 \exp((\mu - \sigma^2/2)t + \sigma W_t)$, and derivation of the GBM moments $\mathbb{E}[S_t] = S_0 e^{\mu t}$ and $\text{Var}(S_t) = S_0^2 e^{2\mu t}(e^{\sigma^2 t} - 1)$.
- `README.md` — project overview, phase plan, philosophy section linking to the pairs trading approach.
- This journal entry.

### Next session

- Write `docs/02-gbm-derivation.md`: alternative derivations of the GBM closed form (direct integration vs Itô on $\ln S$), connection to Ornstein-Uhlenbeck, parameterization choices for the simulator (continuous vs discrete time, drift conventions).
- Begin `src/processes/base.py`: the `StochasticProcess` abstract base class. The interface needs to be designed before any concrete process is implemented; once it locks in, every later process inherits.
- First commit and push.
