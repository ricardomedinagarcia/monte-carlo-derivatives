# Monte Carlo Methods for Derivatives Pricing

A reference implementation of Monte Carlo techniques for pricing options and other derivatives, built from first principles. Each component is preceded by a written derivation in `docs/` and validated against a known analytical result before use.

## Project status

**Phase 1: Stochastic foundations** — in progress.

Five planned phases:

1. Stochastic foundations: SDE simulation engine (GBM, Ornstein-Uhlenbeck), Euler-Maruyama and Milstein discretization, moment-based validation.
2. European pricing and Black-Scholes: analytical BS formula, Monte Carlo pricing, convergence to BS as the unit test.
3. Exotic options: Asian (arithmetic and geometric), barrier with discrete-monitoring bias correction, lookback, basket.
4. Variance reduction: antithetic variates, control variates, importance sampling, quasi-Monte Carlo (Sobol).
5. Greeks: finite difference, pathwise differentiation, likelihood ratio.

A stretch goal of Merton jump-diffusion (semi-closed-form European prices for additional validation) sits at the end.

## Philosophy

The project sits next to my pairs trading research platform (a separate repo) and is built under the same operating principles:

- **Derive before you code.** Each phase is gated on a written derivation in `docs/`. The first one, `docs/01-ito-and-sdes.md`, builds Itô's lemma from Brownian motion and uses it to obtain the closed-form solution to geometric Brownian motion.
- **Validate against ground truth before using.** The simulation engine is validated by matching simulated moments to analytical formulas. The Monte Carlo pricer is validated against Black-Scholes. The exotic-option engine reuses geometric Asian (closed form) to validate arithmetic Asian.
- **Honest documentation of what fails.** Discretization error, Monte Carlo standard error, and the limits of each method are reported in the journal and in the final writeup.

## Structure

```
monte-carlo-derivatives/
├── docs/                          Written derivations (math, first)
├── src/
│   ├── processes/                 SDE simulators (GBM, OU, jump-diffusion)
│   ├── pricing/                   Black-Scholes, European, Asian, barrier, lookback, basket
│   ├── variance_reduction/        Antithetic, control variates, importance sampling, QMC
│   ├── greeks/                    Finite difference, pathwise, likelihood ratio
│   └── utils/                     RNG management, convergence diagnostics
├── tests/                         pytest unit tests
├── notebooks/                     Exploratory work and final result visualization
└── journal.md                     Research journal
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Then run the test suite (once tests exist):

```bash
pytest tests/
```

## References

The treatment in `docs/` follows Shreve, *Stochastic Calculus for Finance II*, for the theoretical foundations and Glasserman, *Monte Carlo Methods in Financial Engineering*, for the simulation methodology. Specific citations appear at the end of each document in `docs/`.

## License

MIT.
