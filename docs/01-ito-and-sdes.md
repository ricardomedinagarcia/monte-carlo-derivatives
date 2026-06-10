# 01 — Itô's Lemma and Stochastic Differential Equations

This document derives Itô's lemma from first principles and uses it to obtain the closed-form solution to geometric Brownian motion. Everything in the project — every SDE we simulate, every discretization scheme we implement, every option we price — rests on the machinery built here.

The presentation is self-contained but assumes comfort with measure-theoretic probability at the level of expectation, variance, and conditional expectation, plus multivariable calculus and Taylor expansion.

## 1. Why ordinary calculus fails

Suppose we want to model an asset price $S_t$ whose returns are random. A natural first attempt:

$$\frac{dS_t}{dt} = \mu S_t + \sigma S_t \cdot \xi_t$$

where $\xi_t$ is some "random noise" term. The classical move is to integrate both sides. But for this to work, $\xi_t$ has to be the derivative of *something* — call it $W_t$, so that $\xi_t = dW_t/dt$. We want $W_t$ to be a continuous random process whose increments are independent and Gaussian.

That process exists. It's Brownian motion. The problem is that its sample paths are **nowhere differentiable**. The expression $dW_t/dt$ is not a function. So the SDE above, written in this form, doesn't actually mean anything.

The fix is to abandon $dW/dt$ and work with the *differential* form $dW_t$ directly, treating it as a formal symbol that obeys specific multiplication rules. The rules turn out to be:

$$(dt)^2 = 0, \quad dt \cdot dW_t = 0, \quad (dW_t)^2 = dt$$

That last identity is the entire reason stochastic calculus differs from ordinary calculus. It produces an extra term in the chain rule — Itô's correction — that has no classical analog. The rest of this document is a careful derivation of why $(dW_t)^2 = dt$, what Itô's lemma is, and how it gives us the closed-form solution to GBM.

## 2. Brownian motion (the Wiener process)

A **standard Brownian motion** is a stochastic process $\{W_t\}_{t \geq 0}$ defined on some probability space $(\Omega, \mathcal{F}, \mathbb{P})$ satisfying:

1. $W_0 = 0$ almost surely.
2. **Independent increments**: for any $0 \leq s_1 < t_1 \leq s_2 < t_2$, the random variables $W_{t_1} - W_{s_1}$ and $W_{t_2} - W_{s_2}$ are independent.
3. **Gaussian increments**: for $0 \leq s < t$,

$$W_t - W_s \sim \mathcal{N}(0, t - s)$$

4. **Path continuity**: the map $t \mapsto W_t(\omega)$ is continuous for almost every $\omega \in \Omega$.

These four conditions uniquely characterize the process up to indistinguishability (this is the Wiener-Lévy construction theorem; we take it as given).

### 2.1 First moments

From property 3 with $s = 0$:

$$\mathbb{E}[W_t] = 0, \quad \text{Var}(W_t) = t$$

For the covariance, take $0 \leq s \leq t$ and write $W_t = W_s + (W_t - W_s)$. Then

$$\text{Cov}(W_s, W_t) = \text{Cov}(W_s, W_s) + \text{Cov}(W_s, W_t - W_s) = s + 0 = s$$

where the second term vanishes by independent increments. So in general $\text{Cov}(W_s, W_t) = \min(s, t)$.

### 2.2 Sample path properties

Three facts about the trajectories $t \mapsto W_t$ matter for what follows:

**Continuous everywhere.** Built into property 4.

**Differentiable nowhere.** With probability one, $W_t$ is not differentiable at any point. The intuition: the increment $W_{t+h} - W_t$ has standard deviation $\sqrt{h}$, so $(W_{t+h} - W_t)/h$ has standard deviation $1/\sqrt{h}$, which blows up as $h \to 0$. The difference quotient has no limit.

**Unbounded total variation, finite quadratic variation.** This is the key one, and it's the subject of the next section.

## 3. Quadratic variation

Let $\Pi_n = \{0 = t_0 < t_1 < \cdots < t_n = T\}$ be a partition of $[0, T]$, and let $\|\Pi_n\| = \max_i (t_{i+1} - t_i)$ denote its mesh.

**Definition.** The **quadratic variation** of $W$ on $[0, T]$ is

$$[W, W]_T := \lim_{\|\Pi_n\| \to 0} \sum_{i=0}^{n-1} (W_{t_{i+1}} - W_{t_i})^2$$

provided this limit exists in some suitable sense.

**Theorem.** For standard Brownian motion, $[W, W]_T = T$ almost surely (and the convergence holds in $L^2$).

**Proof sketch (the $L^2$ statement).** Write $\Delta_i = W_{t_{i+1}} - W_{t_i}$ and $\delta_i = t_{i+1} - t_i$. By the defining properties of Brownian motion, $\Delta_i \sim \mathcal{N}(0, \delta_i)$ and the $\Delta_i$ are mutually independent across $i$.

Let $Q_n = \sum_{i=0}^{n-1} \Delta_i^2$. We compute its mean and variance.

*Mean.* Since $\mathbb{E}[\Delta_i^2] = \delta_i$,

$$\mathbb{E}[Q_n] = \sum_{i=0}^{n-1} \delta_i = T$$

*Variance.* For $Z \sim \mathcal{N}(0, \sigma^2)$, we have $\mathbb{E}[Z^4] = 3\sigma^4$, so $\text{Var}(Z^2) = \mathbb{E}[Z^4] - (\mathbb{E}[Z^2])^2 = 3\sigma^4 - \sigma^4 = 2\sigma^4$. By independence of the $\Delta_i$:

$$\text{Var}(Q_n) = \sum_{i=0}^{n-1} \text{Var}(\Delta_i^2) = \sum_{i=0}^{n-1} 2 \delta_i^2 \leq 2 \|\Pi_n\| \sum_{i=0}^{n-1} \delta_i = 2 \|\Pi_n\| \cdot T$$

As $\|\Pi_n\| \to 0$, $\text{Var}(Q_n) \to 0$. Combined with $\mathbb{E}[Q_n] = T$, this gives $Q_n \to T$ in $L^2$. $\blacksquare$

### 3.1 The informal identity $(dW)^2 = dt$

The result $[W, W]_T = T$ is what justifies the symbolic rule $(dW_t)^2 = dt$. The sum of squared increments over $[0, T]$ converges to $T$, so each infinitesimal contribution $(dW_t)^2$ behaves like $dt$.

This is not equality of random variables. The increment $dW_t$ is random and has mean zero; $dt$ is deterministic. The statement is about *quadratic variation* — that when we sum many small squared increments, we get the deterministic time interval. Inside an Itô calculation, the rule $(dW)^2 = dt$ replaces a random quantity by its $L^2$-limit, and the resulting integral identities turn out to be correct.

Compare with the corresponding rule for $dt$: a partition sum $\sum (t_{i+1} - t_i)^2$ is bounded by $\|\Pi_n\| \cdot T$, which goes to zero. So $(dt)^2 = 0$. And $\sum (t_{i+1} - t_i)(W_{t_{i+1}} - W_{t_i})$ has variance $\sum \delta_i^3 \to 0$, so $dt \cdot dW = 0$.

These three rules — $(dt)^2 = 0$, $dt \cdot dW = 0$, $(dW)^2 = dt$ — are the entire multiplication table of Itô calculus.

## 4. The Itô integral (sketch)

Before stating Itô's lemma, we need to know what $\int_0^T \sigma_s \, dW_s$ means, since Brownian motion has unbounded variation and standard Riemann-Stieltjes integration breaks.

The construction proceeds in stages:

**Step 1.** For simple processes — those of the form $\sigma_s = \sum_j a_j \mathbf{1}_{(t_j, t_{j+1}]}(s)$ where the $a_j$ are random variables measurable with respect to $\mathcal{F}_{t_j}$ — define

$$\int_0^T \sigma_s \, dW_s = \sum_j a_j (W_{t_{j+1}} - W_{t_j})$$

The critical detail: each $a_j$ is evaluated at the *left endpoint* of its interval. This **non-anticipative** choice is what makes the resulting integral a martingale. Other choices (Stratonovich, for instance) give different calculi.

**Step 2.** For any $\sigma$ in the space

$$\mathcal{L}^2 := \left\{\sigma : [0,T] \times \Omega \to \mathbb{R} \,\Big|\, \sigma \text{ is adapted}, \ \mathbb{E}\int_0^T \sigma_s^2 \, ds < \infty\right\}$$

approximate by simple processes and pass to the $L^2$ limit. The key tool is the **Itô isometry**:

$$\mathbb{E}\left[\left(\int_0^T \sigma_s \, dW_s\right)^2\right] = \mathbb{E}\int_0^T \sigma_s^2 \, ds$$

Two properties of the Itô integral that we use repeatedly:

- $\mathbb{E}\left[\int_0^T \sigma_s \, dW_s\right] = 0$ for any $\sigma \in \mathcal{L}^2$.
- The process $M_t = \int_0^t \sigma_s \, dW_s$ is a martingale: $\mathbb{E}[M_t \mid \mathcal{F}_s] = M_s$ for $s \leq t$.

Both follow from the construction.

## 5. Itô's lemma

We can now state the stochastic chain rule.

### 5.1 Setup

An **Itô process** is one of the form

$$X_t = X_0 + \int_0^t \mu_s \, ds + \int_0^t \sigma_s \, dW_s$$

which we write in differential shorthand as

$$dX_t = \mu_t \, dt + \sigma_t \, dW_t$$

Here $\mu_t$ is the **drift** (the deterministic rate of change) and $\sigma_t$ is the **diffusion** (the size of the random perturbation). Both may depend on $t$, $\omega$, and $X_t$ itself.

### 5.2 Statement (one-dimensional Itô's lemma)

Let $X_t$ be an Itô process with $dX_t = \mu_t \, dt + \sigma_t \, dW_t$, and let $f : [0, T] \times \mathbb{R} \to \mathbb{R}$ be in $C^{1,2}$ (continuously differentiable in $t$, twice continuously differentiable in $x$). Then $f(t, X_t)$ is also an Itô process, and

$$df(t, X_t) = \left(\frac{\partial f}{\partial t} + \mu_t \frac{\partial f}{\partial x} + \frac{1}{2}\sigma_t^2 \frac{\partial^2 f}{\partial x^2}\right) dt + \sigma_t \frac{\partial f}{\partial x} dW_t$$

The classical chain rule would give us only the first two terms inside the parentheses plus the $dW$ term. The third term, $\frac{1}{2}\sigma_t^2 \frac{\partial^2 f}{\partial x^2}$, is **Itô's correction**. It comes from the $(dW)^2 = dt$ rule.

### 5.3 Heuristic derivation

Expand $f(t + dt, X_t + dX_t)$ to second order in Taylor series:

$$df = \frac{\partial f}{\partial t} dt + \frac{\partial f}{\partial x} dX + \frac{1}{2}\frac{\partial^2 f}{\partial x^2}(dX)^2 + \frac{\partial^2 f}{\partial t \partial x} \, dt \, dX + \frac{1}{2}\frac{\partial^2 f}{\partial t^2}(dt)^2 + \cdots$$

In classical calculus we drop everything past first order. In Itô calculus we have to keep second-order terms when they involve $dW$, because $(dW)^2$ contributes at first order in $dt$.

Apply the multiplication rules. Substituting $dX = \mu dt + \sigma dW$:

$$(dX)^2 = (\mu \, dt + \sigma \, dW)^2 = \mu^2 (dt)^2 + 2\mu\sigma \, dt \, dW + \sigma^2 (dW)^2$$

Using $(dt)^2 = 0$, $dt \cdot dW = 0$, and $(dW)^2 = dt$:

$$(dX)^2 = \sigma^2 \, dt$$

Similarly, $dt \cdot dX = \mu (dt)^2 + \sigma \, dt \, dW = 0$, and $(dt)^2 = 0$. So all the higher-order terms vanish except $(dX)^2$, which contributes a $dt$ term.

Putting it together:

$$df = \frac{\partial f}{\partial t} dt + \frac{\partial f}{\partial x}(\mu \, dt + \sigma \, dW) + \frac{1}{2}\frac{\partial^2 f}{\partial x^2} \sigma^2 \, dt$$

$$= \left(\frac{\partial f}{\partial t} + \mu \frac{\partial f}{\partial x} + \frac{1}{2}\sigma^2 \frac{\partial^2 f}{\partial x^2}\right) dt + \sigma \frac{\partial f}{\partial x} dW$$

This matches the statement. A rigorous proof replaces "$=$" between symbolic differentials with limits of partition sums and uses Itô isometry plus path continuity to push the convergence through; the heuristic above is correct as a computation but is not a proof.

### 5.4 The integral form

The differential form $df = (\cdots) dt + (\cdots) dW$ is shorthand for the integral identity

$$f(T, X_T) - f(0, X_0) = \int_0^T \left(\frac{\partial f}{\partial t} + \mu_s \frac{\partial f}{\partial x} + \frac{1}{2}\sigma_s^2 \frac{\partial^2 f}{\partial x^2}\right) ds + \int_0^T \sigma_s \frac{\partial f}{\partial x} dW_s$$

where the derivatives are evaluated at $(s, X_s)$. This integral form is what you actually use when computing expectations: the second integral has mean zero (Itô integrals are martingales), so taking $\mathbb{E}[\cdot]$ leaves only the drift integral.

## 6. The canonical example: geometric Brownian motion

Consider the SDE

$$dS_t = \mu S_t \, dt + \sigma S_t \, dW_t, \quad S_0 > 0$$

This is the model used in the Black-Scholes derivation. The drift is proportional to $S_t$ (multiplicative), as is the diffusion. We want a closed-form expression for $S_t$.

### 6.1 Apply Itô to $\ln S$

Let $f(s) = \ln s$. Then $f'(s) = 1/s$, $f''(s) = -1/s^2$. Apply Itô's lemma with $\mu_t = \mu S_t$, $\sigma_t = \sigma S_t$:

$$d(\ln S_t) = \left(0 + \mu S_t \cdot \frac{1}{S_t} + \frac{1}{2}(\sigma S_t)^2 \cdot \left(-\frac{1}{S_t^2}\right)\right) dt + \sigma S_t \cdot \frac{1}{S_t} \, dW_t$$

$$= \left(\mu - \frac{\sigma^2}{2}\right) dt + \sigma \, dW_t$$

The right side has constant coefficients. Integrating from $0$ to $t$:

$$\ln S_t - \ln S_0 = \left(\mu - \frac{\sigma^2}{2}\right) t + \sigma W_t$$

Exponentiating:

$$\boxed{S_t = S_0 \exp\left(\left(\mu - \frac{\sigma^2}{2}\right) t + \sigma W_t\right)}$$

This is the closed-form solution to GBM. Three things to notice:

The drift inside the exponent is $\mu - \sigma^2/2$, not $\mu$. The $-\sigma^2/2$ correction comes directly from Itô's correction term applied to $\ln$. People who forget this and use $\exp(\mu t + \sigma W_t)$ get the wrong answer.

$\ln S_t$ is Gaussian (a deterministic function of $t$ plus $\sigma W_t$), so $S_t$ is **lognormal**. This is why lognormality is built into Black-Scholes.

The solution exists for all $t \geq 0$ and stays strictly positive almost surely, which makes GBM a sensible model for prices. (Compare with arithmetic Brownian motion $dS = \mu \, dt + \sigma \, dW$, which can go negative.)

### 6.2 Moments of GBM

We compute $\mathbb{E}[S_t]$ and $\text{Var}(S_t)$ — these are the analytical targets our simulation engine has to hit in Phase 1.

Recall that $W_t \sim \mathcal{N}(0, t)$, so

$$\ln S_t \sim \mathcal{N}\left(\ln S_0 + (\mu - \sigma^2/2) t, \ \sigma^2 t\right)$$

For $X \sim \mathcal{N}(m, v)$, the moment generating function of the lognormal $e^X$ is $\mathbb{E}[e^{kX}] = \exp(km + k^2 v / 2)$.

**Mean.** Setting $k = 1$:

$$\mathbb{E}[S_t] = \exp\left(\ln S_0 + (\mu - \sigma^2/2) t + \frac{1}{2} \sigma^2 t\right) = S_0 e^{\mu t}$$

The Itô correction $-\sigma^2/2$ in the exponent and the lognormal correction $+\sigma^2/2$ from Jensen's inequality cancel exactly. The expected price grows at rate $\mu$, the drift parameter — clean.

**Second moment.** Setting $k = 2$:

$$\mathbb{E}[S_t^2] = \exp\left(2 \ln S_0 + 2(\mu - \sigma^2/2) t + 2 \sigma^2 t\right) = S_0^2 \, e^{2\mu t + \sigma^2 t}$$

**Variance.**

$$\text{Var}(S_t) = \mathbb{E}[S_t^2] - \mathbb{E}[S_t]^2 = S_0^2 e^{2\mu t}(e^{\sigma^2 t} - 1)$$

These two formulas — $\mathbb{E}[S_t] = S_0 e^{\mu t}$ and $\text{Var}(S_t) = S_0^2 e^{2\mu t}(e^{\sigma^2 t} - 1)$ — are the analytical ground truth our simulated GBM paths have to match within Monte Carlo standard error. That validation is the first concrete output of Phase 1.

## 7. Multidimensional Itô (preview)

For SDEs with multiple correlated Brownian motions (Heston's stochastic volatility model, basket options on $n$ correlated assets), we need the multidimensional version of Itô's lemma. We'll state it carefully when we need it; here is the rule for the multiplication table:

If $dW^{(i)}_t$ and $dW^{(j)}_t$ are two Brownian motions with instantaneous correlation $\rho_{ij}$, then

$$dW^{(i)}_t \cdot dW^{(j)}_t = \rho_{ij} \, dt$$

For independent Brownians ($\rho = 0$) this gives zero. For the same Brownian ($i = j$, $\rho = 1$) it recovers $(dW)^2 = dt$. Every cross-term in a multidimensional Itô calculation reduces to one of these cases.

## 8. What this gives us

Three things now in hand:

1. A rigorous definition of $\int_0^T \sigma_s \, dW_s$ as a martingale-valued process with the Itô isometry.
2. A chain rule (Itô's lemma) that propagates SDEs through smooth functions, with the $\frac{1}{2}\sigma^2 f''$ correction.
3. A closed-form solution to GBM and explicit formulas for its first two moments.

These three are enough to begin Phase 1: implementing simulation engines for GBM and other processes, validating their simulated moments against the analytical formulas in (3), and analyzing their discretization error.

The next document, `02-gbm-derivation.md`, will go deeper into the GBM solution — alternative derivations, the connection to the Ornstein-Uhlenbeck process, and a careful treatment of the parameterization choices that matter when implementing the simulator.

## References

- Shreve, *Stochastic Calculus for Finance II: Continuous-Time Models*. Ch. 3-4 cover Brownian motion and Itô calculus at this level.
- Øksendal, *Stochastic Differential Equations*. The standard graduate text; Ch. 3 has the rigorous construction of the Itô integral.
- Karatzas and Shreve, *Brownian Motion and Stochastic Calculus*. The reference for path properties and quadratic variation.
- Glasserman, *Monte Carlo Methods in Financial Engineering*. Ch. 3 connects the theory to simulation; the project draws on Ch. 3-7 heavily.
