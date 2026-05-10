---
name: add-mc-variance-reduction
description: How to add a Monte Carlo variance-reduction technique to stochastic-rs-quant. Covers antithetic, control-variate, stratified, importance, quasi-MC (Halton/Sobol), and MLMC. Returns McEstimate<T> with 95% CI.
---

# MC variance reduction — stochastic-rs-quant

Variance-reduction techniques in `stochastic-rs-quant` follow a uniform
contract: an MC estimator returns an `McEstimate<T>` struct carrying
`(point_estimate, std_error, n_paths)` from which the caller can
compute a 95 % confidence interval. New estimators ship with a
**reference test** asserting the variance is genuinely reduced versus
plain MC on a synthetic problem.

This SKILL covers the six classical techniques:

| Technique           | Complexity | Use case |
|---------------------|------------|----------|
| Antithetic          | trivial    | symmetric payoffs (calls / puts under GBM) |
| Control variate     | low        | known-mean control with high correlation |
| Stratified          | medium     | low-dim integrand; deterministic strata |
| Importance sampling | medium     | rare events (deep OTM, default probability) |
| Quasi-MC (Halton/Sobol) | medium  | low-discrepancy sequences; smooth integrands |
| MLMC (Multi-Level)  | high       | path-dependent + grid-discretisation error |

References: Glasserman 2003 (chapters 4 & 5 for AT/CV; 6 for IS; 7 for
QMC); Giles 2015 for MLMC.

## 1. The `McEstimate<T>` contract

```rust
// stochastic-rs-quant/src/mc/types.rs

#[derive(Debug, Clone)]
pub struct McEstimate<T: FloatExt> {
    /// Point estimate (sample mean of the payoffs).
    pub mean: T,
    /// Sample standard error (sigma / sqrt(N)).
    pub stderr: T,
    /// Number of paths used.
    pub n_paths: usize,
}

impl<T: FloatExt> McEstimate<T> {
    /// 95% confidence interval (mean ± 1.96 * stderr).
    pub fn ci_95(&self) -> (T, T) {
        let z = T::from_f64_fast(1.96);
        (self.mean - z * self.stderr, self.mean + z * self.stderr)
    }

    /// Variance reduction factor relative to a baseline plain-MC stderr.
    /// Returned as `baseline_stderr / self.stderr` (factor > 1 means
    /// improvement).
    pub fn vr_factor(&self, baseline_stderr: T) -> T {
        baseline_stderr / self.stderr
    }
}
```

Estimators always return `McEstimate<T>`, never raw `f64`. This keeps
the variance information attached to the point estimate; callers
asking for a CI never have to chase down the stderr separately.

## 2. Antithetic — the cheapest win

For an SDE driven by `Z ~ N(0, 1)`:

```rust
pub fn price_antithetic<F>(payoff: F, n_pairs: usize, seed: u64) -> McEstimate<f64>
where F: Fn(f64) -> f64 + Sync {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut samples = Vec::with_capacity(n_pairs);
    for _ in 0..n_pairs {
        let z: f64 = StandardNormal.sample(&mut rng);
        let plus  = payoff( z);
        let minus = payoff(-z);   // antithetic pair
        samples.push((plus + minus) * 0.5);
    }
    let n = samples.len();
    let mean = samples.iter().sum::<f64>() / n as f64;
    let var: f64 = samples.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
    McEstimate { mean, stderr: (var / n as f64).sqrt(), n_paths: 2 * n }
}
```

Antithetic works **only** when the payoff is monotonic in `z`. For
strangles / digitals, antithetic can *increase* variance — always
include the variance-reduction reference test (§7).

## 3. Control variate — when you know `E[Y]`

Pick a control `Y(path)` whose mean `E[Y]` is known analytically and
which has high correlation with the payoff `f(path)`. Then:

```rust
estimator(f) = sample_mean(f) - β * (sample_mean(Y) - E[Y])
where β = Cov(f, Y) / Var(Y)
```

The optimal `β` minimises variance; estimate it from a pilot run of
`n_pilot` paths, then use it on the production `n` paths.

For European options under GBM, the natural control is the GBM
terminal price itself: `Y = S_T - K * e^{-rT}` has known mean = 0 if
correlation is high.

## 4. Quasi-MC — Halton / Sobol

Replace `StandardNormal::sample(rng)` with deterministic
low-discrepancy sequences:

```rust
use sobol_rs::Sobol;

pub fn price_qmc(...) -> McEstimate<f64> {
    let mut sobol = Sobol::new(d);  // d-dimensional sequence
    let mut samples = Vec::with_capacity(n);
    for _ in 0..n {
        let u: Vec<f64> = sobol.next().unwrap();
        let z: Vec<f64> = u.iter().map(|&p| inverse_normal_cdf(p)).collect();
        // ... payoff(z) ...
    }
    // ...
}
```

QMC integration error scales like `(log N)^d / N` rather than
`N^{-1/2}`, so for smooth integrands and small d the convergence is
much faster. Only the dimension `d` is set by the path discretisation
— a 100-step Euler scheme is d = 100, which Sobol handles up to d ≈
1000 reasonably well.

## 5. MLMC — multilevel for path-dependent payoffs

For path-dependent payoffs (Asian, lookback) where the discretisation
introduces O(h^α) bias, MLMC combines coarse + fine levels:

```
E[P_∞] ≈ E[P_0] + Σ_l E[P_l - P_{l-1}]
```

with `P_l` evaluated on `2^l` time steps. The number of paths
decreases geometrically with level. See `mc/mlmc.rs` for the workspace
implementation and Giles (2015) for the algorithmic details.

## 6. Common Random Numbers (CRN)

For Greeks via finite differences (delta = `(P(S+h) - P(S-h)) / 2h`),
share the *same* random draws between the bumped and base pricers.
This is the foundation of the single-pass `greeks()` MC override (see
`greeks-pattern` SKILL).

```rust
let z = sample_terminal_normals(seed, n_paths, n_steps);
let p_base = price_with_normals(s,     z.view());
let p_up   = price_with_normals(s + h, z.view());   // SAME z!
let delta  = (p_up - p_base) / h;
```

Without CRN, the finite-difference estimator's variance is the *sum*
of two independent sample variances; with CRN, it reduces to the
variance of the difference, often by a factor of 100×.

## 7. Mandatory test: variance-reduction factor

```rust
#[test]
fn antithetic_reduces_variance() {
    let plain    = price_plain   (..., seed=42, n=10_000);
    let antithetic = price_antithetic(..., seed=42, n=5_000);  // same total path budget
    let factor = antithetic.vr_factor(plain.stderr);
    assert!(
        factor > 1.5,
        "antithetic should reduce stderr by ≥1.5×, got {factor}"
    );
}
```

Pin both seeds so the test is deterministic. The threshold (1.5× here)
should be set conservatively but high enough that the test fails if
the technique was wired backward.

## 8. Anti-patterns

- **Do not** return a raw `f64` mean without a stderr. Always
  `McEstimate<T>`.
- **Do not** apply antithetic to discontinuous payoffs (digital
  options, indicator functions). Variance can go up.
- **Do not** estimate the optimal control-variate `β` on the
  production paths and re-use the same paths for the estimator.
  Pilot + production must be independent draws.
- **Do not** mix MC and QMC randomness in the same estimator without
  a randomised-QMC scheme. Plain Sobol + plain `StandardNormal::sample`
  produces neither MC convergence nor QMC convergence.
- **Do not** skip the variance-reduction test (§7). It's the only
  check that the technique is wired the right way around.

## 9. Reference impls

- `mc/antithetic.rs` — antithetic for European calls under GBM /
  Heston.
- `mc/control_variate.rs` — control variate for Asian options.
- `mc/importance_sampling.rs` — Esscher tilt for OTM digital options.
- `mc/lsm.rs` — Longstaff-Schwartz for American options (uses CRN
  internally).
- `mc/mlmc.rs` — multilevel for path-dependent payoffs.

## Related SKILLs

- `greeks-pattern` — uses CRN for the single-pass `greeks()` override.
- `add-diffusion-process` — produces the underlying paths; seeded
  constructors are mandatory for CRN.
- `bench-writing` — variance-reduction factor is a natural benchmark
  metric.
