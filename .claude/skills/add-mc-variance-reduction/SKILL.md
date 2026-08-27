---
name: add-mc-variance-reduction
description: How to add a Monte Carlo variance-reduction technique to stochastic-rs-stochastic. Covers antithetic, control-variate, stratified, importance, quasi-MC (Halton/Sobol), and MLMC. Returns McEstimate<T> with 95% CI.
---

# MC variance reduction — stochastic-rs-stochastic

Variance-reduction techniques in `stochastic-rs-stochastic` follow a uniform
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
// stochastic-rs-stochastic/src/mc/mod.rs

#[derive(Debug, Clone)]
pub struct McEstimate<T: FloatExt> {
    /// Estimated mean.
    pub mean: T,
    /// Standard error of the estimate.
    pub std_err: T,
    /// Number of samples used.
    pub n_samples: usize,
}

impl<T: FloatExt> McEstimate<T> {
    /// Symmetric confidence interval `[mean ± z · std_err]`.
    pub fn confidence_interval(&self, z: T) -> (T, T);
    /// 95% confidence interval (z = 1.96).
    pub fn ci_95(&self) -> (T, T);
}
```

Field names are `std_err` and `n_samples` — not `stderr` / `n_paths`.
There is **no `vr_factor` method**; compute the ratio yourself in the
§7 test.

Estimators always return `McEstimate<T>`, never raw `f64`. This keeps
the variance information attached to the point estimate; callers
asking for a CI never have to chase down the standard error separately.

### The estimator signature

The shipped estimators take a **payoff closure over a normal draw**,
not a seed:

```rust
// mc/antithetic.rs
pub fn estimate<T, F>(n_paths: usize, dim: usize, payoff: F) -> McEstimate<T>
where T: FloatExt, F: Fn(&Array1<T>) -> T;
pub fn estimate_par<T, F>(…) -> McEstimate<T> where F: … + Sync;

// mc/control_variates.rs
pub fn estimate<T, F, G>(
    n_paths: usize, dim: usize, payoff: F, control: G, control_mean: T,
) -> McEstimate<T>;

// mc/importance_sampling.rs
pub fn estimate<T, F>(
    n_paths: usize, dim: usize, payoff: F, shift: &Array1<T>,
) -> McEstimate<T>;
```

Match that shape for a new technique: `(n_paths, dim, payoff, …)`,
returning `McEstimate<T>`, with an `estimate_par` twin where the payoff
can be `Sync`.

**Known limitation, not a thing to imitate:** `antithetic`,
`control_variates` and `importance_sampling` draw from
`T::normal_array(dim, 0, 1)`, which is unseeded — so those three are
**not reproducible**. Only `stratified` exposes seeded variants
(`stratified_normals_1d_seeded(n, seed)`,
`stratified_normals_seeded(n_samples, dim, seed)`). If you add a
technique, give it a seeded entry point; if you write a test against
one of the unseeded three, it can only assert statistical bounds, not
pinned numbers.

## 2. Antithetic — the cheapest win

For an SDE driven by `Z ~ N(0, 1)`:

```rust
// mc/antithetic.rs — the shipped body, abridged
pub fn estimate<T, F>(n_paths: usize, dim: usize, payoff: F) -> McEstimate<T>
where T: FloatExt, F: Fn(&Array1<T>) -> T
{
    let two = T::from_f64_fast(2.0);
    let (mut sum, mut sum_sq) = (T::zero(), T::zero());

    for _ in 0..n_paths {
        // The workspace's own Gaussian draw. Per `dev-rules` §7a,
        // `rand_distr::StandardNormal` and `StdRng` belong to `benches/`
        // — never to library code.
        let z = T::normal_array(dim, T::zero(), T::one());
        let neg_z = z.mapv(|v| -v);
        let y = (payoff(&z) + payoff(&neg_z)) / two;
        sum += y;
        sum_sq += y * y;
    }

    let n = T::from_usize_(n_paths);
    let mean = sum / n;
    let variance = sum_sq / n - mean * mean;
    McEstimate { mean, std_err: (variance / n).sqrt(), n_samples: n_paths }
}
```

Note `n_samples: n_paths`, not `2 * n_paths`: a pair counts as one
sample of the averaged estimator, which is what makes the standard
error comparable to plain MC at the same `n_samples`.

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

The optimal `β` minimises variance. Textbook practice is to estimate it
on an independent pilot run and apply it to the production paths, so the
coefficient is uncorrelated with the sample it multiplies.

**The shipped `control_variates::estimate` does not do that** — it
takes `(n_paths, dim, payoff, control, control_mean)` in one call and
estimates `c*` from the same sample it then corrects. That introduces a
small `O(1/n)` bias, which is the standard trade-off (Glasserman §4.1
discusses it) but is worth knowing before you cite this module as the
pattern for a bias-sensitive application. If you add a technique with
the same structure, either follow suit and document the bias, or take a
separate pilot count and document that instead.

For European options under GBM, the natural control is the GBM
terminal price itself: `Y = S_T - K e^{-rT}`, whose mean is known.

## 4. Quasi-MC — Halton / Sobol

Replace the pseudo-random draw with a deterministic low-discrepancy
sequence. Both generators are **in-tree** — there is no `sobol_rs`
dependency:

```rust
use crate::mc::sobol::Sobol;      // or crate::mc::halton::Halton

let sobol = Sobol::new(dim);              // dim-dimensional sequence
let u = sobol.sample::<f64>(n_points);    // Array2<f64>, shape (n_points, dim)
// map each u through the inverse normal CDF, then evaluate the payoff
```

Both types share the same three-method surface: `new(n_dims)`,
`sample::<T>(n_points) -> Array2<T>`, `n_dims()`.

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
    let plain = plain_mc_estimate(10_000, dim, &payoff);
    let anti  = antithetic::estimate(5_000, dim, &payoff);  // same draw budget
    let factor = plain.std_err / anti.std_err;              // no vr_factor method
    assert!(
        factor > 1.5,
        "antithetic should reduce std_err by >=1.5x, got {factor}"
    );
}
```

Compute the ratio explicitly — `McEstimate` has no `vr_factor`. The
threshold (1.5x here) should be conservative but high enough that the
test fails if the technique was wired backward.

Pin the seed **where the technique lets you** (`stratified`'s `*_seeded`
functions). For the three unseeded estimators the test is inherently
statistical: choose a threshold with margin so it does not flake, and
say so in a comment rather than pretending it is deterministic.

## 8. Anti-patterns

- **Do not** return a raw `f64` mean without a stderr. Always
  `McEstimate<T>`.
- **Do not** apply antithetic to discontinuous payoffs (digital
  options, indicator functions). Variance can go up.
- **Do not** silently reuse one sample for both the control-variate
  coefficient and the estimate without saying so. The shipped
  `control_variates::estimate` does exactly that and accepts the
  `O(1/n)` bias; if you copy the shape, document it (§3). If you need
  an unbiased estimator, draw an independent pilot.
- **Do not** mix MC and QMC randomness in the same estimator without
  a randomised-QMC scheme. Plain Sobol mixed with a plain pseudo-random
  draw produces neither MC convergence nor QMC convergence.
- **Do not** use `StdRng` or `rand_distr::StandardNormal` in an
  estimator. `dev-rules` §7a reserves them for `benches/`; use
  `T::normal_array` or a seeded `Simd*` distribution.
- **Do not** skip the variance-reduction test (§7). It's the only
  check that the technique is wired the right way around.

## 9. Reference impls

The `mc/` module is nine files under
`stochastic-rs-stochastic/src/mc/`:

- `mod.rs` — `McEstimate<T>` and the module re-exports. Start here.
- `antithetic.rs` — `estimate` / `estimate_par`; the template for a new
  technique's signature.
- `control_variates.rs` — single control variate; see §3 on its
  coefficient estimation.
- `importance_sampling.rs` — mean-shift tilt, takes `shift: &Array1<T>`.
- `stratified.rs` — the only module with **seeded** entry points
  (`stratified_normals_1d_seeded`, `stratified_normals_seeded`).
- `sobol.rs`, `halton.rs` — in-tree low-discrepancy generators.
- `mlmc.rs` — `Mlmc::new(epsilon, l_min, l_max, n0)` +
  `estimate(level_sampler) -> MlmcResult<T>`; note it returns its own
  result type, not `McEstimate`.
- `lsm.rs` — Longstaff-Schwartz, `Lsm::new(r, tau, n_basis)` +
  `price(paths, payoff)`. **`#[cfg(feature = "openblas")]`-gated** —
  it needs LAPACK for the regression step, so it is absent from a
  default build.

## Related SKILLs

- `greeks-pattern` — uses CRN for the single-pass `greeks()` override.
- `add-diffusion-process` — produces the underlying paths; the seed is
  the constructor's last parameter, and CRN depends on `Clone`
  snapshotting it.
- `feature-flag-management` — `lsm.rs` is `openblas`-gated; a new
  LAPACK-dependent technique needs the same treatment.
- `bench-writing` — variance-reduction factor is a natural benchmark
  metric.
