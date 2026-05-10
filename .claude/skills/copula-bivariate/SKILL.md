---
name: copula-bivariate
description: How to add a bivariate copula to stochastic-rs-copulas. Invoke when implementing Clayton, Frank, Gumbel, Joe, Plackett, FGM-style families, or any 2-d Archimedean / extreme-value copula.
---

# Copula bivariate — stochastic-rs-copulas

Bivariate copulas in `stochastic-rs-copulas` implement the
`BivariateExt` trait, exposing pdf / cdf / inverse / partial
derivative / parametric tau / sampling. The §6.12 audit fix-pass
(rc.0) caught a Frank-tau formula bug where a custom Newton iteration
diverged on positive correlations; the rc.1 fix routed everything
through `roots::find_root_brent`, which is the canonical pattern.

This SKILL codifies the trait surface, the bounds / invalid-θ
contract, the `compute_theta` pattern (closed-form when possible,
Brent root-find otherwise), and the parametric-tau-recovery test that
prevents the §6.12-class regression.

(Note: `NCopula2DExt` was removed in v2.0 — bivariate samplers are
all consolidated under `BivariateExt`.)

## 1. The trait surface

```rust
// stochastic-rs-copulas/src/traits.rs

pub trait BivariateExt {
    /// Parameter bounds for valid copulas. Used by calibrators.
    fn theta_bounds(&self) -> (f64, f64);
    /// Discrete θ values that produce a degenerate copula (e.g. θ = 0
    /// for Clayton produces independence; the sampler may need to
    /// branch around them).
    fn invalid_thetas(&self) -> &[f64];

    /// Inverse Kendall's τ: given a target Kendall correlation, return
    /// the θ that achieves it. Closed-form when possible (Clayton,
    /// Gumbel) — otherwise Brent root-find on the relationship τ(θ).
    fn compute_theta(&self, tau: f64) -> f64;

    /// Joint CDF C(u, v).
    fn cdf(&self, u: f64, v: f64) -> f64;
    /// Joint PDF c(u, v) = ∂²C/(∂u ∂v).
    fn pdf(&self, u: f64, v: f64) -> f64;
    /// Inverse conditional: u_given_v(p, v) such that
    /// P(U ≤ u | V = v) = p — used for sampling via Rosenblatt.
    fn percent_point(&self, p: f64, v: f64) -> f64;
    /// Partial derivative ∂C/∂u — used for conditional sampling.
    fn partial_derivative(&self, u: f64, v: f64) -> f64;
    /// Sample (u, v) pairs.
    fn sample(&self, n: usize) -> ndarray::Array2<f64>;
}
```

## 2. The struct skeleton

```rust
// stochastic-rs-copulas/src/clayton.rs (reference)

use roots::SimpleConvergency;
use roots::find_root_brent;

pub struct Clayton {
    pub theta: f64,
    pub seed: u64,
}

impl BivariateExt for Clayton {
    fn theta_bounds(&self) -> (f64, f64) {
        (-1.0, f64::INFINITY)        // Clayton: θ ∈ (-1, ∞), θ = 0 → indep
    }

    fn invalid_thetas(&self) -> &[f64] {
        &[0.0]                       // Clayton degenerate at θ = 0
    }

    fn compute_theta(&self, tau: f64) -> f64 {
        // Clayton has closed-form: τ = θ / (θ + 2)  →  θ = 2τ / (1 - τ)
        2.0 * tau / (1.0 - tau)
    }

    fn cdf(&self, u: f64, v: f64) -> f64 {
        // C(u, v) = (u^{-θ} + v^{-θ} - 1)^{-1/θ}
        let θ = self.theta;
        (u.powf(-θ) + v.powf(-θ) - 1.0).powf(-1.0 / θ)
    }

    // ... pdf / percent_point / partial_derivative / sample ...
}
```

## 3. The `compute_theta` pattern — closed-form > Brent > custom

The §6.12 trap was a custom Newton iteration on Frank's
`τ → θ` map that diverged on positive correlations because the
secant initialisation hit a flat region. The mandate:

1. **Closed-form first.** If τ(θ) inverts analytically (Clayton, Gumbel,
   FGM), use it. Cite the textbook formula in a comment.

2. **Brent's method second.** When no closed form exists, use
   `roots::find_root_brent`:

   ```rust
   fn compute_theta(&self, tau: f64) -> f64 {
       let f = |theta: f64| -> f64 { tau_from_theta(theta) - tau };
       let bounds = self.theta_bounds();
       let lo = bounds.0.max(-50.0).max(self.theta_bounds().0 + 1e-6);
       let hi = bounds.1.min( 50.0).min(self.theta_bounds().1 - 1e-6);
       let mut conv = SimpleConvergency { eps: 1e-12, max_iter: 100 };
       find_root_brent(lo, hi, &f, &mut conv).unwrap_or(0.0)
   }
   ```

   Brent is bracketing → guaranteed convergence on a sign-change. Newton
   isn't.

3. **Never** roll a custom Newton / secant. The §6.12 trap shipped
   exactly this: a custom hand-rolled iteration with no convergence
   proof.

## 4. Sampling — Rosenblatt transform

The standard 2-d copula sampler:

```rust
fn sample(&self, n: usize) -> ndarray::Array2<f64> {
    let mut rng = StdRng::seed_from_u64(self.seed);
    let unif = Uniform::new_inclusive(0.0, 1.0);
    let mut out = ndarray::Array2::<f64>::zeros((n, 2));
    for i in 0..n {
        let v = unif.sample(&mut rng);
        let p = unif.sample(&mut rng);
        let u = self.percent_point(p, v);  // P(U ≤ u | V = v) = p
        out[[i, 0]] = u;
        out[[i, 1]] = v;
    }
    out
}
```

The `percent_point(p, v)` step is where the inversion happens. For
copulas where this isn't closed-form (FGM-class, Joe), invert
`partial_derivative(u, v) = p` w.r.t. `u` via Brent again.

## 5. Mandatory test: parametric-τ recovery

```rust
#[test]
fn parametric_tau_recovery() {
    let target_tau = 0.5;
    let cop = Clayton {
        theta: Clayton { theta: 0.0, seed: 0 }.compute_theta(target_tau),
        seed: 42,
    };
    let samples = cop.sample(50_000);
    let empirical_tau = compute_kendall_tau(&samples);
    assert!(
        (empirical_tau - target_tau).abs() < 0.02,
        "τ recovery: target {target_tau}, got {empirical_tau}"
    );
}
```

This is the regression test that catches the §6.12 class. Without it,
a wrong `compute_theta` looks fine on the data side but produces
samples with a different τ than requested. **Pin** the seed, **pin** a
50_000-sample bound on the empirical-τ noise.

## 6. Anti-patterns

- **Do not** roll a custom Newton in `compute_theta`. Use Brent.
- **Do not** silently return `0.0` from `compute_theta` on
  out-of-domain τ. Validate bounds at construction or in
  `compute_theta` and panic with a useful message.
- **Do not** branch on `if theta == 0.0` for the degenerate case in the
  hot loop. Pre-check at construction; the sampler should never see
  `theta = 0` for Clayton.
- **Do not** sample without a seed. `seed: u64` field is mandatory for
  reproducibility.

## 7. Reference impls

- `Clayton` (`clayton.rs`) — closed-form `compute_theta`, Rosenblatt
  sampling.
- `Frank` (`frank.rs`) — Brent-based `compute_theta` (rc.1 fix; was
  custom Newton).
- `Gumbel` (`gumbel.rs`) — closed-form via Archimedean generator.
- `FGM` (`fgm.rs`) — Farlie-Gumbel-Morgenstern; bounded τ ∈ [-2/9, 2/9].

## Related SKILLs

- `add-mc-variance-reduction` — when copula sampling is part of an MC
  pricer using common random numbers.
- `python-bindings` — for the `PyClayton` etc. wrappers.
