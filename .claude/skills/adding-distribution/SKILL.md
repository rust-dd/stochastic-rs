---
name: adding-distribution
description: How to add a univariate distribution to stochastic-rs-distributions. Covers SimdXxx struct, sampling pattern (transformation / ziggurat / rejection / inversion), DistributionExt closed-form moments/pdf/cdf/cf, KS-test, and the py_distribution! macro.
---

# Adding distribution — stochastic-rs-distributions

Each distribution lives at `stochastic-rs-distributions/src/<name>.rs`
and ships a `SimdXxx<T>` struct that implements:

1. The `rand_distr::Distribution<T>` trait (per-sample `sample(rng)`).
2. Bulk fillers `fill_slice(rng, dst)` and `fill_slice_fast(dst)` (with
   internal RNG seed advancement).
3. `DistributionExt` for closed-form pdf / cdf / characteristic
   function / moments.
4. The `py_distribution!` macro at the bottom for Python exposure.

The §1.5 audit note "DistributionExt is 18/19 closed-form (not 3/19)"
plus the `feedback_no_statrs_distributions` memory entry are the
load-bearing constraints: closed-form math, written from scratch in
this crate, never `statrs::distribution::*`.

## 1. Pick a sampling strategy

Three patterns, in order of preference:

| Pattern         | When to use                                             | Reference impl |
|-----------------|---------------------------------------------------------|----------------|
| Transformation  | Closed-form `F^{-1}(U)` exists and is fast to evaluate. | `SimdExponential`, `SimdLogNormal` |
| Ziggurat        | Density is unimodal & smooth; need throughput.          | `SimdNormal`, `SimdGamma` |
| Rejection / inversion | Density has heavy tails or kink; need correctness.      | `SimdInverseGamma`, `SimdNig` |

For tail-heavy distributions (NIG, Variance-Gamma, CGMY), the rejection
step needs a documented acceptance ratio in the source comments — the
reviewer needs to verify that the proposal density majorises the target.

## 2. Mandatory surface

```rust
// stochastic-rs-distributions/src/foo.rs

use crate::traits::FloatExt;
use crate::traits::DistributionExt;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

pub struct SimdFoo<T: FloatExt> {
    pub a: T, pub b: T,            // distribution parameters (named, not args)
    rng: std::cell::RefCell<Xoshiro256PlusPlus>,  // seeded RNG (interior mutability)
}

impl<T: FloatExt> SimdFoo<T> {
    /// Construct with a thread-local RNG seed (default).
    pub fn new(a: T, b: T) -> Self {
        let seed = rand::random::<u64>();
        Self::with_seed(a, b, seed)
    }

    /// Construct with an explicit seed. **Mandatory** — every distribution
    /// has both `new` and `with_seed`.
    pub fn with_seed(a: T, b: T, seed: u64) -> Self {
        Self {
            a, b,
            rng: std::cell::RefCell::new(Xoshiro256PlusPlus::seed_from_u64(seed)),
        }
    }

    /// Construct from an existing seedable source (e.g. for chained streams).
    pub fn from_seed_source<R: rand::SeedableRng>(a: T, b: T, src: R) -> Self {
        let mut bytes = [0u8; 8];
        src.fill_bytes(&mut bytes);
        let seed = u64::from_le_bytes(bytes);
        Self::with_seed(a, b, seed)
    }

    /// Bulk fill with parallel SIMD; respects internal seed.
    pub fn fill_slice(&self, dst: &mut [T]) { /* ... */ }

    /// Bulk fill optimised for speed; may use different SIMD intrinsics.
    pub fn fill_slice_fast(&self, dst: &mut [T]) { /* ... */ }
}

impl<T: FloatExt> rand::distributions::Distribution<T> for SimdFoo<T> {
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> T {
        // ... your sampling step ...
    }
}
```

## 3. DistributionExt — closed-form math

```rust
impl<T: FloatExt> DistributionExt<T> for SimdFoo<T> {
    /// PDF f(x). MUST be closed-form. Use special functions from
    /// `crate::special::*` — never `statrs::distribution::*`.
    fn pdf(&self, x: T) -> T { /* derive from scratch */ }

    /// CDF F(x). MUST be closed-form (or an erf / regularised
    /// incomplete gamma call from `crate::special`).
    fn cdf(&self, x: T) -> T { /* ... */ }

    /// Characteristic function φ(u) = E[exp(i u X)]. MUST be derived
    /// from the canonical reference paper — NIG: Barndorff-Nielsen 1997
    /// eq. 3; CGMY: Carr-Geman-Madan-Yor 2002 eq. 3.4; etc.
    fn cf(&self, u: T) -> num_complex::Complex<T> { /* ... */ }

    /// Moments. Provide as many as the literature gives in closed form;
    /// mark unimplemented ones with `unimplemented!("not implemented for {}", type_name)`,
    /// NEVER return 0.0 (that hides the gap).
    fn mean(&self) -> T { /* ... */ }
    fn variance(&self) -> T { /* ... */ }
    fn skewness(&self) -> T { unimplemented!("skewness not implemented for SimdFoo") }
    fn kurtosis(&self) -> T { unimplemented!("kurtosis not implemented for SimdFoo") }
}
```

The 5 currently-unimplemented `unimplemented!` distributions (per the
`project_distribution_ext_status` memory) are intentional: where the
literature has no closed form (e.g. NIG raw moments require Bessel-K
identities), the panic is a documentation device — users should use
empirical moments via `crate::estimators::*`.

## 4. Source-file documentation

The `//!` header MUST include:

```rust
//! # SimdFoo distribution
//!
//! \[LaTeX block — pdf and/or cf\]
//!
//! Reference: <Author, Year>, "<Title>", <Journal>, eq. <number>.
```

Example: `SimdNig` cites Barndorff-Nielsen (1997) eq. 3; `SimdCgmy`
cites Carr-Geman-Madan-Yor (2002) eq. 3.4.

## 5. Testing — KS test + reference comparison

Two mandatory tests:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::stats::ks_test;

    /// 1. Kolmogorov-Smirnov test against the analytical CDF.
    #[test]
    fn ks_test_passes() {
        let d = SimdFoo::<f64>::with_seed(2.0, 3.0, 42);
        let mut samples = vec![0.0; 100_000];
        d.fill_slice_fast(&mut samples);
        let p = ks_test(&samples, |x| d.cdf(x));
        assert!(p > 0.05, "KS p-value = {p}");
    }

    /// 2. Mean / variance via fill_slice match closed-form mean()/variance().
    #[test]
    fn moments_match_closed_form() { ... }
}
```

Plus the workspace-level `distribution_ext_vs_reference` integration
test (in `stochastic-rs-distributions/tests/`) — add a row for the new
distribution comparing pdf/cdf/cf at fixed reference points to a
manually-computed Mathematica/scipy table.

## 6. Python wrapper — `py_distribution!`

Append at the bottom of `src/foo.rs`:

```rust
py_distribution!(PyFoo, SimdFoo,
    sig: (a, b, seed = None, dtype = None),
    params: (a: f64, b: f64),
);
```

The macro generates `PyFoo`, `__new__`, `sample(n)`, `sample_par(m, n)`,
all routed through the `IntoF32` / `IntoF64` shims. Then in
`stochastic-rs-py/src/lib.rs`:

```rust
use stochastic_rs_distributions::foo::PyFoo;
m.add_class::<PyFoo>()?;
```

## 7. CLAUDE.md / prelude updates

- `stochastic-rs-distributions/CLAUDE.md` — list the new distribution.
- The umbrella `CLAUDE.md` workspace layout doesn't list individual
  distributions; only update if the count crosses a notable boundary.

## 8. Anti-patterns

- **Do not** import `statrs::distribution::*`. The
  `feedback_no_statrs_distributions` memory entry is explicit.
- **Do not** return `0.0` from unimplemented moments. Use
  `unimplemented!("...")` so callers fail loudly.
- **Do not** sample without a seeded path. Both `new` (thread-local
  seed) and `with_seed` (explicit) are mandatory.
- **Do not** skip the LaTeX `//!` header — the rust-docs need the
  formula for users skimming.

## 9. Reference impls

- `SimdNormal` (`normal.rs`) — ziggurat; the canonical reference.
- `SimdExponential` (`exponential.rs`) — transformation; thinnest possible.
- `SimdGamma` (`gamma.rs`) — ziggurat with shape > 1, transformation
  fallback for shape ≤ 1.
- `SimdNig` (`nig.rs`) — rejection; tail-heavy; Bessel-K-based pdf.
- `SimdCgmy` (`cgmy.rs`) — Lévy density with rejection; CGMY 2002.

## Related SKILLs

- `add-jump-process` — consumes a distribution as the jump-size
  parameter `D`.
- `python-bindings` — `py_distribution!` macro details.
- `stats-estimator` — for an MLE / MoM estimator that fits the
  distribution to data.
