---
name: adding-distribution
description: How to add a univariate distribution to stochastic-rs-distributions. Covers SimdXxx struct, sampling pattern (transformation / ziggurat / rejection / inversion), DistributionExt closed-form moments/pdf/cdf/cf, KS-test, and the py_distribution! macro.
---

# Adding distribution — stochastic-rs-distributions

Each distribution lives at `stochastic-rs-distributions/src/<name>.rs`
and ships a `SimdXxx<T>` struct that implements:

1. The `rand_distr::Distribution<T>` trait (per-sample `sample(rng)`).
2. A bulk filler `fill_slice(&self, out: &mut [T])` — no RNG argument;
   it advances the type's own internal stream.
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
| Transformation  | Closed-form `F^{-1}(U)` exists and is fast to evaluate. | `SimdExp` (`exp.rs`), `SimdLogNormal` |
| Ziggurat        | Density is unimodal & smooth; need throughput.          | `SimdNormal`, `SimdExpZig` (`exp.rs`) |
| Rejection       | Density has heavy tails or a kink; need correctness.     | `SimdGamma`, `SimdBinomial` (BTRS), `SimdTruncated*` |
| Subordination   | The law is a normal mean-variance mixture.               | `SimdNormalInverseGauss` (over `SimdInverseGauss`) |

Note the naming: the exponential is `SimdExp` / `SimdExpZig` in
`exp.rs`, not `SimdExponential`; the Normal-Inverse-Gaussian is
`SimdNormalInverseGauss` in `normal_inverse_gauss.rs`, not `SimdNig`.
There is no `SimdInverseGamma` and no `SimdCgmy` — CGMY exists in this
workspace as a *process* (`stochastic-rs-stochastic/src/jump/cgmy.rs`),
not as a distribution.

For tail-heavy laws the rejection step needs a documented acceptance
ratio in the source comments — the reviewer needs to verify that the
proposal density majorises the target.

## 2. Mandatory surface

```rust
// stochastic-rs-distributions/src/foo.rs

use crate::simd_rng::SeedExt;
use crate::simd_rng::SimdRng;
use crate::simd_rng::SimdRngExt;
use crate::traits::DistributionExt;
use crate::traits::FloatExt;

pub struct SimdFoo<T: SimdFloatExt, const N: usize = 64, R: SimdRngExt = SimdRng> {
    a: T, b: T,                        // distribution parameters
    buffer: UnsafeCell<[T; N]>,        // amortised sample buffer
    index: UnsafeCell<usize>,
    simd_rng: UnsafeCell<R>,           // the stream that actually gets used
}

impl<T: SimdFloatExt, const N: usize, R: SimdRngExt> SimdFoo<T, N, R> {
    /// Single canonical constructor. The seed source is the last argument,
    /// matching `Gbm::new(..., seed)` across the workspace: pass `&Unseeded`
    /// for an entropy-seeded stream, `&Deterministic::new(s)` for a
    /// reproducible one. There is no `with_seed` / `from_seed_source`.
    #[inline]
    pub fn new<S: SeedExt>(a: T, b: T, seed: &S) -> Self {
        assert!(N >= 8, "buffer size must be at least 8");
        Self {
            a, b,
            buffer: UnsafeCell::new([T::zero(); N]),
            index: UnsafeCell::new(N),
            simd_rng: UnsafeCell::new(seed.rng_ext::<R>()),
        }
    }

    /// Bulk fill. **The `_rng` argument is ignored** — it exists only so the
    /// type satisfies call sites that hand over an `Rng`. The samples come
    /// from `self.simd_rng`, seeded in `new`. Keep the underscore: it is the
    /// signal to readers that seeding an external RNG does nothing.
    /// Bulk fill. Takes **no** RNG argument — the type draws from its
    /// own internal stream, seeded at construction. There is no
    /// `fill_slice_fast` companion; this is the only bulk entry point,
    /// and every `Simd*` type in the crate has exactly this signature.
    pub fn fill_slice(&self, out: &mut [T]) { /* ... */ }
}

// `use rand_distr::Distribution;` — NOT `rand::distributions`, which
// does not exist on rand 0.9 (the workspace pins rand 0.9.2 /
// rand_distr 0.5.1). Per `dev-rules` §7a the trait import stays even
// though the concrete `rand_distr` distributions are banned from
// library code: our own `Simd*` types implement it, and it is how
// `.sample()` resolves.
impl<T: SimdFloatExt, const N: usize, R: SimdRngExt> Distribution<T> for SimdFoo<T, N, R> {
    fn sample<Rr: rand::Rng + ?Sized>(&self, _rng: &mut Rr) -> T {
        // Draws from the type's OWN internal stream, seeded at
        // construction — the `_rng` argument is ignored, exactly as in
        // `fill_slice` above. Underscore it so that is visible.
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

Every distribution file opens with a `//!` header carrying the LaTeX
for the pdf and/or characteristic function — see
`normal_inverse_gauss.rs`, which states `Nig(α, β, δ, μ)` and its
`ψ(u)` in the header before any code.

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
        // The seed is the constructor's last parameter, taken by
        // reference as `&impl SeedExt`. There is no `with_seed`.
        let d = SimdFoo::<f64>::new(2.0, 3.0, &Deterministic::new(42));
        let mut samples = vec![0.0; 100_000];
        d.fill_slice(&mut samples);
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
- **Do not** invent `with_seed` / `from_seed_source`. One constructor,
  `new(params.., seed: &S)`, where `S: SeedExt`.
- **Do not** name the ignored `fill_slice` RNG parameter `rng`. It must be
  `_rng`, or the next reader will believe seeding it has an effect — that
  misreading shipped a flaky `anderson_darling` test for months.
- **Do not** reach for `rand::rng()` or a concrete `rand_distr`
  distribution anywhere outside `benches/`. See `dev-rules` §7a.
- **Do not** skip the LaTeX `//!` header — the rust-docs need the
  formula for users skimming.

## 8a. When the distribution must be `Sync`

`Simd*` types are `!Sync` because of the `UnsafeCell` buffer, so they
cannot be handed to a process that requires
`D: Distribution<T> + Send + Sync` (the jump-size slot of
`CompoundPoisson`, `Bates1996`, `LevyDiffusion`, `JumpFOUCustom`). If the
new distribution is a plausible jump size, also add a stateless companion
in `scalar.rs`:

```rust
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScalarFoo<T> { a: T, b: T }

impl<T: FloatExt> Distribution<T> for ScalarFoo<T> {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> T {
        // inverse CDF or another closed form, drawn from the caller's rng
    }
}
```

Parameters only, no interior mutability — that is what makes it `Sync`.
`ScalarNormal` and `ScalarExp` are the reference impls.

## 9. Reference impls

- `SimdNormal` (`normal.rs`) — ziggurat; the canonical reference. Note
  its full generics: `SimdNormal<T: SimdFloatExt, const N: usize = 64,
  R: SimdRngExt = SimdRng>` — the const `N` is the internal buffer
  length, and most `Simd*` types carry the `R` parameter too.
- `ScalarNormal` / `ScalarExp` (`scalar.rs`) — stateless and `Sync`;
  the **only** types eligible for a process's `D: Distribution<T> +
  Send + Sync` jump slot, because `Simd*` types own an `UnsafeCell`
  buffer and are `!Sync`. See `dev-rules` §7a.
- `SimdExp` / `SimdExpZig` (`exp.rs`) — transformation and ziggurat
  variants of the same law, side by side.
- `SimdGamma` (`gamma.rs`) — rejection (Marsaglia-Tsang) with a
  transformation fallback for shape ≤ 1.
- `SimdNormalInverseGauss` (`normal_inverse_gauss.rs`) — subordination:
  draws an `SimdInverseGauss` mixing variable, then a `SimdNormal`.
  The reference for composing one distribution out of two.
- `SimdTruncatedNormal` / `Exp` / `Beta` / `Gamma` (`truncated.rs`) —
  four truncated laws in one file; the reference for rejection with a
  documented acceptance ratio.

## Related SKILLs

- `add-jump-process` — consumes a distribution as the jump-size
  parameter `D`.
- `python-bindings` — `py_distribution!` macro details.
- `stats-estimator` — for an MLE / MoM estimator that fits the
  distribution to data.
