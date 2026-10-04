---
name: adding-distribution
description: How to add a univariate distribution to stochastic-rs-distributions. Covers SimdXxx struct, sampling pattern (transformation / ziggurat / rejection / inversion), DistributionExt closed-form moments/pdf/cdf/cf, KS-test, and the py_distribution! macro.
---

# Adding distribution — stochastic-rs-distributions

Each distribution lives at `stochastic-rs-distributions/src/<name>.rs`
and ships a stateless `SimdXxx<T>` law (parameters only) that implements:

1. `rand::distr::Distribution<T>` — one honest scalar draw from the
   caller's rng, through `draw_with`.
2. The sealed `SimdDistribution` + `SimdKernel` pair — the stream state
   and the SIMD kernel that a `Seeded<SimdXxx<T>>` drives (`sample`,
   `fill_slice`, `sample_n`, `sample_matrix`, `fork`, all `&mut self`).
3. `DistributionExt` for closed-form pdf / cdf / characteristic
   function / moments.
4. The `py_distribution!` macro at the bottom for Python exposure.

The `feedback_no_statrs_distributions` memory entry and the `Option`
contract of `DistributionExt` are the load-bearing constraints:
closed-form math, written from scratch in this crate, never
`statrs::distribution::*`.

## 1. Pick a sampling strategy

Three patterns, in order of preference:

| Pattern         | When to use                                             | Reference impl |
|-----------------|---------------------------------------------------------|----------------|
| Transformation  | Closed-form `F^{-1}(U)` exists and is fast to evaluate. | `SimdPareto` (`pareto.rs`), `SimdLogNormal` |
| Ziggurat        | Density is unimodal & smooth; need throughput.          | `SimdNormal`, `SimdExp` (`exp.rs`) |
| Rejection       | Density has heavy tails or a kink; need correctness.     | `SimdGamma`, `SimdBinomial` (BTRS), `SimdTruncated*` |
| Subordination   | The law is a normal mean-variance mixture.               | `SimdNormalInverseGauss` (over `SimdInverseGauss`) |

Note the naming: the exponential is `SimdExp` in
`exp.rs`, not `SimdExponential`; the Normal-Inverse-Gaussian is
`SimdNormalInverseGauss` in `normal_inverse_gauss.rs`, not `SimdNig`.
There is no `SimdInverseGamma` and no `SimdCgmy` — CGMY exists in this
workspace as a *process* (`stochastic-rs-stochastic/src/jump/cgmy.rs`),
not as a distribution.

For tail-heavy laws the rejection step needs a documented acceptance
ratio in the source comments — the reviewer needs to verify that the
proposal density majorises the target.

## 2. Mandatory surface

A new law implements three things: `SimdDistribution::init` (the seed
draws and the stream state, in a fixed order), `SimdKernel::{fill, next}`
(the SIMD kernel and the buffered single draw), and `draw_with` (the
scalar algorithm behind the honest `Distribution::sample`). `Seeded`
supplies everything else — no `UnsafeCell`, no `Cell`, no seed argument
on the constructor.

```rust
// stochastic-rs-distributions/src/foo.rs

use rand::Rng;
use rand::distr::Distribution;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::SimdRngExt;

use crate::seeded::Buffered;
use crate::seeded::StreamState;
use crate::traits::SimdFloatExt;
use crate::traits::distribution::Sealed;
use crate::traits::distribution::SimdDistribution;
use crate::traits::distribution::SimdKernel;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SimdFoo<T> {
    a: T,
    b: T,
}

impl<T: SimdFloatExt> SimdFoo<T> {
    pub fn new(a: T, b: T) -> Self {
        assert!(b > T::zero(), "b must satisfy `b > T::zero()`, got b = {b:?}");
        Self { a, b }
    }

    pub fn a(&self) -> T { self.a }

    pub fn b(&self) -> T { self.b }

    /// The scalar algorithm on any `rand` rng; composites reuse it.
    pub(crate) fn draw_with<G: Rng + ?Sized>(&self, rng: &mut G) -> T { /* ... */ }

    fn fill_parts<R: SimdRngExt>(&self, rng: &mut R, out: &mut [T]) { /* the SIMD kernel */ }
}

impl<T: SimdFloatExt> Sealed for SimdFoo<T> {}

impl<T: SimdFloatExt> SimdDistribution for SimdFoo<T> {
    type State<R: SimdRngExt> = StreamState<T, R, 16>;

    fn init<R: SimdRngExt, S: SeedExt>(&self, seed: &S) -> (StreamState<T, R, 16>, u64) {
        let s = seed.next_seed();
        (StreamState { rng: R::from_seed(s), buf: Buffered::new() }, s)
    }
}

impl<T: SimdFloatExt> SimdKernel for SimdFoo<T> {
    type Item = T;

    fn fill<R: SimdRngExt>(&self, st: &mut StreamState<T, R, 16>, out: &mut [T]) {
        self.fill_parts(&mut st.rng, out);
    }

    fn next<R: SimdRngExt>(&self, st: &mut StreamState<T, R, 16>) -> T {
        let StreamState { rng, buf } = st;
        buf.pop(|b| self.fill_parts(rng, b))
    }
}

// `use rand::distr::Distribution;` — not `rand_distr`, a dev-dependency (`dev-rules` §7a).
impl<T: SimdFloatExt> Distribution<T> for SimdFoo<T> {
    fn sample<G: Rng + ?Sized>(&self, rng: &mut G) -> T {
        self.draw_with(rng)
    }
}
```

A composite law holds its components as stateless values and its
`State` as their states (`StudentTState { normal, chisq, buf }`); its
kernel calls the components' `next` / `fill` on those states, so a
sub-stream keeps its draw order. The slow paths of a ziggurat draw
through `crate::source::Source`, which serves both a SIMD engine and
any `rand` rng (`AnyRng`), so the scalar and bulk paths share one
algorithm.

## 3. DistributionExt — closed-form math

```rust
impl<T: SimdFloatExt> crate::traits::DistributionExt for SimdFoo<T> {
  /// Every method returns `Option<f64>` (the trait is `f64`-only); `Some` where the law has a
  /// closed form, and no override — the default `None` — where it has none. Never `Some(0.0)` for a gap.
  fn pdf(&self, x: f64) -> Option<f64> { /* derive from scratch, special functions from `crate::special` */ }

  fn cdf(&self, x: f64) -> Option<f64> { /* erf / regularised incomplete gamma or beta from `crate::special` */ }

  /// Characteristic function from the canonical paper (NIG: Barndorff-Nielsen 1997 eq. 3; CGMY: Carr-Geman-Madan-Yor 2002 eq. 3.4).
  fn characteristic_function(&self, t: f64) -> Option<num_complex::Complex64> { /* … */ }

  fn quantile(&self, p: f64) -> Option<f64> { /* the inverse cdf, or leave the default when there is none */ }

  fn mean(&self) -> Option<f64> { /* … */ }

  fn variance(&self) -> Option<f64> { /* … */ }
}
```

A method with no closed form is simply not overridden. `tests/distribution_ext_coverage.rs` lists, per type, which cells are `Some`: add the new type's row there with the same reference point, and a row to `tests/distribution_ext_vs_reference.rs` or `tests/distribution_ext_closed_forms.rs` with values from an independent source (`statrs` where it has the law, `mpmath` otherwise).

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

Three mandatory tests:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::stats::ks_test;

    /// 1. Kolmogorov-Smirnov test against the analytical CDF.
    #[test]
    fn ks_test_passes() {
        // The stream draws; the stateless law answers `cdf`.
        let d = SimdFoo::<f64>::new(2.0, 3.0);
        let mut stream = SimdFoo::<f64>::new(2.0, 3.0).seeded(&Deterministic::new(42));
        let mut samples = vec![0.0; 100_000];
        stream.fill_slice(&mut samples);
        let p = ks_test(&samples, |x| d.cdf(x));
        assert!(p > 0.05, "KS p-value = {p}");
    }

    /// 2. Mean / variance via fill_slice match closed-form mean()/variance().
    #[test]
    fn moments_match_closed_form() { ... }

    /// 3. The honest `d.sample(&mut SimdRng::from_seed(s))` passes KS too (best of three seeds).
    #[test]
    fn scalar_sample_matches_cdf() { ... }
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

There is exactly **one** `CLAUDE.md` in this repo, at the workspace
root — there are no per-crate `CLAUDE.md` files, so do not look for
`stochastic-rs-distributions/CLAUDE.md`.

- The root `CLAUDE.md`'s workspace layout does not enumerate individual
  distributions, so a new one usually needs no edit there. Update it
  only if you change something it does state — e.g. the
  `DistributionExt` bullet's counts,
  or the `stochastic-rs-py` entry count if you add a Python binding.
- Distributions are **not** in the prelude individually; users reach
  them at `stochastic_rs::distributions::foo::SimdFoo`. Only a new
  *trait* touches `src/traits.rs` and the prelude.

## 8. Anti-patterns

- **Do not** import `statrs::distribution::*`. The
  `feedback_no_statrs_distributions` memory entry is explicit.
- **Do not** return `Some(0.0)` for a moment the law has no closed form for; leave the default `None`, and never `unimplemented!()` — a panic in a trait method is not an error channel.
- **Do not** put a seed or an engine on the law. One constructor,
  `new(params..)`, parameters only; `.seeded(&seed)` binds the stream.
- **Do not** ignore the rng `Distribution::sample` receives: it is the
  honest scalar draw (`draw_with`); the buffered stream is `Seeded::sample`.
- **Do not** reach for `rand::rng()` or a concrete `rand_distr`
  distribution anywhere outside `benches/`. See `dev-rules` §7a.
- **Do not** skip the LaTeX `//!` header — the rust-docs need the
  formula for users skimming.

## 8a. Jump-size laws

A process's jump-size slot (`D: Distribution<T> + Send + Sync`: `CompoundPoisson`, `Bates1996`,
`LevyDiffusion`, `JumpFOUCustom`, …) takes a scalar continuous `Simd*` law itself: it holds parameters only, so it
is `Send + Sync`, and `draw_with` gives it the honest `Distribution::sample(&mut rng)`. `SimdNormal` is the
reference impl; the device engine recognises `SimdNormal` and `SimdExp`.

## 9. Reference impls

- `SimdNormal` (`normal.rs`) — ziggurat; the canonical reference for
  the stateless shape: `SimdNormal<T>` holds `mean` / `std_dev` only,
  its `State<R>` is a `StreamState<T, R, 64>` (the 64-wide buffer lives
  in the stream), and the engine is chosen on `Seeded<SimdNormal<T>, R>`.
- `SimdExp` (`exp.rs`) — the exponential ziggurat, its state a
  `StreamState<T, R, 64>` like `SimdNormal`'s.
- `SimdGamma` (`gamma.rs`) — rejection (Marsaglia-Tsang) with a
  transformation fallback for shape ≤ 1.
- `SimdNormalInverseGauss` (`normal_inverse_gauss.rs`) — subordination:
  draws an `SimdInverseGauss` mixing variable, then a `SimdNormal`.
  The reference for composing one distribution out of two.
- `SimdTruncatedNormal` / `Exp` / `Beta` / `Gamma` (`truncated/`) —
  four truncated laws behind the `truncated.rs` hub; the reference for
  rejection with a documented acceptance ratio.

## Related SKILLs

- `add-jump-process` — consumes a distribution as the jump-size
  parameter `D`.
- `python-bindings` — `py_distribution!` macro details.
- `stats-estimator` — for an MLE / MoM estimator that fits the
  distribution to data.
