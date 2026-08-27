---
name: copula-bivariate
description: How to add a bivariate copula to stochastic-rs-copulas. Invoke when implementing Clayton, Frank, Gumbel, Joe, Plackett, FGM-style families, or any 2-d Archimedean / extreme-value copula.
---

# Copula bivariate — stochastic-rs-copulas

Bivariate copulas live in `stochastic-rs-copulas/src/bivariate/<name>.rs`
and implement `BivariateExt`, defined in
`stochastic-rs-copulas/src/traits/bivariate.rs` and re-exported from
`stochastic-rs-copulas/src/traits.rs`. There are **13** families today
(`grep -c '^pub mod ' stochastic-rs-copulas/src/bivariate.rs`); the
14th `impl BivariateExt for` is a test double in
`traits/bivariate.rs`'s own test module.

Read `stochastic-rs-copulas/src/bivariate.rs`'s module header before
adding a family. It is current and it is the selection guide: which
Kendall's τ each family can represent, which are radially symmetric,
which pay an iterative-solver cost. This SKILL is the *mechanics*; that
header is the *taxonomy*.

(Note: `NCopula2DExt` was removed in v2.0 — bivariate samplers are all
consolidated under `BivariateExt`.)

## 1. What you must implement, and what you get free

`BivariateExt` is a wide trait with only **eleven** required methods.
Everything else — sampling, fitting, inversion, `log_pdf`, `ppf` — is
defaulted on top of them. Required:

```rust
fn r#type(&self) -> CopulaType;          // the family's enum discriminant
fn tau(&self) -> Option<f64>;            // Kendall's tau, None until fit/set
fn set_tau(&mut self, tau: f64);
fn theta(&self) -> Option<f64>;          // shape parameter, None until fit/set
fn set_theta(&mut self, theta: f64);
fn theta_bounds(&self) -> (f64, f64);
fn invalid_thetas(&self) -> Vec<f64>;    // returns owned Vec, not &[f64]
fn compute_theta(&self) -> f64;          // reads self.tau() — takes NO argument
fn tail_dependence(&self) -> TailDependence<f64>;
fn pdf(&self, X: &Array2<f64>) -> Result<Array1<f64>, Box<dyn Error>>;
fn cdf(&self, X: &Array2<f64>) -> Result<Array1<f64>, Box<dyn Error>>;
```

Two shape facts that catch every first attempt:

- **`pdf` / `cdf` are vectorised.** They take an `(n, 2)` `Array2<f64>`
  of `(u, v)` rows and return an `Array1<f64>` of length `n`. They are
  **not** scalar `(u: f64, v: f64) -> f64` functions.
- **`compute_theta` takes no `tau` argument.** It reads `self.tau()`,
  which `fit()` has already set. The tau → theta inversion is a method
  on a *populated* struct, not a free function of tau.

Useful defaults you should usually **not** override:

| Method | Default behaviour |
|---|---|
| `sample(n)` | `SimdUniform` + `Unseeded`, via `sample_with_uniform` |
| `sample_with_seed(n, seed)` | same, with `Deterministic::new(seed)` |
| `fit(X)` | Kendall's tau-b via `kendalls`, then `_compute_theta()` |
| `percent_point(y, V)` | Brent inversion of `partial_derivative_scalar` |
| `partial_derivative(X)` | central-difference on `cdf` |
| `generator(t)` | anchored "not Archimedean" error |
| `log_pdf` / `ppf` | `pdf(X)?.ln()` / alias for `percent_point` |
| `check_theta` / `check_fit` / `check_marginal` | validation, used by the above |

Override `percent_point` and `partial_derivative` only when you have a
closed form — Clayton, Frank, Gumbel, Joe all do, and the closed forms
are both faster and more accurate than the numeric defaults. Override
`generator` if and only if the family is Archimedean.

`tail_dependence` is **required, not defaulted, on purpose**: a silent
`(0.0, 0.0)` fallback would be a correctness bug for Clayton, Gumbel,
Joe, Galambos, Hüsler-Reiss, Marshall-Olkin and Student-t. It must call
`self.assert_theta_valid_for_tail_dependence()` first — it has no
`Result` to propagate through, so it panics on an out-of-domain theta
rather than returning a nonsensical coefficient.

## 2. The struct skeleton

Every family stores its bounds and its `CopulaType` as **fields**, and
carries `theta` / `tau` as `Option<f64>` (unset until `fit` or
`set_theta`). There is **no `seed` field** — seeding enters through
`sample_with_seed(n, seed)`, which builds a `Deterministic`-seeded
`SimdUniform` for that one call.

```rust
// stochastic-rs-copulas/src/bivariate/clayton.rs (reference)

use std::error::Error;
use std::f64;

use ndarray::Array1;
use ndarray::Array2;

use super::CopulaType;
use crate::traits::BivariateExt;
use crate::traits::TailDependence;

#[derive(Debug, Clone)]
pub struct Clayton {
  pub r#type: CopulaType,
  pub theta: Option<f64>,
  pub tau: Option<f64>,
  pub theta_bounds: (f64, f64),
  pub invalid_thetas: Vec<f64>,
}

impl Default for Clayton {
  fn default() -> Self {
    Self {
      r#type: CopulaType::Clayton,
      theta: None,
      tau: None,
      theta_bounds: (0.0, f64::INFINITY),
      invalid_thetas: vec![],
    }
  }
}

impl Clayton {
  pub fn new() -> Self {
    Self::default()
  }
}
```

Then the accessors are one-liners over those fields, and add your
family's discriminant to `CopulaType` plus a `pub mod` line in
`stochastic-rs-copulas/src/bivariate.rs`.

Note Clayton's real bounds: `(0.0, f64::INFINITY)` with
`invalid_thetas: vec![]`. The `θ ∈ (-1, ∞)` domain with a degenerate
point at 0 is the textbook Clayton, but this crate ships the
positive-dependence branch only, and encodes that by the bound rather
than by an invalid-theta entry.

## 3. The `compute_theta` pattern — closed-form > Brent > custom

The §6.12 audit trap was a custom Newton iteration on Frank's τ → θ map
that diverged on positive correlations. The mandate:

1. **Closed-form first.** Clayton: `θ = 2τ/(1-τ)`. Gumbel:
   `θ = 1/(1-τ)`. Cite the textbook formula in a doc comment.

2. **Brent's method second**, when no closed form exists. `Frank`'s
   shipped implementation is the pattern to copy:

   ```rust
   fn compute_theta(&self) -> f64 {
     let tau = self.tau.unwrap();

     if tau.abs() < 1e-12 { return 0.0; }
     if tau >= 1.0 { return f64::INFINITY; }
     if tau <= -1.0 { return f64::NEG_INFINITY; }

     let residual = |theta: f64| Self::_tau_to_theta(tau, theta);
     let mut convergency = SimpleConvergency { eps: 1e-8, max_iter: 100 };
     let (lo, hi) = if tau > 0.0 {
       (1e-8_f64, 50.0_f64)
     } else {
       (-50.0_f64, -1e-8_f64)
     };
     find_root_brent(lo, hi, residual, &mut convergency).unwrap_or(0.0)
   }
   ```

   Brent is bracketing → guaranteed convergence on a sign change.
   Newton is not. Note that Frank brackets *on the sign of tau*: a
   single `(-50, 50)` bracket straddles the θ = 0 root and finds it
   for every input.

3. **Never** roll a custom Newton / secant.

Handle the degenerate ends explicitly, as Frank and Clayton both do
(`τ ≥ 1 → ∞`). `compute_theta` returns a bare `f64` — it has no error
channel, so an unhandled edge shows up later as a `check_fit` failure
far from its cause.

## 4. Sampling — you get it for free, and it is conditional inversion

Do **not** write a `sample`. The default in `BivariateExt` already
implements the conditional-inversion (Rosenblatt) sampler:

```rust
// traits/bivariate.rs, defaulted — reproduced here for orientation only
let mut v = Array1::<f64>::zeros(n);
ud.fill_slice(v.as_slice_mut().unwrap());
let mut c = Array1::<f64>::zeros(n);
ud.fill_slice(c.as_slice_mut().unwrap());
let u = self.percent_point(&c, &v)?;
Ok(stack![Axis(1), u, v])
```

It errors if `tau` is unset or outside `(-1, 1)`, draws both uniforms
from one `SimdUniform<f64>`, and returns an `(n, 2)` `Array2<f64>`
wrapped in `Result`. `sample(&self, ..)` takes `&self`, not `&mut self`.

What you supply is `percent_point` — either a closed form, or nothing
at all, in which case `percent_point_numerical` Brent-inverts your
`partial_derivative` for you. If you override `percent_point` but want
the generic path for a degenerate branch, call
`self.percent_point_numerical(y, V)` — calling `Self::percent_point`
from inside the override just recurses.

Per `dev-rules` §7a, `StdRng` / `rand_distr::Uniform` are **not**
available here: library code draws from the workspace's own
`SimdUniform`, seeded via `Deterministic::new(s)` or `Unseeded`.

## 5. Mandatory tests

Each family's test module in this crate carries, at minimum, a
closed-form tail-dependence check and independence-point checks. Copy
`clayton.rs`'s set:

```rust
#[test]
fn clayton_tail_dependence_closed_form() { /* λ_L = 2^{-1/θ}, λ_U = 0 */ }

#[test]
fn clayton_pdf_at_independence_is_one() { /* c(u,v) = 1 at θ → indep */ }

#[test]
fn clayton_cdf_at_independence_is_uv() { }

#[test]
fn clayton_partial_derivative_at_independence_returns_u() { }
```

Plus the parametric-τ recovery test, which is what catches the §6.12
class — a wrong `compute_theta` looks fine on the data side but
produces samples with a different τ than requested:

```rust
#[test]
fn clayton_independence_sample_kendall_tau_near_zero() {
  let mut cop = Clayton::new();
  cop.set_tau(target_tau);
  cop._compute_theta();
  let samples = cop.sample_with_seed(50_000, 42).unwrap();
  // empirical Kendall's tau on `samples` within tolerance of target_tau
}
```

**Pin the seed** via `sample_with_seed` — never plain `sample()`, which
is `Unseeded` and reseeds from entropy each run. See
`integration-test-writing` for the full pinned-seed mandate.

## 6. Anti-patterns

- **Do not** give the struct a `seed` field. Seeding is per-call, via
  `sample_with_seed`.
- **Do not** write scalar `pdf(u, v)` / `cdf(u, v)`. The trait is
  vectorised over an `(n, 2)` array and returns `Result`.
- **Do not** give `compute_theta` a `tau` parameter. It reads
  `self.tau()`.
- **Do not** roll a custom Newton in `compute_theta`. Use Brent.
- **Do not** return `Vec::new()`-shaped nonsense from
  `tail_dependence`, and do not skip
  `assert_theta_valid_for_tail_dependence()`. `_compute_theta` discards
  its own `check_theta()` result, so `fit()` on out-of-domain data
  silently leaves `theta` out of bounds; that assert is the only guard.
- **Do not** implement `sample` / `fit` / `log_pdf` / `ppf`. They are
  defaulted and the defaults are correct.
- **Do not** implement `generator` for a non-Archimedean family. The
  default already returns the anchored
  `"<Type> is not Archimedean — generator not defined"` error.

## 7. Reference impls

- `Clayton` (`bivariate/clayton.rs`) — closed-form `compute_theta`,
  closed-form `percent_point` / `partial_derivative`, Archimedean
  `generator`. The template.
- `Frank` (`bivariate/frank.rs`) — Brent-based `compute_theta` with
  sign-of-tau bracketing.
- `Gumbel` (`bivariate/gumbel.rs`) — closed-form via Archimedean
  generator; shares Joe's upper-tail formula.
- `Fgm` (`bivariate/fgm.rs`) — Farlie-Gumbel-Morgenstern, narrow
  τ ∈ [-0.22, 0.22].
- `MarshallOlkin` (`bivariate/marshall_olkin.rs`) — the awkward case:
  a `with_alpha_beta` constructor that fits two shock rates directly
  rather than inverting tau, and the only non-exchangeable family here.
  Read it before assuming every family fits the `theta` mould.

## Related SKILLs

- `integration-test-writing` — the pinned-seed mandate these tests obey.
- `add-mc-variance-reduction` — when copula sampling is part of an MC
  pricer using common random numbers.
- `python-bindings` — for the `PyClayton` etc. wrappers.
