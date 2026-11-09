---
name: copula-bivariate
description: How to add a bivariate copula to stochastic-rs-copulas. Invoke when implementing Clayton, Frank, Gumbel, Joe, Plackett, FGM-style families, or any 2-d Archimedean / extreme-value copula.
---

# Copula bivariate — stochastic-rs-copulas

Bivariate copulas live in `stochastic-rs-copulas/src/bivariate/<name>.rs`
and implement `BivariateExt`, defined in
`stochastic-rs-copulas/src/traits/bivariate.rs` and re-exported from
`stochastic-rs-copulas/src/traits.rs`. There are **15** families today
(`grep -c '^pub mod ' stochastic-rs-copulas/src/bivariate.rs`); the four
other `impl BivariateExt for` are test doubles: two in
`traits/bivariate/tests.rs`, `Poisonable` in `gof.rs`'s tests and
`Careless` in `tests/fit_domain.rs`.

Read `stochastic-rs-copulas/src/bivariate.rs`'s module header before
adding a family. It is current and it is the selection guide: which
Kendall's τ each family can represent, which are radially symmetric,
which pay an iterative-solver cost. This SKILL is the *mechanics*; that
header is the *taxonomy*.

(Note: `NCopula2DExt` was removed in v2.0 — bivariate samplers are all
consolidated under `BivariateExt`.)

## 1. What you must implement, and what you get free

`BivariateExt` is a wide trait with only **eleven** required methods.
Everything else — sampling, fitting, inversion, `log_pdf` — is
defaulted on top of them. Required:

```rust
fn family(&self) -> BivariateFamily;     // the family's enum discriminant
fn theta(&self) -> Option<f64>;          // shape parameter, None until fit/set
fn set_theta(&mut self, theta: f64);     // panics unless validate_theta admits theta
fn theta_bounds(&self) -> (f64, f64);
fn invalid_thetas(&self) -> Vec<f64>;    // returns owned Vec, not &[f64]
fn tau_from_theta(&self, theta: f64) -> f64;                      // Kendall's τ of the family at θ
fn theta_from_tau(&self, tau: f64) -> Result<f64, CopulaError>;   // Err outside the reachable τ range
fn tail_dependence(&self) -> Result<TailDependence<f64>, CopulaError>;
fn pdf(&self, x: ArrayView2<f64>) -> Result<Array1<f64>, CopulaError>;
fn cdf(&self, x: ArrayView2<f64>) -> Result<Array1<f64>, CopulaError>;
fn partial_derivative(&self, x: ArrayView2<f64>) -> Result<Array1<f64>, CopulaError>; // h(u | v) = ∂_v C
```

Two shape facts that catch every first attempt:

- **`pdf` / `cdf` are vectorised.** They take an `(n, 2)` `ArrayView2<f64>`
  of `(u, v)` rows (an owned array passes `.view()`) and return an
  `Array1<f64>` of length `n`. They are **not** scalar
  `(u: f64, v: f64) -> f64` functions.
- **`theta_from_tau` returns `Err(InvalidParameter { name: "tau", .. })`
  for a τ the family cannot represent, and its `Ok` always passes
  `validate_theta`** — every return goes through `admitted(self, tau, θ)`,
  which turns a θ rounded onto an excluded value into that τ error, so
  neither `fit` nor a caller storing the inverse can hold an invalid θ.

- **Errors are `CopulaError`** (`crate::error`), never `Box<dyn Error>` or a `String`: a parameter outside its domain is `InvalidParameter { name, value, constraint }` in the crate's assert form, an unfitted copula is `NotFitted`, a batch input without two columns is `DimensionMismatch`, a pseudo-observation off `[0, 1]` (or NaN) is `MarginalOutOfRange`, a 0 or 1 in a fit or the goodness-of-fit test is `MarginalOnBoundary`, a failed uniformity diagnostic is `MarginalNotUniform`.

Useful defaults you should usually **not** override:

| Method | Default behaviour |
|---|---|
| `sample(n, &seed)` | `SimdUniform::new(0, 1).seeded(seed)`, via `sample_with_uniform`; every coordinate in `(0, 1)` (an exact 0 or 1 moves one ulp inside) |
| `fit(x)` | two columns, two rows, entries in `[0, 1]` (`DimensionMismatch`, `InsufficientData`, `MarginalOutOfRange` in that order, no uniformity test), none equal to 0 or 1 (`MarginalOnBoundary`), Kendall's tau-b via `correlation::kendall_tau_b`, then `theta_from_tau` |
| `tau()` | `tau_from_theta` at the current `theta` |
| `percent_point(y, v)` | Brent inversion of `partial_derivative_scalar` (`percent_point_numerical`) |
| `percent_point_unchecked(y, v)` | hidden hook, `None`: the sampler's own uniforms are already in `[0, 1]`, so it takes the family's inverse where the hook answers `Some`, else `percent_point` |
| `generator(t)` | `MarginalOutOfRange` off `[0, 1]` or NaN, then the hidden `generator_unchecked` hook, whose default `None` is the "not Archimedean" `Unsupported` |
| `log_pdf` | `pdf(x)?.ln()` |
| `validate_theta` / `check_theta` / `check_fit` | validation, used by the above |
| `check_marginal(u)` | the Kolmogorov–Smirnov uniformity diagnostic (bound `1.627/√n`); `fit` does not run it |

Override `percent_point_unchecked` and `percent_point` only when you have an
inverse; they are a pair. Answer `percent_point_unchecked` with
`Some(self.unchecked_inverse(y, v))` and override `percent_point` with
`checked_percent_point(self, y, v, Self::unchecked_inverse)`, which checks the
parameters, the lengths and `[0, 1]` once (`MarginalOutOfRange`). The defaults
are `None` and `percent_point_numerical`, so an implementation outside the crate
may override `percent_point` alone, or call the hook from it. Every in-crate
family has the pair: a closed form mapped through
`bivariate::conditional::conditional_quantiles` (`amh`, `clayton`, `fgm`,
`frank`, `gaussian`, `independence`, `marshall_olkin`, `plackett`, `t_copula`),
or a Brent inverse of `ln h` through `numerical_quantiles` (`bb1`, `bb7`,
`galambos`, `gumbel`, `husler_reiss`, `joe`; the last four answer the identity at
their independence value). Both map checked inputs into the `[0, 1]` codomain,
whose clamp absorbs rounding only: a non-finite inverse reads as NaN, so a
closed form must stay finite on the whole closed square. Name it
`Family::inverse(params)` and add it to the excursion sweep in
`bivariate/conditional.rs`. The other batch methods go through `conditional_cdf`
(`partial_derivative`, required: the family's closed form), `copula_cdf` (`cdf`,
exact on the edges), `copula_pdf` and `on_open_square` (`pdf` and `log_pdf`, NaN
on the boundary); each answers `DimensionMismatch` for a column count other than
2, then `MarginalOutOfRange` off `[0, 1]`. Answer `generator_unchecked` with
`Some` if and only if the family is Archimedean.

`tail_dependence` is **required, not defaulted, on purpose**: a silent
`(0.0, 0.0)` fallback would be a correctness bug for Clayton, Gumbel,
Joe, Galambos, Hüsler-Reiss, Marshall-Olkin, Student-t, BB1, BB7 and Amh (at
`θ = 1`, where it is Clayton(1)). It returns
`Result` and opens with the gate `pdf` and `cdf` run, `self.check_fit()?`
(`check_theta()?` is the same check only where `check_fit` is not overridden).

## 2. The struct skeleton

Every family keeps its parameters private and its bounds as constants in `theta_bounds` / `invalid_thetas`; `new`
panics, `try_new` and `from_tau` return `CopulaError`, `Default` is the unfitted state `fit` starts from.
There is **no `seed` field** — seeding enters through `sample(n, &seed)`
with a `Deterministic` (replay) or `Unseeded` (entropy) source.

```rust
// stochastic-rs-copulas/src/bivariate/clayton.rs (reference)

use super::BivariateFamily;
use crate::error::CopulaError;
use crate::traits::BivariateExt;

#[derive(Debug, Clone, Default)]
pub struct Clayton {
  theta: Option<f64>,
}

impl Clayton {
  /// A Clayton copula; panics unless `0 <= theta < ∞`.
  pub fn new(theta: f64) -> Self {
    Self::try_new(theta).unwrap_or_else(|e| panic!("{e}"))
  }

  /// `new` with the domain error returned instead of a panic.
  pub fn try_new(theta: f64) -> Result<Self, CopulaError> {
    let copula = Self { theta: Some(theta) };
    copula.check_theta()?;
    Ok(copula)
  }

  /// The copula whose Kendall's τ is `tau`.
  pub fn from_tau(tau: f64) -> Result<Self, CopulaError> {
    Self::try_new(Self::default().theta_from_tau(tau)?)
  }
}

impl BivariateExt for Clayton {
  fn family(&self) -> BivariateFamily {
    BivariateFamily::Clayton
  }

  fn theta(&self) -> Option<f64> {
    self.theta
  }

  fn set_theta(&mut self, theta: f64) {
    self.validate_theta(theta).unwrap_or_else(|e| panic!("{e}"));
    self.theta = Some(theta);
  }

  fn theta_bounds(&self) -> (f64, f64) {
    (0.0, f64::INFINITY)
  }

  fn invalid_thetas(&self) -> Vec<f64> {
    vec![]
  }

  // tau_from_theta, theta_from_tau, tail_dependence, pdf, cdf, …
}
```

`validate_theta` admits a finite `theta` only, so an infinite bound renders as `< inf` in the panic and the error
(``theta must satisfy `0.0 <= theta < inf`, got theta = -1.0``). Add your family's variant to `BivariateFamily` (its
`ALL` and `parameter_count` too) plus a `pub mod` line in `stochastic-rs-copulas/src/bivariate.rs`.

Note Clayton's real bounds: `(0.0, f64::INFINITY)` with
`invalid_thetas()` returning `vec![]`. The `θ ∈ (-1, ∞)` domain with a degenerate
point at 0 is the textbook Clayton, but this crate ships the
positive-dependence branch only, and encodes that by the bound rather
than by an invalid-theta entry.

## 3. The `theta_from_tau` pattern — closed-form > Brent > custom

The §6.12 audit trap was a custom Newton iteration on Frank's τ → θ map
that diverged on positive correlations. The mandate:

1. **Closed-form first.** Clayton: `θ = 2τ/(1-τ)`. Gumbel:
   `θ = 1/(1-τ)`. Cite the textbook formula in a doc comment.

2. **Brent's method second**, when no closed form exists: the crate's
   `crate::optim::zero`, Brent's own procedure (1973, §6). `Frank`'s
   shipped implementation is the pattern to copy:

   ```rust
   fn theta_from_tau(&self, tau: f64) -> Result<f64, CopulaError> {
     static CAP: OnceLock<f64> = OnceLock::new();
     let cap = *CAP.get_or_init(|| self.tau_from_theta(1e4));
     if !(tau > -cap && tau < cap) {
       return Err(tau_out_of_range(tau, format!("{:?} < tau < {cap:?}", -cap)));
     }
     let theta = if tau.abs() < 1e-12 {
       0.0
     } else {
       let (lo, hi) = if tau > 0.0 { (0.0, 1e4) } else { (-1e4, 0.0) };
       invert_tau(|theta| self.tau_from_theta(theta) - tau, lo, hi, 1e-14)?
     };
     admitted(self, tau, theta)
   }
   ```

   Brent is bracketing → guaranteed convergence on a sign change.
   Newton is not. Frank brackets *on the sign of tau* from θ = 0, where
   `tau_from_theta` is exactly 0, and takes the range ends from the same
   `tau_from_theta`, so every τ that passes the range check brackets.
   Compute range ends once (a constant or a `OnceLock`, never per call),
   and pass `invert_tau` the absolute tolerance the family's τ rule
   actually resolves: `1e-14` behind a closed form, `1e-8` behind a
   quadrature. A bracket that grows over several decades is solved in
   `ln θ`; state the solver's top as the reachable range.

3. **Never** roll a custom Newton / secant.

Return `Err(tau_out_of_range(..))` at the ends of the reachable range;
never saturate.

## 4. Sampling — you get it for free, and it is conditional inversion

Do **not** write a `sample`. The default in `BivariateExt` already
implements the conditional-inversion (Rosenblatt) sampler:

```rust
// traits/bivariate.rs, defaulted — reproduced here for orientation only
let mut v = Array1::<f64>::zeros(n);
ud.fill_slice(v.as_slice_mut().unwrap());
let mut c = Array1::<f64>::zeros(n);
ud.fill_slice(c.as_slice_mut().unwrap());
let u = match self.percent_point_unchecked(c.view(), v.view()) {
  Some(u) => u?,
  None => self.percent_point(c.view(), v.view())?,
};
u.mapv_inplace(inside);
v.mapv_inplace(inside);
Ok(stack![Axis(1), u, v])
```

It is gated by the family's `check_fit`, which reads `theta`, not `tau`:
`NotFitted` while `theta` is unset, `InvalidParameter` when it is outside
`theta_bounds` or in `invalid_thetas`. It draws both uniforms from one seeded
`SimdUniform<f64>` stream and returns an `(n, 2)` `Array2<f64>` wrapped in
`Result`. `sample(&self, ..)` takes `&self`, not `&mut self`.

What you supply is the inverse pair of §1 — a closed form, or nothing
at all, in which case `percent_point_numerical` Brent-inverts your
`partial_derivative` for you. The sampler calls `percent_point_unchecked` on its
own uniforms and falls back to `percent_point` where the hook answers `None`,
so the hook never needs `percent_point` and an override of `percent_point` may
call the hook without recursing. For a degenerate branch use
`self.percent_point_numerical(y, v)` — calling `Self::percent_point` from inside
the override just recurses.

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
class — a wrong `theta_from_tau` looks fine on the data side but
produces samples with a different τ than requested:

```rust
#[test]
fn clayton_independence_sample_kendall_tau_near_zero() {
  let cop = Clayton::from_tau(target_tau).unwrap();
  let samples = cop.sample(50_000, &Deterministic::new(42)).unwrap();
  // empirical Kendall's tau on `samples` within tolerance of target_tau
}
```

**Pin the seed**: `sample(n, &Deterministic::new(42))` — never `&Unseeded` in a test. See
`integration-test-writing` for the full pinned-seed mandate.

## 6. Anti-patterns

- **Do not** give the struct a `seed` field. Seeding is per-call, via
  `sample(n, &seed)`.
- **Do not** expose a parameter field; `theta()` and the family's getters are the read path, `new` / `set_*` the
  validated write path.
- **Do not** write scalar `pdf(u, v)` / `cdf(u, v)`. The trait is
  vectorised over an `(n, 2)` array and returns `Result`.
- **Do not** clamp a τ that the family cannot reach to the boundary;
  return the `tau` error.
- **Do not** skip the parameter gate in `tail_dependence`: open with
  `check_fit()?`, the check `pdf` and `cdf` run.
- **Do not** implement `sample` / `fit` / `log_pdf`. They are
  defaulted and the defaults are correct.
- **Do not** answer `generator_unchecked` for a non-Archimedean family.
  Its default `None` makes `generator` return `CopulaError::Unsupported`
  (`"<Family> is not Archimedean: no generator"`).

## 7. Reference impls

- `Clayton` (`bivariate/clayton.rs`) — closed-form `theta_from_tau`,
  closed-form `percent_point` / `partial_derivative`, Archimedean
  `generator_unchecked`. The template.
- `Frank` (`bivariate/frank.rs`) — Brent-based `theta_from_tau` with
  sign-of-tau bracketing from θ = 0.
- `Gumbel` (`bivariate/gumbel.rs`) — closed-form via Archimedean
  generator; shares Joe's upper-tail formula.
- `Fgm` (`bivariate/fgm.rs`) — Farlie-Gumbel-Morgenstern, narrow
  τ ∈ [-0.22, 0.22].
- `MarshallOlkin` (`bivariate/marshall_olkin.rs`) — the awkward case:
  `new(alpha, beta)` takes the two shock rates directly (`from_tau` and
  `theta_from_tau` cover only the symmetric slice, the one where `theta()`
  is `Some`), and the only non-exchangeable family here.
  Read it before assuming every family fits the `theta` mould.

## Related SKILLs

- `integration-test-writing` — the pinned-seed mandate these tests obey.
- `add-mc-variance-reduction` — when copula sampling is part of an MC
  pricer using common random numbers.
- `python-bindings` — for the `PyClayton` etc. wrappers.
