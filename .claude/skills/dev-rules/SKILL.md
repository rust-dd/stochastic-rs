---
name: dev-rules
description: Development rules for stochastic-rs — enforces project conventions when writing new modules, adding dependencies, or implementing algorithms
---

# Development Rules for stochastic-rs

## 1. Follow folder structure

```
src/
  stochastic/       — stochastic processes (diffusion, volatility, jump, noise, interest, autoregressive, correlation, malliavin, sheet)
  quant/            — quantitative finance (pricing, bonds, portfolio, strategies, calendar, fx, order_book, loss)
  stats/            — statistical estimators and tests (stationarity, normality, spectral, MLE, KDE)
  distributions/    — probability distributions
  copulas/          — copula models (bivariate, multivariate, univariate, empirical, correlation)
  ai/               — neural network based models (volatility calibration)
  traits.rs         — core traits (FloatExt, ProcessExt, MalliavinExt, etc.)
  macros.rs         — helper macros
benches/            — criterion benchmarks
tests/              — integration & comparison tests
```

Place new code in the appropriate existing module. Do not create top-level modules without explicit approval.

## 2. Generic over float

All new structs, traits, and functions must be generic over the float type using the existing `FloatExt` trait bound (`T: FloatExt`). Never hardcode `f64` in new code.

## 3. Use `ndarray` everywhere

Use `ndarray::Array1<T>`, `Array2<T>`, etc. for all numeric arrays. Do not use `Vec<T>` for numerical data. The project already depends on `ndarray`, `ndarray-stats`, and `ndrustfft`.

## 4. Research via arXiv MCP

When implementing a new model or algorithm, use the **arxiv MCP tool** (`mcp__arxiv__arxiv_search_papers`, `mcp__arxiv__arxiv_get_paper`) to find and verify the underlying theory before writing code.

## 5. Comparison tests and benchmarks

Every new module must include:
- **Comparison test**: validate output against the reference implementation (Python, R, MATLAB, or the original paper's numerical examples)
- **Criterion benchmark**: add a bench in `benches/` to track performance

## 6. Scientific references

Every new module must cite its source. Add a doc comment at the top of the file with:
- Paper title and authors
- DOI or arXiv ID
- Example: `//! Reference: Heston (1993), DOI: 10.1093/rfs/6.2.327`

## 7. Prefer maintained libraries over raw implementations

Do not rewrite algorithms that already exist in well-maintained crates (e.g., `faer`, `basin`, `roots`, `ndrustfft`). Use existing crate implementations and only write custom code when no suitable crate exists.

Randomness is the standing exception — see §7a.

## 7a. `rand::rng()` and `rand_distr` belong to benchmarks only

Library code, tests and examples draw randomness from the workspace's own
RNG and distributions. `rand::rng()` and every concrete `rand_distr`
distribution (`Normal`, `Exp`, `Gamma`, `Poisson`, `StandardNormal`, …) are
reserved for `benches/` and the `src/tests/bench_*` plot harnesses, where
`rand_distr` is a dev-dependency (of the umbrella and
`stochastic-rs-distributions`, the only crates that declare it) and the
*baseline being measured*. The `rand` crate itself stays a dependency for
its traits (`Rng`, `RngExt`, `rand::distr::Distribution`,
`rand::seq::SliceRandom`).

| Need | Use |
|------|-----|
| A raw RNG | `SimdRng::new()`, or `SimdRng::from_seed(s)` when reproducible |
| Bulk Gaussian / exponential / … draws | `SimdNormal`, `SimdExp`, `SimdGamma`, `SimdPoisson`, seeded via `Deterministic::new(s)` or `Unseeded` |
| A distribution a *process* will drive (`D: Distribution<T> + Send + Sync`) | the stateless `SimdNormal`, `SimdExp`, ... themselves |

A process's jump-size slot takes any scalar continuous `Simd*` law except `SimdNonCentralChiSquared` (its noncentrality
is a per-draw argument): it holds parameters only, so it is `Send + Sync`, and `Distribution::sample(&mut rng)` draws
from the generator the process passes. A `Seeded` stream is not a law.

Two traps worth naming:

- **The stream lives in `Seeded`.** A ported `Simd*` law holds parameters
  only; `.seeded(&Deterministic::new(42))` binds it to a stream, and
  `sample` / `fill_slice` / `sample_n` / `sample_matrix` then run on
  `&mut self` (`SimdNormal::<f64>::new(0.0, 1.0).seeded(&Deterministic::new(42))`).
  `fill_slice(out)` takes no RNG: the seed goes to `.seeded`, never to a
  draw; a bulk fill driven by an external rng is `fill_with(&mut rng, out)`.
- **Name the trait through `rand::distr::Distribution`.** `rand_distr` is
  not a dependency of any library crate (only a dev-dependency of the
  umbrella and `stochastic-rs-distributions`), so library code cannot
  import it; our own `Simd*` types implement the same trait, which is how
  `.sample()` resolves. Removing those impls would break downstream users.

## 8. Latest dependency versions

When adding a new dependency, always use the latest version available on crates.io. Check with `cargo search <crate>` before adding.

## 9. Comments

A comment or doc block (`//`, `///`, `//!`) is at most two lines, anywhere: library
code, public API, tests, benches and examples. Write one only for what the code cannot
say: a non-obvious constraint, a reason, a unit or a convention. A literature citation
may take one line of its own. Doc-test code does not count. Longer explanations belong
in the PR body.

Never:

- restate the code (`/// Volatility.` on `pub sigma`, `// loop over the paths`);
- narrate history (`previously`, `now uses`, `renamed from`, `legacy`, `kept for
  backward compatibility`, `since the refactor`): the code as it stands is the subject;
- cite plan, task, wave, audit or ruling IDs (`Task 3`, `W6.7`, `A1-c`, `D1`), or
  release numbers (§11);
- use separator banners such as

```
// --- ... ---

###############
# ....        #
###############

// ---------------------------------------------------------------------------
// free-text
// ---------------------------------------------------------------------------
```

Rewrites of three typical offenders (a history-laden note on draw order, a field doc
that restates the field, a long method essay):

```rust
// Drawn before the jump times so the diffusion stream matches the λ = 0 path.
let z = normal.sample();

/// Annualised Black volatility of the underlying.
pub sigma: f64,

/// Fang & Oosterlee (2008), SIAM J. Sci. Comput. 31(2), 826-848.
/// COS price of a European payoff from the log-price characteristic function.
pub fn price(/* … */) -> f64 { /* … */ }
```

## 10. Turbofish over explicit binding-type annotation

Where the same type information can be expressed via turbofish on the call site, prefer turbofish — it travels with the expression and is shorter than a binding-type annotation.

```rust
// Avoid:
let x: f64 = 1.0_f64.ln_1p();
let arr: Array1<f64> = Array1::zeros(8);
let v: Vec<f64> = (0..8).map(|i| i as f64).collect();
let mean: T = sum / T::from_usize_(n);

// Prefer:
let x = 1.0_f64.ln_1p();              // suffix carries the type already
let arr = Array1::<f64>::zeros(8);
let v = (0..8).map(|i| i as f64).collect::<Vec<_>>();
let mean = sum / T::from_usize_(n);   // sum's type already drives T
```

Exceptions where a binding annotation IS warranted:
- The right-hand side has no method-/call-site type to attach turbofish to (e.g. a literal `let p: f64 = 0.5;` when the surrounding code is generic).
- The annotation documents an invariant about the *binding* itself (e.g. `let weights: [f64; 4] = read_calibration();` to lock the array length in the type).
- Inference would otherwise pick a different (numerically wrong) type.

Use the binding annotation in those three cases; everywhere else, turbofish.

## 11. No version-tagged sections in source doc-comments

Doc comments describe what the module / item *does*, not which release ships it. Don't write headers or prose like

```
//! ## v2.3.0 design choice — XYZ
//! ## v2.4 deferred — ABC
//! In v2.3.0 we ship only the closed-form path; the refinement lands in v2.4.
```

Version history belongs in `MIGRATION.md` (the repo root breaking-changes record — there is no `CHANGELOG.md` here) / git log / `docs/V*_UPDATE.md`, not in `///` or `//!` blocks. To record a genuine limitation near the code, describe **what is not supported and why** without the release number (e.g. "Nested-Clayton sampling is not yet implemented — needs Devroye double-rejection"), or use a `// TODO:` with a short rationale. When porting prose from a `V*_UPDATE.md` planning doc into a module header, strip the version prefix.

## 12. One blank line between items

Separate consecutive items — functions (trait-impl methods included), impls,
structs, enums, consts, statics, type aliases, modules, macros — with exactly
one blank line, including before an attribute or doc comment that opens the
next item. rustfmt does not insert these (its `blank_lines_lower_bound` also
pads statements, so it is not used); generated code from patch scripts and
heredocs must emit the blank lines itself. Never:

```rust
impl Foo for Bar {
  fn a(&self) -> f64 {
    1.0
  }
  fn b(&self) -> f64 {
    2.0
  }
}
```

## 13. Library diagnostics go through `log`

Library code reports through the `log` facade (`log::warn!`, `log::trace!`), never `eprintln!`, and never installs a subscriber; binaries, benches and tests do.

## 14. Error policy — one channel per cause, in every crate

- **A parameter precondition broken in a constructor or setter → panic**, one `assert!` per
  argument, the predicate spelled in the message and the value last:
  ``assert!(sigma > T::zero(), "sigma must satisfy `sigma > T::zero()`, got sigma = {sigma:?}");``
  Every float parameter is finite (`x.is_finite()`; a truncation bound may be infinite, so `!x.is_nan()`).
  A method that can panic says so inside its ≤ 2-line doc (`…; panics if x ≤ 0`), never under a separate
  `# Panics` heading. The literal-only Fourier model structs validate once they have constructors.
  In copulas `new` panics, while `try_new` and `from_tau` return `Result<_, CopulaError>` for the
  same preconditions.
- **A data-dependent failure → `Result`**: a calibration that produces no result, an estimate on
  too little or degenerate data, a device that cannot be opened. Non-convergence is `Ok` with
  `converged() == false`, never `Err`. quant and ai return `anyhow::Error`, copulas `CopulaError`,
  the devices `DeviceError`; `Calibrator::Error: Debug + Display + Send + Sync + 'static`. The risk
  estimators (`empirical_cvar`, `historical_var`) still panic on an empty sample; whether short or
  degenerate data there becomes a `Result` is one pending decision for all of them.
- **A numerical evaluation outside its domain, or an inversion with no root → documented `NaN`**:
  a yield at `tau <= 0`, an implied volatility with no root, a Bessel K at `x <= 0`, a pricing
  *query* with a non-positive strike or spot. A `0.0` or any other plausible number is never a
  sentinel, and a query argument never panics — only a model parameter does.
- **A capability an implementor does not have → `Option::None`**: `DistributionExt`, `GreeksExt`,
  `CalibrationResult::max_error`, `Cumulants.c4`. `None` is "no closed form / not computed";
  `Some(f64::NAN)` is "computed, undefined at this point".
- **A reduction never drops a NaN.** `f64::max`/`min` return the other operand when one is NaN;
  use `RealExt::max_or_nan` / `min_or_nan` (`.fold(0.0, f64::max_or_nan)`) wherever the folded
  values can be NaN, so a NaN input surfaces as a NaN output instead of a silently smaller maximum.
