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

Do not rewrite algorithms that already exist in well-maintained crates (e.g., `ndarray-linalg`, `argmin`, `roots`, `ndrustfft`). Use existing crate implementations and only write custom code when no suitable crate exists.

Randomness is the standing exception — see §7a.

## 7a. `rand` and `rand_distr` belong to benchmarks only

Library code, tests and examples draw randomness from the workspace's own
RNG and distributions. `rand::rng()`, `rand::thread_rng()` and every
concrete `rand_distr` distribution (`Normal`, `Exp`, `Gamma`, `Poisson`,
`StandardNormal`, …) are reserved for `benches/` and the `src/tests/bench_*`
plot harnesses, where `rand_distr` is the *baseline being measured* and
must stay.

| Need | Use |
|------|-----|
| A raw RNG | `SimdRng::new()`, or `SimdRng::from_seed(s)` when reproducible |
| Bulk Gaussian / exponential / … draws | `SimdNormal`, `SimdExp`, `SimdGamma`, `SimdPoisson`, seeded via `Deterministic::new(s)` or `Unseeded` |
| A distribution a *process* will drive (`D: Distribution<T> + Send + Sync`) | `ScalarNormal`, `ScalarExp` from `stochastic_rs_distributions::scalar` |

The last row is not a style preference. `Simd*` distributions own an
`UnsafeCell` sample buffer, so they are `!Sync` by construction and cannot
satisfy the `Send + Sync` bound that `ProcessExt` propagates into the
jump-size slot of `CompoundPoisson`, `Bates1996`, `LevyDiffusion` and
`JumpFOUCustom`. The stateless `Scalar*` types sample from the caller's
RNG and exist precisely for that slot.

Two traps worth naming:

- **`fill_slice(out)` takes no RNG at all.** Every `Simd*` bulk fill is
  `pub fn fill_slice(&self, out: &mut [T])` — one argument. The type
  draws from its own internal stream, seeded at construction, so there
  is nowhere to hand an external `StdRng`; the seed must go to the
  constructor (`SimdNormal::<f64>::new(0.0, 1.0, &Deterministic::new(42))`).
  Older notes describing a two-argument `fill_slice(_rng, out)` that
  ignored its first parameter, or a `fill_slice_fast` companion, are
  stale: neither exists.
- **The `rand_distr::Distribution` trait import stays.** Our own `Simd*`
  types implement it, so `use rand_distr::Distribution;` is still how
  `.sample()` resolves. Removing those impls would break downstream users;
  only the concrete `rand_distr` *distributions* are out.

## 8. Latest dependency versions

When adding a new dependency, always use the latest version available on crates.io. Check with `cargo search <crate>` before adding.

## 9. Comment rules
Always follow the Rust inline comment or Rust inline documentation pattern. Never use large ugly separators like
```
// --- ... ---

or 

###############
# ....        #
###############

or 

// ---------------------------------------------------------------------------
// free-text
// ---------------------------------------------------------------------------
```

or similar. Keep the project clean and dont use ugly AI style comments.

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
