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

Version history belongs in `CHANGELOG.md` / git log / `docs/V*_UPDATE.md`, not in `///` or `//!` blocks. To record a genuine limitation near the code, describe **what is not supported and why** without the release number (e.g. "Nested-Clayton sampling is not yet implemented — needs Devroye double-rejection"), or use a `// TODO:` with a short rationale. When porting prose from a `V*_UPDATE.md` planning doc into a module header, strip the version prefix.
