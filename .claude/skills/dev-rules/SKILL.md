---
name: dev-rules
description: Development rules for stochastic-rs — enforces project conventions when writing new modules, adding dependencies, or implementing algorithms
---

# Development Rules for stochastic-rs

## 1. Follow folder structure

```
src/
  stochastic/       — stochastic processes (diffusion, volatility, jump, noise, interest, autoregressive, correlation, malliavin, sheet)
  quant/            — quantitative finance (pricing, bonds, portfolio, strategies, order_book, loss)
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

Do not rewrite algorithms that already exist in well-maintained crates (e.g., `ndarray-linalg`, `argmin`, `roots`, `ndrustfft`, `statrs`). Use existing crate implementations and only write custom code when no suitable crate exists.

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
