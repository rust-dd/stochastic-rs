# CLAUDE.md — stochastic-rs

## Project overview

Rust library for quantitative finance: stochastic process simulation, pricing, statistics, copulas, distributions, and AI-based volatility models. Published on crates.io as `stochastic-rs`.

## Folder structure

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

**Always follow this structure.** Place new code in the appropriate existing module. Do not create top-level modules without explicit approval.

## Development rules

### 1. Generic over float

All new structs, traits, and functions must be generic over the float type using the existing `FloatExt` trait bound (`T: FloatExt`). Never hardcode `f64` in new code.

### 2. Use `ndarray` everywhere

Use `ndarray::Array1<T>`, `Array2<T>`, etc. for all numeric arrays. Do not use `Vec<T>` for numerical data. The project already depends on `ndarray`, `ndarray-stats`, and `ndrustfft`.

### 3. Research via arXiv MCP

When implementing a new model or algorithm, use the **arxiv MCP tool** (`mcp__arxiv__arxiv_search_papers`, `mcp__arxiv__arxiv_get_paper`) to find and verify the underlying theory before writing code.

### 4. Comparison tests and benchmarks

Every new module must include:
- **Comparison test**: validate output against the reference implementation (Python, R, MATLAB, or the original paper's numerical examples)
- **Criterion benchmark**: add a bench in `benches/` to track performance

### 5. Scientific references

Every new module must cite its source. Add a doc comment at the top of the file with:
- Paper title and authors
- DOI or arXiv ID
- Example: `//! Reference: Heston (1993), DOI: 10.1093/rfs/6.2.327`

### 6. Prefer maintained libraries over raw implementations

Do not rewrite algorithms that already exist in well-maintained crates (e.g., `ndarray-linalg`, `argmin`, `roots`, `ndrustfft`, `statrs`). Use existing crate implementations and only write custom code when no suitable crate exists.

### 7. Latest dependency versions

When adding a new dependency, always use the latest version available on crates.io. Check with `cargo search <crate>` before adding.

## Build & test

```bash
cargo build                    # build
cargo test                     # run tests
cargo bench                    # run benchmarks
cargo build --features cuda    # build with CUDA support
cargo build --features python  # build with Python bindings (pyo3)
```
