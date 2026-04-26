# CLAUDE.md — stochastic-rs

Rust library for quantitative finance: stochastic process simulation, pricing, statistics, copulas, distributions, and AI-based volatility models. Published on crates.io as `stochastic-rs`.

## Workspace layout

Top-level workspace with sub-crates as siblings; `stochastic-rs` is the
umbrella that re-exports everything via `pub use`.

```
stochastic-rs/                        (workspace root + umbrella)
├── stochastic-rs-core/               — simd_rng (foundation)
├── stochastic-rs-distributions/      — FloatExt/SimdFloatExt + distributions
├── stochastic-rs-stochastic/         — ProcessExt + 140+ processes
├── stochastic-rs-copulas/            — BivariateExt + copulas
├── stochastic-rs-stats/              — estimators
├── stochastic-rs-quant/              — pricing/calibration/vol_surface + PricerExt/ModelPricer/ToModel
├── stochastic-rs-ai/                 — neural surrogates (feature-gated upstream)
├── stochastic-rs-viz/                — Plotly grid plotter
└── stochastic-rs-py/                 — placeholder (Phase 6 follow-up)
```

The umbrella crate `stochastic-rs` keeps the existing public API
(`stochastic_rs::stochastic::diffusion::gbm::GBM`, etc.) — sub-crate split is
transparent to users.

## Build & test

```bash
cargo build --workspace                          # build all sub-crates
cargo test --workspace                           # run all tests
cargo check --workspace --no-default-features    # fastest check (default)
cargo bench                                      # run benchmarks (umbrella)
cargo check -p stochastic-rs --features ai       # with AI surrogates
cargo build -p stochastic-rs-distributions       # build single sub-crate
```

## Clippy usage

Always run `cargo clippy` to adopt the latest compiler recommendations.

## Key traits

- `FloatExt` — core float trait bound; lives in `stochastic-rs-distributions::traits`
- `SimdFloatExt` — SIMD-friendly subset of `FloatExt`
- `ProcessExt<T>` — stochastic process simulation; lives in `stochastic-rs-stochastic::traits`
- `MalliavinExt<T>` / `Malliavin2DExt<T>` — finite-difference Malliavin Greeks
- `BivariateExt` / `MultivariateExt` / `NCopula2DExt` — copula traits in `stochastic-rs-copulas::traits`
- `TimeExt` / `PricerExt` — option pricing with date support (`tau_with_dcc` for day-count-aware maturity)
- `ModelPricer` — concrete-typed pricer interface (no `&dyn` / `Box<dyn>`)
- `ToModel` — `Calibrator → Concrete pricer` bridge via associated type `Model: ModelPricer`
- `FourierModelExt` — characteristic function models (blanket impl → `ModelPricer` → `ModelSurface`)
- `CalendarExt` — pluggable holiday calendars for business day adjustment
- `DistributionExt` — characteristic function / pdf / cdf / moments; implemented for Normal/LogNormal/Gamma (others stub to 0.0)

## Prelude

```rust
use stochastic_rs::prelude::*;
```

Brings: `BivariateExt`, `DistributionExt`, `DistributionSampler`, `FloatExt`,
`ModelPricer`, `PricerExt`, `ProcessExt`, `SimdFloatExt`, `TimeExt`,
`ToModel`, `Moneyness`, `OptionStyle`, `OptionType`.

## Skills

- Development rules and conventions: `.claude/skills/dev-rules/SKILL.md`
- New module integration checklist: `.claude/skills/new-module/SKILL.md`
