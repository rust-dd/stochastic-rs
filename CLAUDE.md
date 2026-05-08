# CLAUDE.md — stochastic-rs

Rust library for quantitative finance: stochastic process simulation, pricing, statistics, copulas, distributions, and AI-based volatility models. Published on crates.io as `stochastic-rs`.

## Workspace layout

Top-level workspace with sub-crates as siblings; `stochastic-rs` is the
umbrella that re-exports everything via `pub use`.

```
stochastic-rs/                        (workspace root + umbrella)
├── stochastic-rs-core/               — simd_rng (foundation)
├── stochastic-rs-distributions/      — FloatExt/SimdFloatExt + distributions
├── stochastic-rs-stochastic/         — ProcessExt + 120+ processes (incl. interest::lmm::Lmm drift-coupled LMM)
├── stochastic-rs-copulas/            — BivariateExt + copulas
├── stochastic-rs-stats/              — estimators
├── stochastic-rs-quant/              — pricing/calibration/vol_surface + PricerExt/ModelPricer/ToModel
├── stochastic-rs-ai/                 — neural surrogates (feature-gated upstream)
├── stochastic-rs-viz/                — Plotly grid plotter
└── stochastic-rs-py/                 — pyo3 cdylib (102 PyO3 classes for distributions + stochastic; quant/copulas/stats/ai are 2.x follow-up). Built via `maturin` (see pyproject.toml `[tool.maturin] manifest-path`)
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
- `BivariateExt` / `MultivariateExt` — copula traits in `stochastic-rs-copulas::traits` (note: `NCopula2DExt` was removed in v2.0 — bivariate samplers consolidated under `BivariateExt`)
- `TimeExt` / `PricerExt` — option pricing with date support (`tau_with_dcc` for day-count-aware maturity)
- `ModelPricer` — concrete-typed pricer interface (no `&dyn` / `Box<dyn>`)
- `ToModel` / `ToShortRateModel` — `Calibrator → Concrete pricer` bridge via associated type
- `FourierModelExt` — characteristic function models (blanket impl → `ModelPricer` → `ModelSurface`)
- `Calibrator` / `CalibrationResult` — unified calibration trait surface (Result-based, `type Params`, `type Error = anyhow::Error`)
- `Instrument` / `InstrumentExt` / `PricingEngine` / `PricingResult` — QuantLib-style decoupling (`AnalyticBSEngine`, `AnalyticHestonEngine`)
- `GreeksExt` + `Greeks` struct — first- + second-order Greeks (delta/gamma/vega/theta/rho/vanna/charm/volga/veta), aggregator with single-pass override for MC pricers
- `CalendarExt` — pluggable holiday calendars for business day adjustment
- `DistributionExt` — characteristic function / pdf / cdf / moments. **18/19** distributions implement closed-form (only `ComplexDistribution` lacks; 5 named no-closed-form `unimplemented!()` cases on specific moments — see `project_distribution_ext_status.md` memory). Defaults `unimplemented!("not implemented for {type_name}")`, **never `0.0`**.

## Prelude

```rust
use stochastic_rs::prelude::*;
```

Brings 20 items in 5 groups:

- **Trait core**: `FloatExt`, `SimdFloatExt`, `ProcessExt`, `BivariateExt`, `DistributionExt`, `DistributionSampler`, `TimeExt`
- **Pricing**: `PricerExt`, `ModelPricer`, `GreeksExt`
- **Calibration**: `Calibrator`, `CalibrationResult`, `ToModel`
- **Instrument / engine**: `Instrument`, `InstrumentExt`, `PricingEngine`, `PricingResult`
- **Option types**: `Moneyness`, `OptionStyle`, `OptionType`

`MalliavinExt` / `Malliavin2DExt` are intentionally **not** in the prelude (0 in-tree impls — deferred). Reach via `stochastic_rs::traits::MalliavinExt`. `MultivariateExt` (openblas-only) and `CallableDist` (python-only) likewise reachable via `traits::*` but excluded from the prelude to keep it feature-flag-free.

## Skills

- Development rules and conventions: `.claude/skills/dev-rules/SKILL.md`
- New module integration checklist: `.claude/skills/new-module/SKILL.md`
