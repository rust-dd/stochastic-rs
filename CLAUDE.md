# CLAUDE.md — stochastic-rs

Rust library for quantitative finance: stochastic process simulation, pricing, statistics, copulas, distributions, and AI-based volatility models. Published on crates.io as `stochastic-rs`.

## Workspace layout

Top-level workspace with sub-crates as siblings; `stochastic-rs` is the
umbrella that re-exports everything via `pub use`.

```
stochastic-rs/                        (workspace root + umbrella)
├── stochastic-rs-core/               — simd_rng (foundation)
├── stochastic-rs-distributions/      — FloatExt/SimdFloatExt + distributions
├── stochastic-rs-stochastic/         — ProcessExt + 127 processes (incl. interest::lmm::Lmm drift-coupled LMM)
├── stochastic-rs-copulas/            — BivariateExt + copulas (13 bivariate + 8 multivariate)
├── stochastic-rs-stats/              — estimators
├── stochastic-rs-quant/              — pricing/calibration/vol_surface + PricerExt/ModelPricer/ToModel
├── stochastic-rs-ai/                 — neural surrogates (feature-gated upstream)
└── stochastic-rs-py/                 — pyo3 cdylib (234 entries: 218 PyO3 classes + 16 pyfunctions, 13 of the classes openblas-gated, across distributions/stochastic/quant/copulas/stats; AI bindings deferred to 2.x). Built via `maturin` (see pyproject.toml `[tool.maturin] manifest-path`)
```

The umbrella crate `stochastic-rs` keeps the existing public API
(`stochastic_rs::stochastic::diffusion::gbm::GBM`, etc.) — sub-crate split is
transparent to users.

## Build & test

```bash
cargo build --workspace                                        # build all sub-crates
cargo test --workspace --exclude stochastic-rs-py              # run all tests
cargo check --workspace --no-default-features                  # fastest check (default)
cargo bench                                                    # run benchmarks (umbrella)
cargo check -p stochastic-rs --features ai                     # with AI surrogates
cargo build -p stochastic-rs-distributions                     # build single sub-crate
```

`--exclude stochastic-rs-py` is required, not optional: that crate forces
`pyo3/extension-module` unconditionally, so a plain `cargo test --workspace`
fails to link on any machine (there is no host Python providing the
extension-module symbols outside a `maturin`-built `.so`) — confirmed by
reproducing the linker error on a clean checkout. CI's `test` job
(`.github/workflows/rust.yml`) works around the same constraint the same
way.

## Clippy usage

Always run `cargo clippy` to adopt the latest compiler recommendations.

## Key traits

Shapes below are read from the trait definitions, not remembered — re-check
against `stochastic-rs-quant/src/traits/*.rs` and
`stochastic-rs-distributions/src/traits/distribution.rs` before trusting an
older summary.

- `FloatExt` — core float trait bound; lives in `stochastic-rs-distributions::traits`
- `SimdFloatExt` — SIMD-friendly subset of `FloatExt`
- `ProcessExt<T>` — stochastic process simulation (`sample`/`sample_par`/`sample_map`); lives in `stochastic-rs-stochastic::traits`. **127** concrete implementors (`grep -rn "ProcessExt<T> for\|ProcessExt<T," stochastic-rs-stochastic/src --include='*.rs' | grep -v /traits/ | wc -l`); the exhaustive per-directory breakdown and the guard that keeps it honest live in `stochastic-rs-stochastic/tests/reproducibility_all_processes.rs`
- `MalliavinExt<T>` / `Malliavin2DExt<T>` — finite-difference Malliavin Greeks (0 in-tree implementors today — deferred)
- `BivariateExt` / `MultivariateExt` — copula traits in `stochastic-rs-copulas::traits`; **13** bivariate + **8** multivariate implementors (note: `NCopula2DExt` was removed in v2.0 — bivariate samplers consolidated under `BivariateExt`)
- `TimeExt` — day-count-aware maturity: `tau()`, `tau_or_from_dates()`, `tau_with_dcc(dcc)` (explicit day-count override), both NaN-on-missing-data by convention
- `PricerExt: TimeExt` — the legacy bundled-market-data pricer surface: `calculate_call_put()`, `calculate_price()`, `implied_volatility()` (defaults to NaN)
- `ModelPricer` — `price_call(s, k, r, q, tau)` / `price_put` (put-call parity default) / `price_option`; separates the model from the pricing query, unlike `PricerExt` which bundles market data into the pricer — this is what makes vectorized pricing across a strike/maturity grid possible
- `ToModel` / `ToShortRateModel` — bridge a `Calibrator`'s output to a concrete pricer via associated type: `ToModel::Model: ModelPricer` for spot/strike models, `ToShortRateModel::Model` (no `ModelPricer` bound) for short-rate models (Hull-White, Black-Karasinski, G2++) that price off a yield curve and drift offset instead
- `FourierModelExt` — `chf(t, xi)` (characteristic function) + `cumulants(t)`; blanket-implements `ModelPricer` via Gil-Pelaez quadrature, and every `ModelPricer` (Fourier-based or not) blanket-implements `ModelSurface` (`vol_surface(s, r, q, strikes, maturities)`)
- `Calibrator` / `CalibrationResult` — `Calibrator::calibrate(initial) -> Result<Output, Error>`, `type Params: Clone`; `type Error` is a free associated type, not fixed by the trait, but all **12** in-tree calibrators set it to `anyhow::Error` by convention. `CalibrationResult` requires `rmse()`/`converged()`/`params()` and defaults `loss_score()`/`iterations()`/`message()` to `None` and `max_error()` to NaN
- `Instrument` / `InstrumentExt` / `PricingEngine` / `PricingResult` — QuantLib-style decoupling (`AnalyticBSEngine`, `AnalyticHestonEngine`)
- `GreeksExt` + `Greeks` struct — first- + second-order Greeks (delta/gamma/vega/theta/rho/vanna/charm/volga/veta), aggregator with single-pass override for MC pricers
- `CalendarExt` — pluggable holiday calendars for business day adjustment (`is_business_day`)
- `DistributionExt` — characteristic function / pdf / cdf / moments. **18/19** distributions implement closed-form (only `ComplexDistribution` lacks; 5 named no-closed-form `unimplemented!()` cases on specific moments — see `project_distribution_ext_status.md` memory). Defaults `unimplemented!("... is not implemented for {type}")`, **never `0.0`**.

## Prelude

```rust
use stochastic_rs::prelude::*;
```

Brings **29** items in 7 groups (`awk '/pub mod prelude/,/^}/' src/lib.rs | grep -c "^  pub use"`):

- **Trait core**: `FloatExt`, `SimdFloatExt`, `ProcessExt`, `BivariateExt`, `DistributionExt`, `DistributionSampler`, `TimeExt`
- **Pricing**: `PricerExt`, `ModelPricer`, `GreeksExt`
- **Calibration**: `Calibrator`, `CalibrationResult`, `ToModel`
- **Instrument / engine**: `Instrument`, `InstrumentExt`, `PricingEngine`, `PricingResult`
- **Option types**: `Moneyness`, `OptionStyle`, `OptionType`
- **Backend / sampling**: `Backend`, `Cpu`, `PathSampler`, `VolterraKernel`
- **Estimation**: `HurstEstimator`, `FractalDimEstimator`, `HypothesisTest`, `DiffusionModel`, `TailDependence`

`MalliavinExt` / `Malliavin2DExt` are intentionally **not** in the prelude (0 in-tree impls — deferred). Reach via `stochastic_rs::traits::MalliavinExt`. `MultivariateExt` (openblas-only) and `CallableDist` (python-only) likewise reachable via `traits::*` but excluded from the prelude to keep it feature-flag-free.

## Skills

- Development rules and conventions: `.claude/skills/dev-rules/SKILL.md`
- New module integration checklist: `.claude/skills/new-module/SKILL.md`
