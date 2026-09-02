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
├── stochastic-rs-quant/              — pricing/calibration/vol_surface + ModelPricer/ShortRatePricer/ToModel
├── stochastic-rs-ai/                 — neural surrogates (feature-gated upstream)
└── stochastic-rs-py/                 — pyo3 cdylib (250 entries: 232 PyO3 classes + 18 pyfunctions, across distributions/stochastic/quant/copulas/stats; AI bindings deferred to 2.x). Built via `maturin` (see pyproject.toml `[tool.maturin] manifest-path`)
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

- `RealExt` — scalar real-number bound (arithmetic, conversions, constants — no SIMD, no RNG); the bound analytic pricing code takes, and the door a custom scalar (AAD dual, tape node) can implement; lives in `stochastic-rs-distributions::traits`
- `SimdFloatExt` — 8-lane SIMD surface over `RealExt`, plus the uniform RNG fills
- `FloatExt` — the full simulation-grade bound: `RealExt + SimdFloatExt` + batched normal-fill/fGN scratch; only `f32`/`f64` can implement it, so anything bounded on it is closed to custom scalars by construction
- `ProcessExt<T>` — stochastic process simulation (`sample`/`sample_par`/`sample_map`); lives in `stochastic-rs-stochastic::traits`. **127** concrete implementors (`grep -rn "ProcessExt<T> for\|ProcessExt<T," stochastic-rs-stochastic/src --include='*.rs' | grep -v /traits/ | wc -l`); the exhaustive per-directory breakdown and the guard that keeps it honest live in `stochastic-rs-stochastic/tests/reproducibility_all_processes.rs`
- `BivariateExt` / `MultivariateExt` — copula traits in `stochastic-rs-copulas::traits`; **13** bivariate + **8** multivariate implementors (note: `NCopula2DExt` was removed in v2.0 — bivariate samplers consolidated under `BivariateExt`)
- `TimeExt` — day-count-aware maturity: `tau()`, `tau_or_from_dates()`, `tau_with_dcc(dcc)` (explicit day-count override), both NaN-on-missing-data by convention. Implemented by **instruments only** — `EuropeanOption` and `DigitalOption`, **2** production implementors (`grep -rn "impl TimeExt for" stochastic-rs-quant/src`); a pricer takes `tau` as a query argument and holds no dates. The old intention to move its role into `calendar` is **dropped**, not pending: the arithmetic already lives there and what the trait adds is an instrument concern
- `ModelPricer` — `price_call(s, k, r, q, tau)` / `price_put` (put-call parity default) / `price_option`; the struct holds model parameters and the query travels as arguments, which is what makes vectorized pricing across a strike/maturity grid possible. It replaced the bundled-market-data `PricerExt` (`calculate_call_put()` / `calculate_price()` / `implied_volatility()`), retired once its last implementor moved off it
- `ToModel` / `ToShortRateModel` — bridge a `Calibrator`'s output to a concrete pricer via associated type: `ToModel::Model: ModelPricer` for spot/strike models, `ToShortRateModel::Model` (no `ModelPricer` bound) for short-rate models (Hull-White, Black-Karasinski, G2++) that price off a yield curve and drift offset instead
- `FourierModelExt` — `chf(t, xi)` (characteristic function) + `cumulants(t)`; blanket-implements `ModelPricer` via Gil-Pelaez quadrature. `ModelSurface` (`vol_surface(s, r, q, strikes, maturities)`) blanket-implements over `VanillaEuropeanCall`, **not** over `ModelPricer` — its Black inversion is only meaningful for a European vanilla call, and the marker carries a `vanilla_call_forward` hook so a model whose carry is not `r - q` states its own forward rather than having one assumed
- `Calibrator` / `CalibrationResult` — `Calibrator::calibrate(initial) -> Result<Output, Error>`, `type Params: Clone`; `type Error` is a free associated type, not fixed by the trait, but all **12** in-tree calibrators set it to `anyhow::Error` by convention. `CalibrationResult` requires `rmse()`/`converged()`/`params()` and defaults `loss_score()`/`iterations()`/`message()` to `None` and `max_error()` to NaN
- `Instrument` / `InstrumentExt` / `PricingEngine` / `PricingResult` — QuantLib-style decoupling (`AnalyticBSEngine`, `AnalyticHestonEngine`)
- `Greeks` struct — first- + second-order Greeks (delta/gamma/vega/theta/rho/vanna/charm/volga/veta); the return type of the **5** identical inherent `greeks(s, k, r, q, tau, option_type)` aggregators (`BSMPricer`, `HestonPricer`, `Merton1976Pricer`, `CashOrNothingPricer`, `AssetOrNothingPricer`)
- `GreeksExt` — no-argument Greeks for the **2** query-bundled Monte Carlo Malliavin estimators only (`GbmMalliavinGreeks`, `HestonMalliavinGreeks`), whose `greeks()` override shares one simulation across the accessors. Not the crate's Greeks interface and not in the prelude — a pricer's Greeks are the inherent method above
- `CalendarExt` — pluggable holiday calendars for business day adjustment (`is_business_day`)
- `DistributionExt` — characteristic function / pdf / cdf / moments. **30** of the **34** distribution types implement it (`grep -rhE -A1 "impl.*DistributionExt" stochastic-rs-distributions/src --include='*.rs' | grep -oE "for (Simd[A-Za-z]+|ComplexDistribution)" | sed 's/for //' | sort -u | wc -l` — the `-A1` is load-bearing, three impl headers wrap onto a second line and a single-line grep undercounts to 23). The four without it are `SimdDirichlet`, `SimdWishart`, `SimdNonCentralChiSquared` and `ComplexDistribution`. Coverage **inside** an impl is uneven, so "implements `DistributionExt`" is not "has every closed form": 6 of the 30 (`SimdGed`, `SimdSkellam`, the four `truncated` wrappers) override only `pdf`/`cdf`, and 6 (`SimdAlphaStable`, `SimdBeta`, `SimdHypergeometric`, `SimdInverseGauss`, `SimdNormalInverseGauss`, `SimdVarianceGamma`) carry named no-closed-form `unimplemented!()` on specific methods. The authoritative per-type breakdown is the module docs in `stochastic-rs-distributions/src/lib.rs`. Defaults `unimplemented!("... is not implemented for {type}")`, **never `0.0`**.

## Prelude

```rust
use stochastic_rs::prelude::*;
```

Brings **25** items in 6 groups (`awk '/pub mod prelude/,/^}/' src/lib.rs | grep -c "^  pub use"`):

- **Trait core**: `RealExt`, `FloatExt`, `SimdFloatExt`, `ProcessExt`, `BivariateExt`, `MultivariateExt`, `DistributionExt`, `DistributionSampler`, `TimeExt`
- **Pricing**: `ModelPricer`
- **Calibration**: `Calibrator`, `CalibrationResult`, `ToModel`
- **Option types**: `Moneyness`, `OptionStyle`, `OptionType`
- **Backend / sampling**: `Backend`, `Cpu`, `PathSampler`, `VolterraKernel`
- **Estimation**: `HurstEstimator`, `FractalDimEstimator`, `HypothesisTest`, `DiffusionModel`, `TailDependence`

`MultivariateExt` joined the prelude in 3.0, when the linalg stack moved to the pure-Rust faer and its feature-gate exclusion reason died. `CallableDist` (python-only) stays reachable via `traits::*` but out of the prelude to keep it feature-flag-free. Same for `ShortRatePricer` (prices off a yield curve, not a spot/strike query), the two markers `VanillaEuropeanCall` / `ToShortRateModel`, and `GreeksExt` (2 implementors, both Monte Carlo estimators, 0 generic consumers — a no-argument trait beside a query-taking `ModelPricer` advertised a symmetry the crate does not have). The `Instrument`/`InstrumentExt`/`PricingEngine`/`PricingResult` four left the prelude in 3.0: two instruments and two engines (three engine×instrument pairings) are a cross-engine comparison harness for validating models on the same European vanilla, not a third pricing layer — the crate's two layers are `ModelPricer` (spot/strike query) and the instruments' `.valuation(curve)`. All four stay hub-reachable via `traits::*`, as is `FgnBackend` (the fGN capability subtrait of the prelude's `Backend` device marker — named only when writing generic code over backends).

Hub membership is **independent of prelude membership**: `src/traits.rs` mirrors every trait each sub-crate exports from its own `traits` module, prelude-excluded ones included. The quant half is derivable, and `tests/prelude_completeness.rs` turns a dropped re-export into a compile error:

```bash
diff <(grep '^pub use \(calibration\|instrument\|pricing\|short_rate\|time\)::' stochastic-rs-quant/src/traits.rs | sed 's/.*:://;s/;//' | sort) \
     <(grep '^pub use stochastic_rs_quant::traits::' src/traits.rs | sed 's/.*:://;s/;//' | sort)
```

## Skills

- Development rules and conventions: `.claude/skills/dev-rules/SKILL.md`
- New module integration checklist: `.claude/skills/new-module/SKILL.md`
