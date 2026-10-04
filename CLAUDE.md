# CLAUDE.md — stochastic-rs

Rust library for quantitative finance: stochastic process simulation, pricing, statistics, copulas, distributions, and AI-based volatility models. Published on crates.io as `stochastic-rs`.

## Workspace layout

Top-level workspace with sub-crates as siblings; `stochastic-rs` is the
umbrella that re-exports everything via `pub use`.

```
stochastic-rs/                        (workspace root + umbrella)
├── stochastic-rs-core/               — simd_rng (foundation)
├── stochastic-rs-distributions/      — FloatExt/SimdFloatExt + distributions
├── stochastic-rs-stochastic/         — ProcessExt + 132 processes (incl. interest::lmm::Lmm drift-coupled LMM, volatility::heston_slv::HestonSlv)
├── stochastic-rs-copulas/            — BivariateExt + copulas (15 bivariate + 8 multivariate)
├── stochastic-rs-stats/              — estimators
├── stochastic-rs-quant/              — pricing/calibration/vol_surface + ModelPricer/ShortRatePricer/ToModel
├── stochastic-rs-ai/                 — experimental neural surrogates + surrogate→Calibrator bridge (feature-gated upstream, outside the stability promise)
└── stochastic-rs-py/                 — pyo3 cdylib (308 entries: 286 PyO3 classes + 22 pyfunctions, plus 3 classes + 1 pyfunction behind the py `ai` feature (source builds only), across distributions/stochastic/quant/copulas/stats; AI bindings behind the py `ai` feature). Built via `maturin` (see pyproject.toml `[tool.maturin] manifest-path`)
```

The umbrella crate `stochastic-rs` keeps the existing public API
(`stochastic_rs::stochastic::diffusion::gbm::GBM`, etc.) — sub-crate split is
transparent to users.

The umbrella re-exports `ndarray`, `num_complex`, `num_traits`, `rand` and
`chrono` (as `stochastic_rs::ndarray` and so on) because callers build and pass
their types; use these paths or depend on the same versions. `wide` (the SIMD
vectors of `SimdFloatExt` and `SimdRngExt`) and `anyhow` (calibration and SLV errors)
also appear in public signatures. A semver-incompatible release of any of these
seven crates (for a 0.x crate, a minor bump such as ndarray 0.17 → 0.18) is a
major release of `stochastic-rs`.

## Build & test

```bash
cargo build --workspace                                        # build all sub-crates
cargo test --workspace --exclude stochastic-rs-py              # run all tests
cargo check --workspace --no-default-features                  # fastest check (default)
cargo bench                                                    # run benchmarks (umbrella)
cargo check -p stochastic-rs --features ai                     # with AI surrogates
cargo build -p stochastic-rs-distributions                     # build single sub-crate
```

`--exclude stochastic-rs-py` keeps the Rust gate independent of a Python toolchain: the crate has no Rust
tests (its pytest suite runs in the `python_smoke` job). A plain `cargo test --workspace` also links now
that no manifest enables pyo3's deprecated `extension-module` feature, but needs a shared libpython at
build and run time (`PYO3_PYTHON` picks it). Never export `PYO3_BUILD_EXTENSION_MODULE` in a shell; it
is for maturin.

`Cargo.lock` is committed and CI passes `--locked`; a manifest change and its
lockfile update go in the same commit. The Apple back-ends are gated with
`all(feature = "metal", target_os = "macos")` (likewise `accelerate`), never a
bare `feature = "metal"`.

## Clippy usage

Always run `cargo clippy` to adopt the latest compiler recommendations.

## Key traits

Shapes below are read from the trait definitions, not remembered — re-check
against `stochastic-rs-quant/src/traits/*.rs` and
`stochastic-rs-distributions/src/traits/distribution.rs` before trusting an
older summary.

- Error policy — panic for a broken constructor precondition, `Result` for data, documented NaN out of domain, `None` for an absent capability, NaN-propagating reductions (`max_or_nan`): `.claude/skills/dev-rules/SKILL.md` §14.
- `RealExt` — scalar real-number bound (arithmetic, conversions, constants — no SIMD, no RNG); the bound analytic pricing code takes, and the door a custom scalar (AAD dual, tape node) can implement; lives in `stochastic-rs-distributions::traits`
- `SimdFloatExt` — 8-lane SIMD surface over `RealExt`, plus the uniform RNG fills; sealed (`f32`/`f64`); carries the four hidden Markov-lift kernels (`history_sum_fused` …), so `T: FloatExt` is the only bound the rough family needs
- `FloatExt` — the full simulation-grade bound: `RealExt + SimdFloatExt` + batched normal-fill/fGN scratch; sealed through `SimdFloatExt` (only `f32`/`f64` implement it), so anything bounded on it is closed to custom scalars by construction
- `ProcessExt<T>` — stochastic process simulation (`sample`/`sample_par`/`sample_map`, its borrowing twin `sample_map_view` for the single-row processes — the callback reads a row of the device batch where `sample_map` hands it a copy, 1.4-2.1x on Metal — plus the fallible `try_sample`/`try_sample_par` that report a `DeviceError` instead of panicking, and `device_ready()` — provided, default `true` — which says whether the configuration a process holds runs on a device backend; a `false` samples on the host bit-identically to the `Cpu` build, the one fallback rule, never a panic); lives in `stochastic-rs-stochastic::traits`; sealed together with `PathSampler` — every implementor carries `impl<…> crate::traits::Sealed for X<…> {}` with its own generics directly above the impl, which is all the seal asks of a new process. Every process also has `.on::<B>()` (a backend's default handle) and the public `with_backend(handle)` (an explicit one, `Cuda::new(1)` or `Cuda::from_env()?`; `backend()` returns `&B`), both from `backend_switch!`. **142** concrete implementors over **132** processes — the ten extra are launch and row views (`AdgRow`, `BgmRow`, `CfouParts`, `LmmLaunch`, `McgnsLaunch`, `MultiGbmLaunch`, `MultifactorHestonLaunch`, `MultivariateHawkesLaunch`, `WishartLaunch`, `WuZhangPair`), not processes of their own (`grep -rhE -A1 "impl.*ProcessExt<T>" stochastic-rs-stochastic/src --include='*.rs' | grep -oE "for [A-Z][A-Za-z0-9]*" | sed 's/for //' | sort -u | wc -l` — the `-A1` is load-bearing, an `impl` header that wraps onto a second line is invisible to a single-line grep); the exhaustive per-directory breakdown and the guard that keeps it honest live in `stochastic-rs-stochastic/tests/reproducibility_all_processes.rs`
- `BivariateExt` / `MultivariateExt` — copula traits in `stochastic-rs-copulas::traits`; **15** bivariate + **8** multivariate implementors (note: `NCopula2DExt` was removed in v2.0 — bivariate samplers consolidated under `BivariateExt`); every fallible method returns `CopulaError` (`#[non_exhaustive]`, `Send + Sync`, nine variants)
- `TimeExt` — day-count-aware maturity: `tau()`, `tau_or_from_dates()`, `tau_with_dcc(dcc)` (explicit day-count override), both NaN-on-missing-data by convention. Implemented by **instruments only** — `EuropeanOption` and `DigitalOption`, **2** production implementors (`grep -rn "impl TimeExt for" stochastic-rs-quant/src`); a pricer takes `tau` as a query argument and holds no dates. The old intention to move its role into `calendar` is **dropped**, not pending: the arithmetic already lives there and what the trait adds is an instrument concern
- `ModelPricer` — `price_call(s, k, r, q, tau)` / `price_put` (put-call parity default) / `price_option`; the struct holds model parameters and the query travels as arguments, which is what makes vectorized pricing across a strike/maturity grid possible. It replaced the bundled-market-data `PricerExt` (`calculate_call_put()` / `calculate_price()` / `implied_volatility()`), retired once its last implementor moved off it
- `ToModel` / `ToShortRateModel` — bridge a `Calibrator`'s output to a concrete pricer via associated type: `ToModel::Model: ModelPricer` for spot/strike models, `ToShortRateModel::Model` (no `ModelPricer` bound) for the short-rate models (Hull-White, Black-Karasinski, G2++); `to_short_rate_model(&self)` takes no arguments because the result stores the curve-side inputs (`initial_rate`, HW's `theta`, BK's `long_run_rate`) its calibrator was given
- `FourierModelExt` — `chf(t, xi)` (characteristic function) + `cumulants(t)`; blanket-implements `ModelPricer` via Gil-Pelaez quadrature. `ModelSurface` (`vol_surface(s, r, q, strikes, maturities)`) blanket-implements over `VanillaEuropeanCall`, **not** over `ModelPricer` — its Black inversion is only meaningful for a European vanilla call, and the marker carries a `vanilla_call_forward` hook so a model whose carry is not `r - q` states its own forward rather than having one assumed; `Cumulants.c4` is `Option<f64>` (`None` for the Heston family and Bates)
- `Calibrator` / `CalibrationResult` — `Calibrator::calibrate(initial) -> Result<Output, Error>`, `type Params: Clone`; `type Error: Debug + Display + Send + Sync + 'static`, which all **18** in-tree calibrators meet with `anyhow::Error`. `CalibrationResult` requires `rmse()`/`converged()`/`params()` and defaults `loss_score()`, `iterations()`, `message()` and `max_error()` to `None`
- `Instrument` / `InstrumentExt` / `PricingEngine` / `PricingResult` — QuantLib-style decoupling (`AnalyticBSEngine`, `AnalyticHestonEngine`)
- `Greeks` struct — first- + second-order Greeks (delta/gamma/vega/theta/rho/vanna/charm/volga/veta); the return type of the **5** identical inherent `greeks(s, k, r, q, tau, option_type)` aggregators (`BSMPricer`, `HestonPricer`, `Merton1976Pricer`, `CashOrNothingPricer`, `AssetOrNothingPricer`)
- `GreeksExt` — no-argument Greeks for the **2** query-bundled Monte Carlo Malliavin estimators only (`GbmMalliavinGreeks`, `HestonMalliavinGreeks`), whose `greeks()` override shares one simulation across the accessors. Not the crate's Greeks interface and not in the prelude — a pricer's Greeks are the inherent method above; `GreeksExt`'s accessors return `Option<f64>`
- `Fn1D` / `Fn2D` — type-erased callables of time (and state) a process takes as coefficients (`Cheyette`, `VolterraSde`, `Hjm`, `HestonSlv`); `Fn2D::Expr(Program)` holds a coefficient written as an `Expr` (`stochastic-rs-distributions/src/traits/callable.rs`: sixteen node kinds, `Expr::compile() -> Result<Program, ProgramError>` (also `TryFrom<Expr> for Fn2D`) yields a ≤ 62-op postfix `Program`, stack ≤ 8; the three enums are `#[non_exhaustive]`), the one form the device kernels interpret — a Rust closure, a Python callable or a tabulated `Fn2D::Grid(Grid2D)` (`traits/grid.rs`, bilinear with a flat hold outside the grid; what a calibrated `LeverageSurface` converts into) keeps such a process on the host
- `CalendarExt` — pluggable holiday calendars for business day adjustment (`is_business_day`)
- `SimdDistribution` / `SimdKernel` / `Seeded<D, R>` — a stateless law, its SIMD kernel, and the seeded stream that implements the sealed `&mut self` `DistributionSampler` (D18); `.seeded(&seed)` binds, `fill_with(&mut rng, out)` bulk-fills from any rng
- `DistributionExt` — `Option`-valued characteristic function / pdf / cdf / `quantile` / moments (default `None`, **never `0.0`**, no panic; `Some(NaN)` only where the quantity provably does not exist or the argument lies outside its domain). **32** of the **36** distribution types implement it (`grep -rhE "impl.*DistributionExt for" stochastic-rs-distributions/src --include='*.rs' | grep -oE "for (Simd[A-Za-z]+|ComplexDistribution)" | sort -u | wc -l`), plus `Gbm`'s terminal law. The four without it are `SimdDirichlet`, `SimdWishart`, `SimdNonCentralChiSquared` and `ComplexDistribution`. Coverage inside an impl is pinned cell by cell by `stochastic-rs-distributions/tests/distribution_ext_coverage.rs`, the authority — it fails when an implementor has no row (`Gbm`'s row is in `gbm.rs`'s tests).
- `HypothesisTest` — `statistic()` and `null_rejected() -> Option<bool>`; lives in `stochastic-rs-distributions::traits::hypothesis` (stats re-exports it) so the **16** implementors span stats (15) and copulas (`GofResult`): `grep -rhE "HypothesisTest for" stochastic-rs-stats/src stochastic-rs-copulas/src --include='*.rs' | wc -l`

## Prelude

```rust
use stochastic_rs::prelude::*;
```

Brings **27** items in 6 groups (`awk '/pub mod prelude/,/^}/' src/lib.rs | grep -c "^  pub use"`):

- **Trait core**: `RealExt`, `FloatExt`, `SimdFloatExt`, `ProcessExt`, `BivariateExt`, `MultivariateExt`, `DistributionExt`, `DistributionSampler`, `TimeExt`
- **Pricing**: `ModelPricer`
- **Calibration**: `Calibrator`, `CalibrationResult`, `ToModel`
- **Option types**: `Moneyness`, `OptionStyle`, `OptionType`
- **Backend / sampling**: `Backend`, `Cpu`, `VolterraKernel`, `Seeded`, `SimdDistribution`, `SimdKernel`
- **Estimation**: `HurstEstimator`, `FractalDimEstimator`, `HypothesisTest`, `DiffusionModel`, `TailDependence`

`MultivariateExt` joined the prelude in 3.0, when the linalg stack moved to the pure-Rust faer and its feature-gate exclusion reason died. `ShortRatePricer` (prices off a yield curve, not a spot/strike query), the two markers `VanillaEuropeanCall` / `ToShortRateModel`, and `GreeksExt` (2 implementors, both Monte Carlo estimators, 0 generic consumers — a no-argument trait beside a query-taking `ModelPricer` advertised a symmetry the crate does not have) stay reachable via `traits::*` but out of the prelude. The `Instrument`/`InstrumentExt`/`PricingEngine`/`PricingResult` four left the prelude in 3.0: two instruments and two engines (three engine×instrument pairings) are a cross-engine comparison harness for validating models on the same European vanilla, not a third pricing layer — the crate's two layers are `ModelPricer` (spot/strike query) and the instruments' `.valuation(curve)`. All four stay hub-reachable via `traits::*`, as are `FgnBackend`, `SheetBackend` and `EulerBackend` (the fGN, sheet-pipeline and Euler-engine capability subtraits of the prelude's `Backend` device marker — named only when writing generic code over backends). `EulerKernel` and `EulerSystem` left the hub in 3.0: with `EulerSpec`, `EulerCoefficients`, the lift/table/program specs and the slot caps they are the engine's own machinery — `pub` for the crate's tests and probes, `#[doc(hidden)]`, outside the stability promise (see the `euler` module doc's "What is public here"). `PathSampler` is in neither the prelude nor the hub: `#[doc(hidden)]`, sealed, reachable as `stochastic_rs::stochastic::traits::PathSampler` only for the GAT bound.

Hub membership is **independent of prelude membership**: `src/traits.rs` mirrors every documented trait each sub-crate exports from its own `traits` module, prelude-excluded ones included. The quant half is derivable, and `tests/prelude_completeness.rs` turns a dropped re-export into a compile error:

```bash
diff <(grep '^pub use \(calibration\|instrument\|pricing\|short_rate\|time\)::' stochastic-rs-quant/src/traits.rs | sed 's/.*:://;s/;//' | sort) \
     <(grep '^pub use stochastic_rs_quant::traits::' src/traits.rs | sed 's/.*:://;s/;//' | sort)
```

## Skills

- Development rules and conventions: `.claude/skills/dev-rules/SKILL.md`
- New module integration checklist: `.claude/skills/new-module/SKILL.md`
- New process checklist (host + device + bindings + docs): `.claude/skills/new-process/SKILL.md`
