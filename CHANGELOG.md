# Changelog

All notable changes to `stochastic-rs` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0-rc.2] — 2026-05-10

Third release candidate. Closes the §1.6 stable-cut residuals **plus**
all 40 §2 deferred-to-2.0.x P2 polish items **plus** the 16 §3 SKILLs
from `docs/V2_RC2_TODO.md` after the rc.1 fix-pass (memo:
`.claude/projects/.../memory/project_v2_release_status.md`):

### §2 P2 polish (40 items, all in-tree)

- Pricing: `unreachable!()` cleanups, NaN-tolerant `partial_cmp`,
  regime-switching matrix ops via ndarray operators (was hand-written
  triple loops), document hardcoded paper maturities,
  fourier_malliavin canonical-window comments,
  heston_stoch_corr FFT-vs-quadrature doc fix.
- Vol/risk: SVI butterfly arb-freeness check + Durrleman `g(k)` ;
  Rockafellar-Uryasev tail-share correction in `historical_es` ;
  `max_peak_to_recovery_duration` companion to existing duration ;
  `fourier_model_surface_fft_with(...)` exposes Carr-Madan grid params ;
  `try_from_quotes` Result variant on `ImpliedVolSurface` ;
  Sortino / momentum ScoreWeighted convention doc ; `mat_inverse` →
  `nalgebra::try_inverse` ; Strategies module: `Strategy` trait + `Backtest`
  driver + `MarketBar` / `StrategyAction` types ; bucket_dv01 sign doc.
- Rates: `curves::Instrument` → `BootstrapInstrument` (+ alias) ;
  five new day-count conventions (BUS/252, ACT/364, ACT/ACT AFB, NL/365,
  30E/360 ISDA) ; ScheduleBuilder long-first / long-last `StubConvention` ;
  observable Mutex poisoning is logged + recovered (was panic) ;
  fx Display fallback to NaN ; short_rate round-isize defensive default.
- Microstructure / factors: stray no-ops removed (shrinkage, bootstrap,
  migration) ; loss.rs metrics return NaN on zero-divisor ; pca.rs
  `to_f64` defensive ; `CalibrationLossScore::get` → NaN + `try_get` ;
  scipy.linalg.expm regression test for migration matrix `expm`.

### §3 SKILLs (16 written, `.claude/skills/`)

- **Tier 1** (release-level): `release-checklist`,
  `feature-flag-management`, `calibration-pattern`, `greeks-pattern`.
- **Tier 2** (domain): `add-diffusion-process`, `add-fractional-process`,
  `add-jump-process`, `adding-distribution`, `adding-python-binding`,
  `stats-estimator`.
- **Tier 3** (niche): `copula-bivariate`, `add-gpu-sampler`,
  `add-mc-variance-reduction`, `vol-surrogate-nn`,
  `integration-test-writing`, `bench-writing`.

### §1.6 stable-cut residuals

- `cargo clippy --workspace --all-targets -- -D warnings` is now clean
  (was 3 errors: deprecated `FromPyObject` on `PyNigFourier`, lazy
  `unwrap_or_else` in `order_book::add_order`, negated-comparison in
  `PyNigFourier::new`).
- `stats::fukasawa_hurst` test suite (`estimate_h_from_simulated_rv`,
  `rough_vs_smooth_distinguished`, `table1_m72_accuracy`) now uses
  `Fou::seeded(...)` for the volatility path so the entire RNG chain is
  pinned.
- Python coverage gaps from `§1.5` filled: `PyHscmModel`, `PyHscmCalibrator`
  (+ `PyHscmMarketOption`), `PyHullWhiteBond` (with
  `HullWhiteBond.from_curve(...)` taking a `DiscountCurve`), and
  `empirical_cvar` exposed as a top-level `#[pyfunction]` with the
  documented `alpha < 0.5` precondition translated to `PyValueError`.
- `portfolio::optimizers::empirical_cvar` is now `pub` (was a private
  helper) so external callers can reach it directly.
- Added `docs/BENCH_BASELINE.md` documenting the criterion baseline
  workflow (`--save-baseline rc2` / `--baseline rc2`) for tracking
  perf regressions ≤ 5% pre-stable.

## [2.0.0-rc.1] — 2026-05-10

Second release candidate. Closes the 6 P0 + 13 P1 + 23 P2 items found by the
2026-05-07 audit (`docs/V2_RELEASE_AUDIT_2026-05-07.md`) **plus** the 11 P0 +
3 P1 silent-correctness bugs surfaced by the 2026-05-10 deep quant audit
(`docs/QUANT_AUDIT_2026-05-10_*.md`). Adds a proper drift-coupled LIBOR
Market Model and expands the Python surface from 102 to 210 entries
(excludes AI bindings by design).

### Quant deep audit fixes (2026-05-10)

Eleven P0 silent-correctness bugs and three P1 trait/test-tolerance issues
found by a 4-agent deep audit covering pricing/calibration (~25k LOC),
vol_surface/risk/portfolio (~6.7k LOC), rates/lattice/instruments (~13.3k
LOC), and traits/factors/microstructure (~7k LOC). All 14 closed in this
release.

**Pricing & calibration:**

- `calibration/heston_stoch_corr.rs:202` — `let _ = slsqp::minimize(...)`
  discarded the calibrated parameters. `HscmCalibrationResult` returned the
  initial guess as if calibrated. Fixed by capturing the slsqp return tuple;
  `HscmCalibrationResult` gains `converged` (real SLSQP termination status)
  and `final_objective` fields. Existing `calibration_runs` test marked
  `#[ignore]` (it is slow); a new synthetic-prices recovery test asserts
  `final_objective < initial_sse * 0.5`.
- `calibration/levy.rs:197-204` — `LevyCalibrationResult::to_model` for
  `LevyModelType::Nig` wrapped NIG params in `CGMYFourier` with hardcoded
  `y = 0.5`. Added `pricing/fourier.rs::NigFourier` (Barndorff-Nielsen 1997
  characteristic function) and rewired `LevyModel::Nig`. New round-trip
  test asserts `to_model().price_call(...)` matches `fourier_call_price`.
- `pricing/heston_stoch_corr.rs:191,212,379` — HSCM characteristic function
  used `iu·r` in the drift instead of `iu·(r−q)` and `HscmModel::price_call`
  silently dropped its `_q` argument. Threaded q through the ODE drift
  coefficient and the `ModelPricer` impl. New parity-with-q test asserts
  `C - P = S·e^(-qτ) - K·e^(-rτ)` for q > 0.
- `pricing/rbergomi.rs:114` + `calibration/rbergomi.rs:824,881` —
  `RBergomiPricer::price_call` discarded `_q` and
  `simulate_rbergomi_terminal_samples` had no q parameter. Drift now uses
  `(r-q-0.5·v)·dt`. `RBergomiCalibrator::with_dividend_yield(q)` builder
  added. New `respects_dividend_yield` test asserts ATM call drops as q
  rises.
- `pricing/fourier.rs:152` — `CarrMadanPricer::price_call` returned `0.0`
  silently for log-strikes outside the FFT grid. Now returns `f64::NAN` so
  calibration objectives are poisoned (forcing detection) rather than
  silently zero-residualed at the wings. New `strike_in_grid()` helper for
  callers who want to detect-and-extend.
- `calibration/heston.rs:1561` — Cui (2017) Jacobian numeric-consistency
  test tolerance tightened from `2e-1` (20%) to `5e-3`. Cui math is correct
  line-for-line vs the paper; the loose tolerance would have masked
  5–10% regressions.

**Vol surface & risk:**

- `vol_surface/ssvi.rs:309-312` — `SsviSurface::is_calendar_spread_free`
  only checked the ATM term structure $\theta_t$; missed off-ATM violations
  that pass the butterfly bound. Now takes a `ks: &[T]` parameter and
  verifies the full Gatheral & Jacquier 2014 Theorem 4.2 condition
  $w(k, \theta_{n+1}) \geq w(k, \theta_n)$ across the smile grid.
  `is_atm_calendar_spread_free` retained as the ATM-only check.
  `pipeline.rs::build_surface_from_iv` now feeds the union of slice
  log-moneyness grids into the calendar check.
- `portfolio/optimizers.rs:102-113` — `empirical_cvar` `alpha` is the **tail
  proportion** to average (e.g. `0.05` = worst 5%) but the rest of the risk
  module uses `confidence = 1 - tail`. Added `assert!(alpha < 0.5)` so users
  who pass `0.95` (confidence-style) crash loudly instead of silently
  averaging nearly the whole distribution. `cvar_alpha` field doc on
  `PortfolioEngineConfig` updated.

**Rates / lattice / bonds:**

- `bonds/cir.rs:13-49` — closed-form bond formula returned `A·exp(+B·r)`
  instead of `A·exp(-B·r)` (sign error: ZCB increased with the short rate).
  Field doc-comments swapped `theta`/`mu` from Vasicek's workspace
  convention. Both fixed; new `zcb_decreases_with_rate` regression test
  mirrors `vasicek.rs:122-136`.
- `bonds/hull_white.rs:40-58` — closed-form bond used `Utc::now().year()`
  in the price calculation (non-determinism), unwrapped eval/expiration
  unconditionally, and hardcoded a flat zero curve. **Rewritten** against
  Brigo & Mercurio §3.3.2 / Hull-White (1990) extended-Vasicek closed form
  using a `DiscountCurve`-projected representation (initial discount
  factors + instantaneous forward at evaluation time). New
  `HullWhite::from_curve` builder; new `zcb_at_t_zero_matches_market_curve`
  no-arbitrage test; new `deterministic_no_clock_dependency` regression
  asserts the `Utc::now()` poison is gone. References:
  Kisbye-Meier 2017 (arXiv:1707.02496 §3, eq. 3.3-3.4).

**Microstructure & traits:**

- `microstructure/kyle.rs:110-118` — multi-period Kyle 1985 backward
  recursion `α_{n-1} = (1 - √α_n)/2` was non-canonical and disagreed with
  `single_period_kyle` even at `n_periods = 1` (β·λ = 0.25 vs canonical
  0.5). Re-derived against Cetin-Larsen 2023 (arXiv:2307.09392) Theorem 2.1
  using the substitution $γ_n = α_n λ_n$ (cubic recursion in γ + forward
  Σ recursion). New `multi_period_one_round_matches_single_period` and
  `multi_period_two_round_matches_canonical` analytic-benchmark tests.
- `traits/time.rs:14-37` — `tau_or_from_dates` and `tau_with_dcc` panicked
  when neither `tau` nor `(eval, expiration)` was set. Now return `f64::NAN`
  (matching the crate's `Greeks::default = nan()` and
  `CalibrationResult::max_error` defaults). `pricing/merton_jump.rs:210`
  switched from `self.tau.unwrap()` to `self.tau_or_from_dates()`.

**Trait conformance (P1 — must ship pre-stable, breaking later):**

- Three calibrators now implement the unified `Calibrator` trait surface
  with `Result<Output, Error>` signature: `CgmysvCalibrator`,
  `HKDECalibrator`, and a new `HscmCalibrator` wrapper around the free
  `calibrate_hscm` function. (`HullWhiteSwaptionCalibrator` already had it.)
  Generic pipelines (`build_surface_from_calibration` etc.) can now consume
  these.
- `calibration/sabr.rs:126-141` — `SabrParams ↔ DVector<f64>` round-trip
  silently forced `β = 1.0`. Public `From` impls now use a 4-vec
  `[α, β, ν, ρ]` (lossless). Internal LM solver uses a private
  `as_lm_vec()` 3-vec helper that excludes β (β remains user-set, not
  optimised). New `sabr_params_dvector_round_trip_preserves_beta` test.

### P0 release-blocker fixes

- `stochastic-rs-stochastic/src/process/fbm.rs` — duplicate `Array2` import on
  the `python + gpu` feature combo (E0252) fixed by unifying the import under
  a single `cfg(any(...))` block. `cargo check --workspace --all-features`
  now passes. CI matrix expanded to test the combo (see §5.2).
- `stochastic-rs-stochastic/src/interest/bgm.rs` — header rewritten to honestly
  scope the implementation as a parallel array of forward-Euler-stepped
  multiplicative martingales (`L(t+dt) = L(t)(1 + λ·ΔW)`), **not** a BGM/LMM,
  **not** exact log-normal, paths can go negative for non-trivial `λ √dt`. A
  proper drift-coupled LIBOR Market Model is now available as
  `interest::lmm::Lmm` (spot-LIBOR measure, log-Euler positivity-preserving
  stepping, optional Cholesky correlation).
- `diffusion/{ou,cir}.rs` + `interest/vasicek.rs` — `theta` / `mu` field doc
  swap. `theta` is now correctly documented as mean-reversion **speed** and
  `mu` as long-run **mean**, matching the implementation `dx = theta·(mu - x)·dt`.
- `copulas::multivariate::{Tree, Vine}` — explicit module-level scope-doc
  identifying these as **Gaussian-collapsed implied-correlation copulas**,
  not real R-vine pair-copula constructions. Rename deferred to a future 2.x
  minor.
- `stats::fou_estimator::FilterType::Classical` — variant removed. The public
  path that previously hit `unimplemented!()` no longer compiles.
- `tests/debug_fukasawa.rs` — `#[ignore]`-d (was a no-assertion diagnostic
  test polluting `cargo test --workspace` output).

### P1 fixes

- CI matrix: 8-element feature combo (`""`, `ai`, `openblas`, `yahoo`, `python`,
  `gpu`, `python,gpu`, `openblas,ai,yahoo`) + new `lint` job
  (`cargo fmt --check` + `cargo clippy --workspace -D warnings`).
  `actions-rs/toolchain@v1` → `dtolnay/rust-toolchain@stable`.
- `publish.sh` — pre-publish gate (fmt + clippy + workspace test) added
  before the publish loop. Optional `--skip-gate` bypass flag.
- `copulas::multivariate::{Tree, Vine}` — replaced 4 `rand::random::<f64>()`
  call sites with `SimdNormal::fill_slice_fast` so seeded usage is deterministic.
- 17 production `panic!`s in `stochastic-rs-stochastic` refactored to
  constructor-side validation (`validate_drift_args`, `validate_n_or_tmax`
  helpers). Runtime panics now reserved for true invariant violations with
  precise messages.
- `stochastic-rs-distributions` test pollution: 12 bench-style tests
  `#[ignore]`-d. `cargo test -p stochastic-rs-distributions` 119s → 0.03s.
- PyPI workflow now builds `macos-13 + x86_64` alongside `aarch64`. Intel Mac
  users can `pip install stochastic-rs` again.
- `stats::fou_estimator` V1/V2/V4 — refactored from struct-based API to
  free functions over `ArrayView1<f64>` (zero-copy boundary). Hurst/sigma
  setters replaced by explicit `*_override: Option<f64>` parameters. V3
  remains a sim+est round-trip helper. 786 → 562 LoC; 2 → 11 tests; 3
  bit-exact regression tests guard the refactor against numerical drift.
- `stats::fd::FractalDim` — `panic!()` paths replaced with
  `Result<FdResult, FdError>`. 6-variant `FdError` enum
  (`PathTooShort {got, required}`, `NonPositiveP(f64)`, `KmaxTooSmall(usize)`,
  `DegeneratePath`, `NotEnoughScales`, `RegressionFailed`).
- `bergomi.rs` / `rbergomi.rs` — honest scope-doc identifying these as
  scaled-Brownian-motion approximations (variance-matched, **not** true Volterra
  integrals). User pointers added to `crate::rough::MarkovLift`,
  `crate::rough::rl_heston::RlHeston`, `crate::rough::rl_bs::RlBs`,
  `crate::process::volterra::Volterra`.
- `diffusion::Gbm::sample` — x0 default `0.0` → `1.0`, consistent with
  `terminal_lognormal_params` (GBM at 0 is an absorbing fixed point).
- `stochastic-rs-ai` dead code removed: `lib.rs::DataSet` struct, `utils.rs`
  (duplicate `train_test_split_for_array2`), and the unused `traits.rs`
  re-export module.
- `stochastic-rs-core/src/python.rs` is now `#[cfg(feature = "python")]`-gated
  (was always-compiled).
- `docs/V1_TO_V2.md` §2.1 prelude list — corrected from 13 to 20 entries,
  grouped by Trait core / Pricing / Calibration / Instrument-engine / Option
  types. `MultivariateExt` and `CallableDist` feature-gate explanation added.

### Added

- **`stochastic-rs-stochastic::interest::lmm::Lmm`** — drift-coupled LIBOR
  Market Model with spot-LIBOR measure, log-Euler positivity-preserving
  stepping, and optional Cholesky correlation matrix.
- **`examples/calibration_demo.rs`** — end-to-end multi-maturity calibration
  example (BSM + Heston).
- **`Vasicek::from_fou_estimate`** — stats↔quant adapter bridging
  `FouEstimateResult` to a calibrated Vasicek short-rate model.
- **`StochVolNn::predict_implied_vol_surface`** + thin wrappers for `HestonNn`
  / `OneFactorNn` / `RBergomiNn`. Gated behind the `quant` cargo feature on
  `stochastic-rs-ai`; the umbrella `ai` feature pulls it in automatically.
- **`HypothesisTest`** trait + 8 implementations (ADF, KPSS, ERS,
  PhillipsPerron, LeybourneMcCabe, JarqueBera, AndersonDarling,
  ShapiroFrancia).
- **`VariableDimensional<T>`** marker trait — output-type marker for
  `multivariate_hawkes` and similar processes returning `Vec<Array1<T>>`.
- **`ComplexPathOutput<T>`** marker trait — output-type marker for processes
  with complex-valued sample paths (e.g. `cfou`).
- **Frank copula** `compute_theta` now uses Brent root-finding + chunked
  Gauss-Legendre quadrature; the prior custom Newton solver had a math bug
  in the Frank tau formula (Genest-MacKay 1986 reference now correct).
- **`stochastic-rs-viz`** — split into `plottable.rs` + `grid_plotter.rs` +
  `convenience.rs`; new `Plottable<T>` trait + `GridPlotter` builder. New
  `plot_process` / `plot_distribution` / `plot_vol_surface` convenience
  functions.

### Python bindings (`stochastic-rs-py`)

96 new PyO3 classes + 12 pyfunctions for **210 total entries** (was 102 in
rc.0). All wrappers accept an optional `seed=None` keyword argument routed
through `Deterministic::new(seed)`. New surface:

- **Distributions / stochastic** (existing) — every `Simd*` distribution and
  `Process*` wrapper now seed-aware.
- **Quant pricers** (~20): BSM, Heston, Sabr, Merton1976, Asian, Barrier,
  Compound, Chooser, Cliquet, Gap, SuperShare, CashOrNothing, AssetOrNothing,
  Floating-and-Fixed Lookback, BjerksundStensland2002, DoubleBarrier,
  MCBarrier, VarianceSwap, KirkSpread.
- **Quant Fourier** (8 + Carr-Madan engine): BSM, Heston, DoubleHeston,
  HKDE, Bates, Kou, MertonJD, CGMY, VG.
- **Bonds**: Vasicek, CIR.
- **Calibrators** (10): BSM, Heston, Sabr, SVJ, DoubleHeston, HKDE, RBergomi,
  Levy, SabrCaplet, Cgmysv.
- **Vol surface** (5): SviRawParams, SsviParams, SviCalibrator, SsviCalibrator,
  ImpliedVolSurface (FFT).
- **Risk** (3): VaR, ExpectedShortfall, DrawdownStats.
- **Microstructure** (3 class + 5 fn): AlmgrenChrissPlan, KyleEquilibrium,
  OrderBook + multi_period_kyle, roll/effective/corwin_schultz_spread,
  propagator_price_impact.
- **Curves** (3): DiscountCurve, NelsonSiegel, ZeroCouponInflationCurve.
- **Factors** (2 fn + 3 openblas-gated): ledoit_wolf_shrinkage,
  sample_covariance, PCA, FamaMacBeth, PairsStrategy.
- **Copulas** (5 class + 4 fn): Clayton, Gumbel, Frank, Independence,
  EmpiricalCopula2D + kendall_tau_matrix, tau_matrix_to_corr_matrix,
  tau_to_corr, corr_to_tau.
- **Stats** — Normality (3), Stationarity (5, openblas-gated), Hurst (2),
  Heston (2), Realised (6), Spectral/Changepoint (3), Density (3),
  Econometrics (4, openblas-gated), MCMC (1 fn).

Skipped per design: AI bindings (user decision); HullWhite bond
(`fn(f64)->f64` field), HwSwaptionCalibrator (`&'a DiscountCurve` lifetime),
UKF/ParticleFilter (closure args; only MCMC got a Python-callback wrapper),
Cashflows (CurveProvider trait), inflation swaps (InflationCurve trait).

### Changed (breaking from rc.0)

- `stats::fou_estimator::FilterType` — `Classical` variant removed (was
  panic-only). Enum is now `enum FilterType { Daubechies }`.
- `stats::fou_estimator::FOUParameterEstimationV{1,2,4}` structs replaced by
  free functions `estimate_fou_v{1,2,4}(path: ArrayView1<f64>, ...)`. V3
  retained as a sim+est round-trip helper. The `FouEstimateResult` struct
  used by `quant::calibration::rbergomi` is unchanged.
- `stats::fd::FractalDim::estimate` returns `Result<FdResult, FdError>`
  instead of panicking on degenerate input.
- `Gbm::sample` x0 default is now `1.0` (was `0.0`, which is an absorbing
  fixed point and inconsistent with `terminal_lognormal_params`).

### Documentation

- `CLAUDE.md` workspace layout now reflects the 210-entry Python surface and
  the new `interest::lmm::Lmm` module.
- `docs/V2_RELEASE_AUDIT_2026-05-07.md` — full file-by-file audit.
- `publish.sh` header comments updated.

## [2.0.0-rc.0] — 2026-05-07

First release candidate. Closes the 7 P0 release blockers found by the
2026-04-28 audit (`docs/API_AUDIT_2026-04-28.md`).

### Highlights

- 9-crate workspace at edition 2024, all crates synchronised at 2.0.0-rc.0.
- New v2 surface: `Instrument` + `PricingEngine` (`AnalyticBSEngine`,
  `AnalyticHestonEngine`), reactive market cache (`Cached` / `MarketObserver`),
  FX module (`fx::delta` / `fx::smile` / Vanna-Volga), variance swap, Total
  Return Swap, Almgren-Chriss execution-cost feedback.
- `examples/full_pipeline.rs` exercises the v2 trait surface end-to-end.
- `Calibrator` / `CalibrationResult` traits Result-based with `type Params`
  and `type Error = anyhow::Error`. 9 `Calibrator` + 12 `CalibrationResult`
  implementations.
- `GreeksExt` extended with `vanna` / `charm` / `volga` / `veta`. New
  `Greeks` aggregator struct returned by `GreeksExt::greeks()`. MC pricers
  override `greeks()` for single-pass consistency (Codex stop-time review
  caught the default-aggregator pitfall on `GbmMalliavinGreeks` /
  `HestonMalliavinGreeks`).
- `DistributionExt` defaults are now `unimplemented!()` with informative
  messages (was `0.0`). 18/19 distributions provide closed-form
  characteristic functions / pdfs / cdfs / moments. Only
  `ComplexDistribution` is intentionally unimplemented (combinator).
- `PricerExt::implied_volatility` default `0.0` → `f64::NAN`.
- fBM property test `fbm_terminal_marginal_is_gaussian_with_correct_scale`
  now uses a pinned seed (`0xF_B_C0FFEE_u64`).
- Date / day-count uniformity: every pricer routes through
  `tau_or_from_dates()`; the pre-rc.0 `calculate_tau_in_days` time-unit bug
  is gone (symbol-grep clean).
- Python packaging fixed: `pyproject.toml` `manifest-path =
  "stochastic-rs-py/Cargo.toml"`; the cdylib registers 102 PyO3 classes
  spanning distributions + stochastic. (Expanded to 210 entries in rc.1.)
- Feature propagation: `gpu`, `cuda-native`, `metal`, `accelerate` now
  forward through the umbrella crate.
- `docs.rs` configuration locks features to `["openblas", "ai", "yahoo"]`
  rather than `all-features = true`.

### Breaking from beta.3

- `Calibrator::calibrate` returns `Result<Self::Output, Self::Error>` (was
  `Self::Output`).
- `CalibrationResult` now requires `type Params: Clone` and exposes
  `params() / iterations() / message() / max_error()`.
- `PricerExt::derivatives() -> Vec<f64>` is `#[deprecated(since = "2.0.0-beta.3")]`;
  use `GreeksExt::greeks() -> Greeks` instead.
- `DistributionExt` defaults panic via `unimplemented!()` rather than
  silently returning `0.0`. Custom impls relying on the default for
  `pdf` / `cdf` / `moments` need to override.

## [2.0.0-beta.3] — 2026-05-04

Section 6 architectural pass: `Instrument` + `PricingEngine`, reactive market
cache, variance swap, FX delta + Vanna-Volga smile, Total Return Swap,
end-to-end example. See `docs/API_AUDIT_2026-04-28.md` for the closure
report.

## [2.0.0-beta.2] — 2026-04-26

Workspace migration (PR #23): umbrella → 9 sibling sub-crates. Prelude
introduced. `ToModel` holes closed (13 impls). `traits.rs` split per
sub-crate. `numerics` namespace. Naming aliases (`OU`, `KOU`). Stats
`*Result` pattern uniform (15+ Result structs). Copula duplication resolved
(`samples::*Copula2D` consolidated under `BivariateExt`).

## [2.0.0-beta.1] and earlier

See `docs/API_AUDIT_2026-04.md` and `docs/WORKSPACE_MIGRATION.md` for the
beta.1 baseline and the workspace migration plan.

[2.0.0-rc.2]: https://github.com/dancixx/stochastic-rs/releases/tag/v2.0.0-rc.2
[2.0.0-rc.1]: https://github.com/dancixx/stochastic-rs/releases/tag/v2.0.0-rc.1
[2.0.0-rc.0]: https://github.com/dancixx/stochastic-rs/releases/tag/v2.0.0-rc.0
[2.0.0-beta.3]: https://github.com/dancixx/stochastic-rs/releases/tag/v2.0.0-beta.3
[2.0.0-beta.2]: https://github.com/dancixx/stochastic-rs/releases/tag/v2.0.0-beta.2
