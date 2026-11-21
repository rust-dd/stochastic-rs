---
name: calibration-pattern
description: How to implement a model calibrator in stochastic-rs (BSM, Heston, SABR, SVI, …). Invoke when adding a new calibration routine that fits a model to market option prices or implied vols.
---

# Calibration pattern — stochastic-rs

A calibrator in `stochastic-rs-quant` takes its market data in its constructor,
fits a model from a start and returns a `Calibrated<Params, D>` or a `QuantError`.
Sixteen quant calibrators share this one shape
(`stochastic-rs-quant/tests/calibration_errors.rs` asserts `Send + Sync` for each);
`ToModel` and `ToShortRateModel` bridge a result to the pricer or lattice model it
determines.

This SKILL codifies the pattern so the next calibrator is a mechanical exercise,
not a re-derivation.

## 1. The trait surface

`crate::traits::calibration` (`stochastic-rs-quant/src/traits/calibration.rs`):

```rust
pub trait Calibrator {
  type Params: Clone;
  type Output: CalibrationResult<Params = Self::Params>;
  type Error: std::fmt::Debug + std::fmt::Display + Send + Sync + 'static;

  fn calibrate(&self, initial: Option<Self::Params>) -> Result<Self::Output, Self::Error>;
}

pub trait CalibrationResult {
  type Params: Clone;

  fn rmse(&self) -> f64;
  fn converged(&self) -> bool;
  fn params(&self) -> Self::Params;
  fn loss_score(&self) -> Option<&CalibrationLossScore> { None }
  fn iterations(&self) -> Option<usize> { None }
  fn message(&self) -> Option<&str> { None }
  fn max_error(&self) -> Option<f64> { None }
  fn history(&self) -> &[CalibrationHistory<Self::Params>] { &[] }
}

pub trait ToModel {
  type Model: ModelPricer;
  fn to_model(&self) -> Self::Model;
}

// No `ModelPricer` bound: a short-rate model prices off a curve, not a spot/strike query.
pub trait ToShortRateModel {
  type Model;
  fn to_short_rate_model(&self) -> Self::Model;
}
```

`calibrate(None)` starts from the calibrator's own start; `calibrate(Some(p))`
replaces it with a value of the type the fit returns. Every quant calibrator returns
`calibration::Calibrated<P, D = ()>`, which implements `CalibrationResult`:

```rust
pub struct Calibrated<P, D = ()> {
  pub params: P,
  pub loss: CalibrationLossScore,
  pub converged: bool,
  pub iterations: Option<usize>,
  pub message: Option<String>,
  pub max_error: Option<f64>,
  pub history: Vec<CalibrationHistory<P>>,
  pub diagnostics: D,
}
```

`D` holds output beside the parameters (`SmileStrikes` for `SabrSmileCalibrator`,
the inner Heston fit for `HestonSlvCalibrator`) and is `()` everywhere else.
`CalibrationHistory<P> { iteration, params, residuals, loss }` holds the start as
entry 0 and then each accepted iterate, when the options record history; the trail
belongs to the result, never to the calibrator.

`CalibrationOptions { max_iterations, tolerance, patience, record_history,
loss_metrics, regularization }` is the knob set fifteen of the sixteen take through
`with_options`; `SabrSmileCalibrator` keeps its basin-hopping knobs
(`with_basin_hopping_iters`, `with_short_tenor_threshold`, `with_strike_bounds`,
`with_success_tol`). `max_iterations: None` keeps each optimiser's own cap (the
Levenberg–Marquardt budget `patience·(n + 1)`, Nelder–Mead 400, 600 or 1000 (G2++),
SLSQP 500, Adam 60). `regularization` is the Tikhonov pull of §4.

A fit honours every knob it receives or, once the knob is set away from its default,
answers `Err(QuantError::Unsupported { what })`: it names the knobs it cannot read to
`CalibrationOptions::check`, which also answers a knob written through its field
outside its builder's domain with `InvalidInput { what: "options" }`
(`options.lm(..)` calls it for every Levenberg–Marquardt fit).

`CalibrationLossScore` (in `crate::types`) is a HashMap of `LossMetric → f64`.
`LossMetric` has **nine** variants — `Mae, Mse, Rmse, Mpe, Mape, Mspe, Rmspe, Mre,
Mrpe` — each computed by its function in `crate::loss` (`loss::mae(market, model)`,
…). `get(metric)` returns `f64::NAN` for a missing entry, never 0.0: a silent zero would read as a
perfect fit.

## 2. The files of a calibrator

For a new `XyzCalibrator`, you typically touch:

```
stochastic-rs-quant/src/calibration/xyz.rs    -- the calibrator and its problem
stochastic-rs-quant/src/pricing/xyz.rs        -- the model it fits (already exists)
stochastic-rs-quant/src/python/               -- PyXyzCalibrator (python-bindings SKILL)
stochastic-rs-py/src/lib.rs                   -- m.add_class registration
```

Plus `pub mod xyz` and its re-exports in `stochastic-rs-quant/src/calibration.rs`.

## 3. The calibrator skeleton

A quote calibrator on the Levenberg–Marquardt core:

```rust
// stochastic-rs-quant/src/calibration/xyz.rs
use ndarray::Array1;
use ndarray::Array2;
use ndarray::ArrayView1;

use crate::OptionQuote;
use crate::QuantError;
use crate::calibration::Calibrated;
use crate::calibration::CalibrationOptions;
use crate::calibration::least_squares::LeastSquaresProblem;
use crate::calibration::least_squares::Trail;
use crate::calibration::least_squares::minimize;
use crate::calibration::quote::assert_spot_and_rates;
use crate::calibration::quote::validate_quotes;
use crate::pricing::xyz::XyzModel;
use crate::pricing::xyz::XyzParams;
use crate::traits::ModelPricer;
use crate::traits::ToModel;

impl ToModel for Calibrated<XyzParams> {
  type Model = XyzModel;

  fn to_model(&self) -> XyzModel {
    XyzModel::new(self.params)
  }
}

/// Xyz fit to vanilla quotes by Levenberg–Marquardt.
#[derive(Clone, Debug)]
pub struct XyzCalibrator {
  s: f64,
  r: f64,
  q: f64,
  pub quotes: Vec<OptionQuote>,
  /// `None` starts from (0.1, 0.5, 0.3).
  pub start: Option<XyzParams>,
  /// Levenberg–Marquardt knobs; `regularization` pulls `(a, b, c)`.
  pub options: CalibrationOptions,
}

impl XyzCalibrator {
  /// Panics unless `s` is finite and positive and `r`, `q` finite; `calibrate` checks the quotes.
  pub fn new(s: f64, r: f64, q: f64, quotes: Vec<OptionQuote>) -> Self {
    assert_spot_and_rates(s, r, q);
    Self { s, r, q, quotes, start: None, options: CalibrationOptions::default() }
  }

  pub fn s(&self) -> f64 {
    self.s
  }

  pub fn r(&self) -> f64 {
    self.r
  }

  pub fn q(&self) -> f64 {
    self.q
  }

  /// Panics unless `start` passes `XyzParams::new`.
  pub fn with_start(mut self, start: XyzParams) -> Self {
    self.start = Some(XyzParams::new(start.a, start.b, start.c));
    self
  }

  pub fn with_options(mut self, options: CalibrationOptions) -> Self {
    self.options = options;
    self
  }
}

impl crate::traits::Calibrator for XyzCalibrator {
  type Params = XyzParams;
  type Output = Calibrated<XyzParams>;
  type Error = QuantError;

  /// A start replaces `self.start`; `InvalidInput` unless the one used passes `XyzParams::new`.
  fn calibrate(&self, initial: Option<XyzParams>) -> Result<Calibrated<XyzParams>, QuantError> {
    validate_quotes(&self.quotes)?;
    let start = initial.or(self.start).unwrap_or(XyzParams { a: 0.1, b: 0.5, c: 0.3 });
    if let Some(reason) = start.violation() {
      return Err(QuantError::invalid("start", reason));
    }
    let lm = self.options.lm(true, 3)?;
    let (fitted, report) = minimize(XyzProblem::new(self, start), lm);
    let model = fitted.model_prices();
    let trail = fitted.trail.into_entries();
    report.calibrated(fitted.params, &fitted.market, &model, self.options.loss_metrics, trail)
  }
}

struct XyzProblem<'a> {
  calibrator: &'a XyzCalibrator,
  params: XyzParams,
  market: Vec<f64>,
  trail: Trail<XyzParams>,
}

impl<'a> XyzProblem<'a> {
  fn new(calibrator: &'a XyzCalibrator, params: XyzParams) -> Self {
    let market = calibrator.quotes.iter().map(|quote| quote.price).collect();
    Self { calibrator, params, market, trail: Trail::new() }
  }

  fn model_prices(&self) -> Vec<f64> {
    let XyzCalibrator { s, r, q, .. } = *self.calibrator;
    let model = XyzModel::new(self.params);
    let price = |o: &OptionQuote| model.price_option(s, o.strike, r, q, o.tau, o.option_type);
    self.calibrator.quotes.iter().map(price).collect()
  }
}

impl LeastSquaresProblem for XyzProblem<'_> {
  fn set_params(&mut self, x: ArrayView1<f64>) {
    self.params = XyzParams { a: x[0], b: x[1], c: x[2] };
  }

  fn params(&self) -> Array1<f64> {
    Array1::from_vec(vec![self.params.a, self.params.b, self.params.c])
  }

  fn residuals(&mut self) -> Option<Array1<f64>> {
    let model = self.model_prices();
    let residuals = Array1::from_iter(self.market.iter().zip(&model).map(|(m, p)| m - p));
    let options = &self.calibrator.options;
    self.trail.stage(options, &self.market, &model, &self.params, residuals.view());
    Some(residuals)
  }

  fn jacobian(&self) -> Option<Array2<f64>> {
    todo!("analytic, or central differences of `residuals`")
  }

  fn accept(&mut self) {
    self.trail.accept();
  }
}
```

`report.calibrated` is the one Levenberg–Marquardt tail: it reprices for `loss` and
`max_error`, takes `iterations` and `message` from the stop and answers a start the
optimiser could not evaluate with `Err(QuantError::Optimizer)`. Another optimiser
calls `options.check(&[..])` with the knobs it cannot read and builds the result
itself, `Calibrated::new(params, loss, converged)` with `.with_iterations(..)`,
`.with_max_error(..)`, `.with_message(..)` and `.with_history(..)`; its `converged`
is a stop on the optimiser's own criterion, never a cap, and `message` names the
stop.

The spot and rates are private: `new` checks them and `s()`, `r()` and `q()` read them
back, so no fit runs at a spot `new` did not check; to fit elsewhere, build another
calibrator. Where the problem lives in a sibling module, make them `pub(super)`.

Field names follow the crate-wide vocabulary: `s` spot, `k` or `strike`, `tau` time
to maturity in years, `r` and `q` the rates (`r_d`/`r_f` only where two rates
genuinely exist); never `spot`, `s0`, `maturity` or `risk_free`. A quote is the one
`crate::OptionQuote { strike, tau, price, option_type }`, priced as its own
`option_type`.

## 4. Optimizer choices

`stochastic-rs` uses **five** optimizers depending on the problem shape. Check the
file you are copying from rather than assuming.

- **`basin` Levenberg–Marquardt through `calibration::least_squares`** — the
  workhorse for a least-squares residual fit. Implement `LeastSquaresProblem`
  (`set_params` from an `ArrayView1<f64>`, then `params` / `residuals` / `jacobian` /
  `accept` over `ndarray::Array1<f64>` and `Array2<f64>`) and drive it with
  `least_squares::minimize(problem, options.lm(pivoted_qr, n)?)`. Used by
  `BsmCalibrator`, `HestonCalibrator` (`calibration/heston/{calibrator,lsq}.rs`),
  `SabrCalibrator`, `SvjCalibrator`, `DoubleHestonCalibrator`, `HkdeCalibrator`,
  `LevyCalibrator`, `CgmysvCalibrator`, and both vol-surface fits
  (`vol_surface/svi.rs`, `vol_surface/ssvi/calibrate.rs`), which pass their own
  `LmOptions`. Two wrappers drive the same core: Heston and CGMYSV call
  `reseat::minimize_reseated`, which restarts a fit stopped on a saturated logistic
  coordinate, and the boxed fits (HKDE, SVJ, Double Heston and Lévy) call
  `least_squares::minimize_in_box`, which holds a coordinate on a bound its step
  reaches when the cost falls across it.
  - **The pull.** The core appends `CalibrationOptions::regularization`'s rows
    `√λⱼ(θⱼ − θ⁰ⱼ)` and their Jacobian to every fit; `options.lm(pivoted_qr, n)`
    answers a pull of another length with `DimensionMismatch`. A problem whose
    coordinates are not its parameters in their natural order overrides
    `natural_params` and `natural_jacobian` (`∂θ/∂x`): Heston and CGMYSV for their
    logistic coordinates, SABR for its `(α, ν, ρ)` order. Name the order on the
    calibrator's `options` field.
- **`basin` Nelder–Mead** — derivative-free simplex fits: `calibration/hw_swaption.rs`,
  `calibration/tree_swaption.rs`, `calibration/sabr_caplet.rs`. Reuse
  `calibration::nelder_mead_fit`, which keeps the sample-standard-deviation stop,
  records the best vertex per iteration, refuses `patience` and builds the
  `Calibrated`; the pull is a cost penalty there (`Regularization::penalty`). Bounded
  objectives and finite differences clamp their inputs: do not rely on the solver to
  keep iterates feasible.
- **`basin` L-BFGS-B under basin-hopping** — a bounded gradient fit:
  `SabrSmileCalibrator` (`vol_surface/sabr_smile/objective.rs`), on its own
  basin-hopping knobs rather than `CalibrationOptions`.
- **`slsqp` crate** — explicit bounds and constraints. One in-tree user:
  `HscmCalibrator` (`calibration/heston_stoch_corr.rs`), whose `tolerance` is SLSQP's
  `ftol_abs` and whose `converged` holds only for `Success`, `FtolReached`,
  `XtolReached` or `StopValReached`.
- **A hand-written routine** — a loss that is not a sum of squares.
  `RBergomiCalibrator` (`calibration/rbergomi/calibrator.rs`) runs projected Adam
  on the mean empirical Wasserstein-1 distance (`super::loss::empirical_wasserstein_1`).

HSCM, rBergomi and Heston SLV answer a pull with
`Err(QuantError::Unsupported { what: "regularization" })`, as every fit answers a knob
it cannot read (§1).

**Do not** add another optimizer crate; `basin`, `slsqp` and the hand-written route
above cover everything we have needed.

## 5. The `Result` contract (dev-rules §14)

1. **A broken constructor or builder argument panics**, the predicate in the
   message and the value last: `assert_spot_and_rates(s, r, q)` in `new`,
   `with_start` through the model's checked constructor.
2. **Market data is `Err(QuantError::…)`**: `validate_quotes` gives
   `InsufficientData` for no quote and `InvalidQuote { index }` for the first bad
   one; a start passed to `calibrate` or written through the `start` field outside
   the model's domain is `InvalidInput { what: "start" }`; a start the optimiser
   cannot evaluate is `Optimizer`; a knob is answered as §1 says.
3. **Non-convergence is `Ok`** with `converged: false` and, where the optimiser
   reports one, the stop reason in `message`. A cap is never a convergence.
4. **A model price that is not finite stays not finite**:
   `calibration::clip_negative` floors a finite negative price at 0 and passes NaN
   and ±∞ through, so the optimiser rejects the trial. Never floor with `f64::max`,
   which turns a NaN into a plausible 0.

## 6. `ToModel` / `ToShortRateModel` bridges

Implement `ToModel for Calibrated<XyzParams>` when the parameters determine a
`ModelPricer`; `build_surface_from_calibration` then works for free. A short-rate
calibrator (Hull–White, Black–Karasinski, G2++) implements `ToShortRateModel`, with
the curve-side inputs riding in its `Params`.

## 7. Testing requirements

A new calibrator ships with:

```rust
#[test]
fn round_trip_recovery() {
  // Price quotes from known parameters, calibrate, recover them within the fit's tolerance.
}

#[test]
fn optimizer_actually_runs() {
  // The trail's first loss exceeds the final one, so a discarded `minimize` cannot pass.
}
```

Then add the calibrator to `stochastic-rs-quant/tests/calibration_errors.rs` (its
`TABLE` row answers no quote, a NaN price, a non-finite start, a negative spot and
a carry no model can price; its `Send + Sync` list) and, for a Levenberg–Marquardt
fit, a pull pin to `stochastic-rs-quant/tests/calibration_regularization.rs`.

The progress test is the most important: `let _ = slsqp::minimize(...)` once
compiled fine while a test asserting a non-empty output passed and the calibrator
was a no-op.

## 8. Python wrapper (if exposing to Python)

Follow the `python-bindings` SKILL:

```rust
#[pyclass(name = "XyzCalibrator")]
pub struct PyXyzCalibrator {
  inner: crate::calibration::xyz::XyzCalibrator,
}

#[pymethods]
impl PyXyzCalibrator {
  #[new]
  fn new(s: f64, r: f64, q: f64, quotes: Vec<(f64, f64, f64)>) -> PyResult<Self> {
    stochastic_rs_distributions::python::value_error_on_panic(|| {
      let quote = |(k, tau, price)| OptionQuote::new(k, tau, price, OptionType::Call);
      let quotes = quotes.into_iter().map(quote).collect();
      Self { inner: crate::calibration::xyz::XyzCalibrator::new(s, r, q, quotes) }
    })
  }

  /// Returns `(a, b, c, rmse, converged)`.
  fn calibrate(&self) -> PyResult<(f64, f64, f64, f64, bool)> {
    use crate::traits::{CalibrationResult, Calibrator};
    let fit = self.inner.calibrate(None)
      .map_err(|e| PyValueError::new_err(format!("XYZ calibration failed: {e}")))?;
    Ok((fit.params.a, fit.params.b, fit.params.c, fit.rmse(), fit.converged))
  }
}
```

`value_error_on_panic` turns a constructor panic into `ValueError`, and a
`QuantError` reaches Python as `ValueError` carrying its `Display` text. Register
the class in `stochastic-rs-py/src/lib.rs` with `m.add_class::<PyXyzCalibrator>()?`.

## 9. Anti-patterns

- **Do not** discard optimizer output (`let _ = optimizer::minimize(...)`); see the
  progress test in §7.
- **Do not** return `Err` for a fit that did not converge; report
  `converged: false`.
- **Return `QuantError`** — never `anyhow`, `String` or a model-specific error enum.
- **Do not** return a model from `calibrate`. Return the `Calibrated`, so callers
  can judge the fit, and convert with `ToModel`.
- **Do not** keep the trail or any per-call state on the calibrator
  (`Rc<RefCell<…>>` made calibrators `!Send`); the fit's problem owns it and hands
  it to the result.
- **Do not** silently fall back to the start on failure. Either answer `Err` or
  report `converged: false` so downstream knows.
- **Do not** leave a `CalibrationOptions` knob unread: honour it or name it to
  `CalibrationOptions::check`, which answers it with `Unsupported`.

## 10. Reference impls

`heston`, `rbergomi`, `hkde`, `levy`, `double_heston`, `svj` and `heston_slv` are
**directories** under `stochastic-rs-quant/src/calibration/`; `bsm`,
`heston_stoch_corr`, `cgmysv`, `sabr`, `sabr_caplet`, `hw_swaption` and
`tree_swaption` are files with a sibling directory for their tests where they have
one (`tree_swaption/` also holds `g2pp_european`, the G2++ European swaption). In
increasing complexity:

- `BsmCalibrator` (`calibration/bsm.rs`) — the simplest full example: one
  volatility, vega-weighted residuals, on `least_squares::minimize`.
- `SabrCalibrator` (`calibration/sabr.rs`) — a reordered `natural_jacobian`.
- `HestonCalibrator` (`calibration/heston/calibrator.rs`, the problem in
  `calibration/heston/lsq.rs`) — five parameters in logistic coordinates, Cui et
  al.'s analytic Jacobian, the Feller projection and an MLE seed.
- `HkdeCalibrator` (`calibration/hkde/calibrator.rs`) — Heston with Kou
  double-exponential jumps, residuals weighted by inverse Black vega (Agazzotti et
  al. 2025, eq. 13).
- `HscmCalibrator` (`calibration/heston_stoch_corr.rs`) — the crate's only SLSQP
  user, with bounds.
- `RBergomiCalibrator` (`calibration/rbergomi/calibrator.rs`) — projected Adam on a
  Wasserstein loss, with `q` a constructor argument.
- `HestonSlvCalibrator` (`calibration/heston_slv/calibrator.rs`) — an inner Heston
  fit, a Dupire local volatility and a leverage surface; `Params` is the whole
  `HestonSlvFit`.

**SVI / SSVI are not `Calibrator` implementors.** The Rust entry points are free
functions — `calibrate_svi` (`vol_surface/svi.rs`) and `calibrate_ssvi`
(`vol_surface/ssvi/calibrate.rs`, whose LM problem is `SsviLmProblem` with an
analytic `jacobian`). The names `SviCalibrator` / `SsviCalibrator` exist only as
`#[pyclass(name = ...)]` labels on the PyO3 wrappers in
`stochastic-rs-quant/src/python/vol_surface.rs`.

## Related SKILLs

- `python-bindings` — for the PyXyz wrapper layer.
- `release-checklist` — `MIGRATION.md` notes a new calibrator's parameter
  conventions in the breaking-change list.
- `stats-estimator` — sister pattern for statistical estimators (a similar
  `Result` shape, but no `ToModel` bridge).
