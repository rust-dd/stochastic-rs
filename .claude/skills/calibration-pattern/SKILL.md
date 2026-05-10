---
name: calibration-pattern
description: How to implement a model calibrator in stochastic-rs (BSM, Heston, SABR, SVI, …). Invoke when adding a new calibration routine that fits a model to market option prices or implied vols.
---

# Calibration pattern — stochastic-rs

A "calibrator" in `stochastic-rs-quant` consumes market option data (or
implied-vol slices) plus an initial guess and returns either the fitted
parameters + a converged-status flag, or a typed error. Twelve
calibrators currently follow this pattern (post rc.1's three new ones:
`HscmCalibrator`, `CgmysvCalibrator`, `HKDECalibrator`); thirteen
`ToModel` impls bridge the calibrator output back to a `ModelPricer`.

This SKILL codifies the pattern so adding a calibrator #13 is a
mechanical exercise, not a re-derivation.

## 1. The trait surface

Three traits in `crate::traits::calibration` (see
`stochastic-rs-quant/src/traits/calibration.rs`):

```rust
pub trait Calibrator {
    type InitialGuess;                       // typically [f64; N] or Option<[f64; N]>
    type Params;                             // the calibrated parameter struct
    type Output: CalibrationResult<Params = Self::Params>;
    type Error;                              // anyhow::Error in production code

    fn calibrate(
        &self,
        initial: Option<Self::InitialGuess>,
    ) -> Result<Self::Output, Self::Error>;
}

pub trait CalibrationResult {
    type Params;

    fn rmse(&self) -> f64;
    fn converged(&self) -> bool;
    fn params(&self) -> Self::Params;

    // Optional richer breakdown (provided default returns None):
    fn loss_score(&self) -> Option<&CalibrationLossScore> { None }
    fn iterations(&self) -> Option<usize> { None }
    fn message(&self) -> Option<&str> { None }
    fn max_error(&self) -> Option<f64> { None }
}

pub trait ToModel {
    type Model;
    fn to_model(&self, r: f64, q: f64) -> Self::Model;
}
```

`CalibrationLossScore` (in `crate::types`) is a HashMap of `LossMetric →
f64` covering rmse / mae / mape / mspe / mre. The rc.2 fix made
`get(metric)` return `f64::NAN` for missing entries (was 0.0). See the
rc.2 CHANGELOG entry on `CalibrationLossScore`.

## 2. The four files of a calibrator

For a new `XyzCalibrator`, you typically touch:

```
stochastic-rs-quant/src/calibration/xyz.rs    -- the calibrator itself
stochastic-rs-quant/src/pricing/xyz.rs        -- the underlying pricer / model (already exists)
stochastic-rs-quant/src/python.rs             -- PyXyzCalibrator wrapper (see python-bindings SKILL)
stochastic-rs-py/src/lib.rs                   -- m.add_class registration
```

Plus one of:

```
stochastic-rs-quant/src/lib.rs           -- pub use export
stochastic-rs-quant/src/calibration.rs   -- pub mod xyz
```

## 3. The minimal calibrator skeleton

```rust
// stochastic-rs-quant/src/calibration/xyz.rs

use crate::pricing::xyz::XyzModel;

/// Market option for XYZ calibration.
#[derive(Clone, Debug)]
pub struct MarketOption {
    pub strike: f64,
    pub maturity: f64,
    pub price: f64,
    pub rate: f64,
}

/// Calibrated parameter set.
#[derive(Clone, Debug)]
pub struct XyzParams {
    pub a: f64,
    pub b: f64,
    pub c: f64,
}

/// Calibration result. Embeds rmse / converged / final_objective; the
/// CalibrationResult impl derives the trait-level fields from these.
#[derive(Clone, Debug)]
pub struct XyzCalibrationResult {
    pub a: f64,
    pub b: f64,
    pub c: f64,
    pub rmse: f64,
    pub converged: bool,
    pub final_objective: f64,
}

impl crate::traits::CalibrationResult for XyzCalibrationResult {
    type Params = XyzParams;
    fn rmse(&self) -> f64 { self.rmse }
    fn converged(&self) -> bool { self.converged && self.rmse.is_finite() }
    fn params(&self) -> Self::Params {
        XyzParams { a: self.a, b: self.b, c: self.c }
    }
}

impl crate::traits::ToModel for XyzCalibrationResult {
    type Model = XyzModel;
    fn to_model(&self, _r: f64, _q: f64) -> XyzModel {
        XyzModel { a: self.a, b: self.b, c: self.c }
    }
}

/// The calibrator. Stateful struct so generic pipelines can consume it.
#[derive(Clone, Debug)]
pub struct XyzCalibrator {
    pub s0: f64,
    pub options: Vec<MarketOption>,
    pub max_iter: usize,
}

impl XyzCalibrator {
    pub fn new(s0: f64, options: Vec<MarketOption>) -> Self {
        Self { s0, options, max_iter: 500 }
    }
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }
}

impl crate::traits::Calibrator for XyzCalibrator {
    type InitialGuess = [f64; 3];
    type Params = XyzParams;
    type Output = XyzCalibrationResult;
    type Error = anyhow::Error;

    fn calibrate(
        &self,
        initial: Option<Self::InitialGuess>,
    ) -> Result<Self::Output, Self::Error> {
        let guess = initial.unwrap_or([0.1, 0.5, 0.3]);
        // ... call slsqp / argmin / your optimizer of choice ...
        Ok(XyzCalibrationResult { /* ... */ })
    }
}
```

## 4. Optimizer choices

`stochastic-rs` uses three optimizers depending on the problem shape:

- **`slsqp` crate** — when there are explicit `[f64; N]` bounds and you
  need constraints. Used by `HscmCalibrator`, `RBergomiCalibrator`. The
  `(f, init, &bounds, cons, data)` calling convention is uniform.
- **`argmin` (LBFGS / Newton-CG / NelderMead)** — when the problem is
  unconstrained or when you want a pluggable line search. Used by
  `SsviCalibrator`, `SviCalibrator` (Levenberg-Marquardt).
- **Custom Levenberg-Marquardt via `nalgebra`** — when the residuals
  function is closed-form and you want analytic Jacobian. Used by
  `SsviLmProblem` (rc.2 added the analytic Jacobian — see
  `vol_surface/ssvi.rs::SsviLmProblem::jacobian`).

**Do not** add a new optimizer crate; the three above are sufficient
for everything we've needed. Adding a 4th adds compile time without
new capability.

## 5. The `Result<Output, Error>` contract

Calibrators **must** return `Result`. The three failure modes:

1. **Input validation:** invalid parameters (e.g. negative volatility
   in initial guess, empty option set, mismatched maturity grid).
   Return `Err(anyhow::anyhow!("..."))` early.

2. **Optimizer non-convergence:** the optimizer ran out of iterations
   or hit a numeric stop. **Do not** return `Err` for this; instead
   return `Ok(result)` with `result.converged = false`. Callers can
   inspect this via `CalibrationResult::converged()`.

3. **Catastrophic numeric failure:** NaN/Inf in the objective, matrix
   singularity in a derived calculation. Return `Err(anyhow::anyhow!)`
   so the user is forced to handle it.

The split between (2) and (3) is important: a converged-but-poor fit
is informative ("calibrator can't match this slice"), while NaN
contamination silently propagates. Don't blur the boundary.

## 6. Optional: `ToModel` / `ToShortRateModel` bridges

If your calibration result has a 1-to-1 mapping to a model that
implements `ModelPricer`, add `impl ToModel for XyzCalibrationResult`.
Generic vol-surface routines like `build_surface_from_calibration` then
work for free.

For interest-rate calibrators (HW1F, G2++, SABR-caplet, …) where the
output is a *short-rate* model that prices bonds/swaps rather than
options, implement `ToShortRateModel` instead. The two traits don't
conflict; some calibrators implement both.

## 7. Testing requirements

A new calibrator must ship with at least three tests:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    /// 1. Construction validates inputs (empty options should error).
    #[test]
    fn rejects_empty_options() {
        let cal = XyzCalibrator::new(100.0, vec![]);
        assert!(cal.calibrate(None).is_err());
    }

    /// 2. Round-trip: synthesise market data from a known parameter set,
    ///    calibrate, recover the parameters within RMSE < 1e-3.
    #[test]
    fn round_trip_recovery() {
        // ...
    }

    /// 3. final_objective < initial_sse * 0.5 — confirms the optimizer
    ///    actually moved away from the initial guess. This catches the
    ///    rc.0 trap where `let _ = slsqp::minimize(...)` silently
    ///    discarded the optimizer output and `XyzCalibrationResult`
    ///    returned the initial guess as if calibrated.
    #[test]
    fn optimizer_actually_runs() {
        // ...
    }
}
```

The third test is the most important. It is *exactly* the test that
caught the rc.0 HSCM issue — `let _ = slsqp::minimize(...)` compiled
fine, the test that asserted "calibrator output is non-empty" passed,
but the calibrator was a no-op. The progress test makes this category
of bug impossible to hide.

## 8. Python wrapper (if exposing to Python)

Follow the `python-bindings` SKILL. The standard wrapper:

```rust
// stochastic-rs-quant/src/python.rs

#[pyclass(name = "XyzCalibrator", unsendable)]
pub struct PyXyzCalibrator {
    inner: crate::calibration::xyz::XyzCalibrator,
}

#[pymethods]
impl PyXyzCalibrator {
    #[new]
    #[pyo3(signature = (s0, options, max_iter=500))]
    fn new(s0: f64, options: Vec<PyXyzMarketOption>, max_iter: usize) -> Self {
        let inner_options = options.into_iter().map(|o| o.inner).collect();
        Self {
            inner: crate::calibration::xyz::XyzCalibrator::new(s0, inner_options)
                .with_max_iter(max_iter),
        }
    }

    /// Returns `(a, b, c, rmse, converged)`.
    #[pyo3(signature = (initial=None))]
    fn calibrate(&self, initial: Option<[f64; 3]>) -> PyResult<(f64, f64, f64, f64, bool)> {
        use crate::traits::Calibrator;
        let res = self.inner.calibrate(initial)
            .map_err(|e| PyValueError::new_err(format!("XYZ calibration failed: {e}")))?;
        Ok((res.a, res.b, res.c, res.rmse, res.converged))
    }

    /// Calibrate and return the resulting `XyzModel` for direct pricing.
    #[pyo3(signature = (initial=None))]
    fn calibrate_to_model(&self, initial: Option<[f64; 3]>) -> PyResult<PyXyzModel> {
        use crate::traits::Calibrator;
        let res = self.inner.calibrate(initial)
            .map_err(|e| PyValueError::new_err(format!("XYZ calibration failed: {e}")))?;
        Ok(PyXyzModel { inner: res.to_model(0.0, 0.0) })
    }
}
```

Then register in `stochastic-rs-py/src/lib.rs`:

```rust
use stochastic_rs_quant::python::PyXyzCalibrator;
// ...
m.add_class::<PyXyzCalibrator>()?;
```

## 9. Anti-patterns

- **Do not** discard optimizer output (`let _ = optimizer::minimize(...)`).
  See section 7 test #3 for why.
- **Do not** return `Err` on a non-converged optimizer; use
  `converged: false`. Reserve `Err` for catastrophic failure.
- **Do not** invent a custom error type when `anyhow::Error` works.
  Calibrators are user-facing surfaces; `anyhow` is the project default
  (per `Calibrator::Error = anyhow::Error` convention).
- **Do not** return a model from `calibrate()` directly. Always return
  the `CalibrationResult` (with rmse + converged) so callers can decide
  whether to trust the fit. Use `ToModel` for the conversion.
- **Do not** silently fall back to the initial guess on failure. Either
  fail loudly (Err) or report `converged: false` so downstream knows.

## 10. Reference impls

When in doubt, copy the pattern from one of these (in increasing
complexity):

- `BSMCalibrator` (`calibration/bsm.rs`) — single-parameter, closed-form,
  no optimizer.
- `HestonCalibrator` (`calibration/heston.rs`) — 5-parameter LM with Cui
  Jacobian.
- `SsviCalibrator` (`vol_surface/ssvi.rs::SsviLmProblem`) — 3-parameter
  LM with closed-form analytic Jacobian (rc.2).
- `HscmCalibrator` (`calibration/heston_stoch_corr.rs`) — 9-parameter
  SLSQP with bounds.
- `RBergomiCalibrator` (`calibration/rbergomi.rs`) — multi-stage
  (terminal samples + per-slice fit) with `with_dividend_yield(q)`
  builder.

## Related SKILLs

- `python-bindings` — for the PyXyz wrapper layer.
- `release-checklist` — CHANGELOG should note any new calibrator's
  parameter conventions in the breaking-change list.
- `stats-estimator` — sister pattern for statistical estimators (similar
  Result<XxxResult, _> shape, but no `ToModel` bridge).
