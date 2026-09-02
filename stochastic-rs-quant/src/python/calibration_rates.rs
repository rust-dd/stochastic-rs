use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::curves::PyDiscountCurve;
use crate::calibration::hw_swaption::HullWhiteSwaptionCalibrator;
use crate::calibration::hw_swaption::SwaptionQuote;
use crate::calibration::tree_swaption::BlackKarasinskiSwaptionCalibrator;
use crate::calibration::tree_swaption::G2ppSwaptionCalibrator;
use crate::curves::DiscountCurve;
use crate::instruments::option::types::SwaptionDirection;
use crate::traits::Calibrator;

/// `(expiry, tenor, black_vol, fixed_accrual, direction)` tuples, direction
/// `"payer"` or `"receiver"`.
fn parse_quotes(quotes: Vec<(f64, f64, f64, f64, String)>) -> PyResult<Vec<SwaptionQuote>> {
  quotes
    .into_iter()
    .map(|(expiry, tenor, black_vol, fixed_accrual, direction)| {
      let direction = match direction.to_ascii_lowercase().as_str() {
        "payer" | "p" => SwaptionDirection::Payer,
        "receiver" | "r" => SwaptionDirection::Receiver,
        other => {
          return Err(PyValueError::new_err(format!(
            "direction must be 'payer' or 'receiver', got '{other}'"
          )));
        }
      };
      Ok(SwaptionQuote {
        expiry,
        tenor,
        black_vol,
        fixed_accrual,
        direction,
        weight: None,
      })
    })
    .collect()
}

fn as_value_error(e: anyhow::Error) -> PyErr {
  PyValueError::new_err(e.to_string())
}

/// Hull–White `(a, σ)` against a swaption grid priced by Jamshidian's
/// decomposition on the given discount curve.
#[pyclass(name = "HullWhiteSwaptionCalibrator", unsendable)]
pub struct PyHullWhiteSwaptionCalibrator {
  quotes: Vec<SwaptionQuote>,
  curve: DiscountCurve<f64>,
  notional: f64,
}

#[pymethods]
impl PyHullWhiteSwaptionCalibrator {
  /// `quotes` are `(expiry, tenor, black_vol, fixed_accrual, direction)`.
  #[new]
  #[pyo3(signature = (quotes, curve, notional=1.0))]
  fn new(
    quotes: Vec<(f64, f64, f64, f64, String)>,
    curve: &PyDiscountCurve,
    notional: f64,
  ) -> PyResult<Self> {
    Ok(Self {
      quotes: parse_quotes(quotes)?,
      curve: curve.inner.clone(),
      notional,
    })
  }

  /// Returns `(mean_reversion, sigma, rmse, converged)`.
  #[pyo3(signature = (initial_guess=None))]
  fn calibrate(&self, initial_guess: Option<(f64, f64)>) -> PyResult<(f64, f64, f64, bool)> {
    let calibrator = HullWhiteSwaptionCalibrator::new(&self.quotes, &self.curve, self.notional);
    let r = calibrator
      .calibrate(initial_guess)
      .map_err(as_value_error)?;
    Ok((r.mean_reversion, r.sigma, r.rmse, r.converged))
  }
}

/// Black–Karasinski `(a, σ)` against a swaption grid repriced on the
/// log-rate trinomial tree.
#[pyclass(name = "BlackKarasinskiSwaptionCalibrator", unsendable)]
pub struct PyBlackKarasinskiSwaptionCalibrator {
  quotes: Vec<SwaptionQuote>,
  curve: DiscountCurve<f64>,
  notional: f64,
  initial_rate: f64,
  long_run_rate: f64,
  steps_per_year: usize,
  max_iters: u64,
}

#[pymethods]
impl PyBlackKarasinskiSwaptionCalibrator {
  /// `quotes` are `(expiry, tenor, black_vol, fixed_accrual, direction)`;
  /// `steps_per_year` sets the tree resolution.
  #[new]
  #[pyo3(signature = (quotes, curve, initial_rate, long_run_rate, notional=1.0, steps_per_year=8, max_iters=400))]
  #[allow(clippy::too_many_arguments)]
  fn new(
    quotes: Vec<(f64, f64, f64, f64, String)>,
    curve: &PyDiscountCurve,
    initial_rate: f64,
    long_run_rate: f64,
    notional: f64,
    steps_per_year: usize,
    max_iters: u64,
  ) -> PyResult<Self> {
    Ok(Self {
      quotes: parse_quotes(quotes)?,
      curve: curve.inner.clone(),
      notional,
      initial_rate,
      long_run_rate,
      steps_per_year,
      max_iters,
    })
  }

  /// Returns `(mean_reversion, sigma, rmse, converged)`.
  #[pyo3(signature = (initial_guess=None))]
  fn calibrate(&self, initial_guess: Option<(f64, f64)>) -> PyResult<(f64, f64, f64, bool)> {
    let calibrator = BlackKarasinskiSwaptionCalibrator::new(
      &self.quotes,
      &self.curve,
      self.notional,
      self.initial_rate,
      self.long_run_rate,
      self.steps_per_year,
    )
    .with_max_iters(self.max_iters);
    let r = calibrator
      .calibrate(initial_guess)
      .map_err(as_value_error)?;
    Ok((r.mean_reversion, r.sigma, r.rmse, r.converged))
  }
}

/// G2++ `(a, b, σ, η, ρ)` against a swaption grid repriced on the
/// two-factor trinomial tree.
#[pyclass(name = "G2ppSwaptionCalibrator", unsendable)]
pub struct PyG2ppSwaptionCalibrator {
  quotes: Vec<SwaptionQuote>,
  curve: DiscountCurve<f64>,
  notional: f64,
  initial_rate: f64,
  steps_per_year: usize,
  max_iters: u64,
}

#[pymethods]
impl PyG2ppSwaptionCalibrator {
  /// `quotes` are `(expiry, tenor, black_vol, fixed_accrual, direction)`;
  /// the two-factor tree has `(2L + 1)²` nodes on level `L`, so keep
  /// `steps_per_year` modest.
  #[new]
  #[pyo3(signature = (quotes, curve, initial_rate, notional=1.0, steps_per_year=4, max_iters=400))]
  fn new(
    quotes: Vec<(f64, f64, f64, f64, String)>,
    curve: &PyDiscountCurve,
    initial_rate: f64,
    notional: f64,
    steps_per_year: usize,
    max_iters: u64,
  ) -> PyResult<Self> {
    Ok(Self {
      quotes: parse_quotes(quotes)?,
      curve: curve.inner.clone(),
      notional,
      initial_rate,
      steps_per_year,
      max_iters,
    })
  }

  /// Returns `(a, b, sigma, eta, rho, rmse, converged)`.
  #[pyo3(signature = (initial_guess=None))]
  fn calibrate(
    &self,
    initial_guess: Option<[f64; 5]>,
  ) -> PyResult<(f64, f64, f64, f64, f64, f64, bool)> {
    let calibrator = G2ppSwaptionCalibrator::new(
      &self.quotes,
      &self.curve,
      self.notional,
      self.initial_rate,
      self.steps_per_year,
    )
    .with_max_iters(self.max_iters);
    let r = calibrator
      .calibrate(initial_guess)
      .map_err(as_value_error)?;
    let p = &r.params;
    Ok((
      p.mean_reversion_x,
      p.mean_reversion_y,
      p.sigma_x,
      p.sigma_y,
      p.rho,
      r.rmse,
      r.converged,
    ))
  }
}
