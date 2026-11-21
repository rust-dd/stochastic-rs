use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::curves::PyDiscountCurve;
use crate::QuantError;
use crate::calibration::CalibrationOptions;
use crate::calibration::hw_swaption::HullWhiteParams;
use crate::calibration::hw_swaption::HullWhiteSwaptionCalibrator;
use crate::calibration::hw_swaption::SwaptionQuote;
use crate::calibration::tree_swaption::BlackKarasinskiParams;
use crate::calibration::tree_swaption::BlackKarasinskiSwaptionCalibrator;
use crate::calibration::tree_swaption::G2ppParams;
use crate::calibration::tree_swaption::G2ppSwaptionCalibrator;
use crate::curves::DiscountCurve;
use crate::instruments::option::types::SwaptionDirection;
use crate::traits::CalibrationResult;
use crate::traits::Calibrator;

/// `(tau, tenor, black_vol, fixed_accrual, direction)` tuples, direction
/// `"payer"` or `"receiver"`.
fn parse_quotes(quotes: Vec<(f64, f64, f64, f64, String)>) -> PyResult<Vec<SwaptionQuote>> {
  quotes
    .into_iter()
    .map(|(tau, tenor, black_vol, fixed_accrual, direction)| {
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
        tau,
        tenor,
        black_vol,
        fixed_accrual,
        direction,
        weight: None,
      })
    })
    .collect()
}

fn as_value_error(e: QuantError) -> PyErr {
  PyValueError::new_err(e.to_string())
}

/// Hull–White `(a, σ)` against a swaption grid priced by Jamshidian's
/// decomposition on the given discount curve.
#[pyclass(name = "HullWhiteSwaptionCalibrator", unsendable)]
pub struct PyHullWhiteSwaptionCalibrator {
  quotes: Vec<SwaptionQuote>,
  curve: DiscountCurve<f64>,
  notional: f64,
  start: HullWhiteParams,
}

#[pymethods]
impl PyHullWhiteSwaptionCalibrator {
  /// `quotes` are `(tau, tenor, black_vol, fixed_accrual, direction)`.
  /// `initial_rate` and `theta` are the tree model's time-0 short rate and drift offset.
  #[new]
  #[pyo3(signature = (quotes, curve, initial_rate, theta, notional=1.0))]
  fn new(
    quotes: Vec<(f64, f64, f64, f64, String)>,
    curve: &PyDiscountCurve,
    initial_rate: f64,
    theta: f64,
    notional: f64,
  ) -> PyResult<Self> {
    stochastic_rs_distributions::python::value_error_on_panic(|| -> PyResult<Self> {
      let this = Self {
        quotes: parse_quotes(quotes)?,
        curve: curve.inner.clone(),
        notional,
        start: HullWhiteParams {
          mean_reversion: 0.05,
          sigma: 0.01,
          initial_rate,
          theta,
        },
      };
      this.calibrator();
      Ok(this)
    })?
  }

  /// Returns `(mean_reversion, sigma, rmse, converged)`; `converged` is `false` at the iteration
  /// cap and for a stagnated, collapsed, flat or nonfinite simplex.
  #[pyo3(signature = (initial_guess=None))]
  fn calibrate(&self, initial_guess: Option<(f64, f64)>) -> PyResult<(f64, f64, f64, bool)> {
    let start = initial_guess.map(|(mean_reversion, sigma)| HullWhiteParams {
      mean_reversion,
      sigma,
      ..self.start
    });
    let fit = self.calibrator().calibrate(start).map_err(as_value_error)?;
    Ok((
      fit.params.mean_reversion,
      fit.params.sigma,
      fit.rmse(),
      fit.converged,
    ))
  }
}

impl PyHullWhiteSwaptionCalibrator {
  fn calibrator(&self) -> HullWhiteSwaptionCalibrator<'_> {
    HullWhiteSwaptionCalibrator::new(&self.quotes, &self.curve, self.notional, self.start)
  }
}

/// Black–Karasinski `(a, σ)` against a swaption grid repriced on the
/// log-rate trinomial tree.
#[pyclass(name = "BlackKarasinskiSwaptionCalibrator", unsendable)]
pub struct PyBlackKarasinskiSwaptionCalibrator {
  quotes: Vec<SwaptionQuote>,
  curve: DiscountCurve<f64>,
  notional: f64,
  steps_per_year: usize,
  start: BlackKarasinskiParams,
  options: CalibrationOptions,
}

#[pymethods]
impl PyBlackKarasinskiSwaptionCalibrator {
  /// `quotes` are `(tau, tenor, black_vol, fixed_accrual, direction)`;
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
    max_iters: usize,
  ) -> PyResult<Self> {
    stochastic_rs_distributions::python::value_error_on_panic(|| -> PyResult<Self> {
      let this = Self {
        quotes: parse_quotes(quotes)?,
        curve: curve.inner.clone(),
        notional,
        steps_per_year,
        start: BlackKarasinskiParams {
          mean_reversion: 0.1,
          sigma: 0.2,
          initial_rate,
          long_run_rate,
        },
        options: CalibrationOptions::default().with_max_iterations(max_iters),
      };
      this.calibrator();
      Ok(this)
    })?
  }

  /// Returns `(mean_reversion, sigma, rmse, converged)`; `converged` is `false` at the iteration
  /// cap and for a stagnated, collapsed, flat or nonfinite simplex.
  #[pyo3(signature = (initial_guess=None))]
  fn calibrate(&self, initial_guess: Option<(f64, f64)>) -> PyResult<(f64, f64, f64, bool)> {
    let start = initial_guess.map(|(mean_reversion, sigma)| BlackKarasinskiParams {
      mean_reversion,
      sigma,
      ..self.start
    });
    let fit = self.calibrator().calibrate(start).map_err(as_value_error)?;
    Ok((
      fit.params.mean_reversion,
      fit.params.sigma,
      fit.rmse(),
      fit.converged,
    ))
  }
}

impl PyBlackKarasinskiSwaptionCalibrator {
  fn calibrator(&self) -> BlackKarasinskiSwaptionCalibrator<'_> {
    BlackKarasinskiSwaptionCalibrator::new(
      &self.quotes,
      &self.curve,
      self.notional,
      self.steps_per_year,
      self.start,
    )
    .with_options(self.options.clone())
  }
}

/// G2++ `(a, b, σ, η, ρ)` against a swaption grid priced by Brigo & Mercurio's exact European.
#[pyclass(name = "G2ppSwaptionCalibrator", unsendable)]
pub struct PyG2ppSwaptionCalibrator {
  quotes: Vec<SwaptionQuote>,
  curve: DiscountCurve<f64>,
  notional: f64,
  start: G2ppParams,
  options: CalibrationOptions,
}

#[pymethods]
impl PyG2ppSwaptionCalibrator {
  /// `quotes` are `(tau, tenor, black_vol, fixed_accrual, direction)`; `initial_rate` is φ;
  /// `max_iters=None` is the fit's own cap, 1000.
  #[new]
  #[pyo3(signature = (quotes, curve, initial_rate, notional=1.0, max_iters=None))]
  fn new(
    quotes: Vec<(f64, f64, f64, f64, String)>,
    curve: &PyDiscountCurve,
    initial_rate: f64,
    notional: f64,
    max_iters: Option<usize>,
  ) -> PyResult<Self> {
    stochastic_rs_distributions::python::value_error_on_panic(|| -> PyResult<Self> {
      let this = Self {
        quotes: parse_quotes(quotes)?,
        curve: curve.inner.clone(),
        notional,
        start: G2ppParams {
          mean_reversion_x: 0.5,
          mean_reversion_y: 0.05,
          sigma_x: 0.01,
          sigma_y: 0.005,
          rho: -0.5,
          phi: initial_rate,
        },
        options: match max_iters {
          Some(cap) => CalibrationOptions::default().with_max_iterations(cap),
          None => CalibrationOptions::default(),
        },
      };
      this.calibrator();
      Ok(this)
    })?
  }

  /// Returns `(a, b, sigma, eta, rho, rmse, converged)`; `converged` is `false` at the iteration
  /// cap and for a stagnated, collapsed, flat or nonfinite simplex.
  #[pyo3(signature = (initial_guess=None))]
  fn calibrate(
    &self,
    initial_guess: Option<[f64; 5]>,
  ) -> PyResult<(f64, f64, f64, f64, f64, f64, bool)> {
    let start = initial_guess.map(|[a, b, sigma, eta, rho]| G2ppParams {
      mean_reversion_x: a,
      mean_reversion_y: b,
      sigma_x: sigma,
      sigma_y: eta,
      rho,
      ..self.start
    });
    let fit = self.calibrator().calibrate(start).map_err(as_value_error)?;
    let p = &fit.params;
    Ok((
      p.mean_reversion_x,
      p.mean_reversion_y,
      p.sigma_x,
      p.sigma_y,
      p.rho,
      fit.rmse(),
      fit.converged,
    ))
  }
}

impl PyG2ppSwaptionCalibrator {
  fn calibrator(&self) -> G2ppSwaptionCalibrator<'_> {
    G2ppSwaptionCalibrator::new(&self.quotes, &self.curve, self.notional, self.start)
      .with_options(self.options.clone())
  }
}
