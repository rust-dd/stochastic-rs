use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::calibration::HscmParams;
use crate::traits::ModelPricer;

fn hscm_start(guess: [f64; 9]) -> HscmParams {
  let [
    kappa_v,
    theta_v,
    sigma_v,
    v0,
    kappa_r,
    mu_r,
    sigma_r,
    rho0,
    rho2,
  ] = guess;
  HscmParams {
    kappa_v,
    theta_v,
    sigma_v,
    v0,
    kappa_r,
    mu_r,
    sigma_r,
    rho0,
    rho2,
  }
}

/// Heston-stochastic-correlation model wrapper. Implements the Carr-Madan
/// pricer through `ModelPricer`. 9 parameters: `(v0, kappa_v, theta_v,
/// sigma_v, rho0, kappa_r, mu_r, sigma_r, rho2)`.
///
/// The Python class name is unchanged; the Rust type it wraps is now
/// `HestonStochCorrPricer`, which absorbed the former `HscmModel`.
#[pyclass(name = "HscmModel", from_py_object, unsendable)]
#[derive(Clone)]
pub struct PyHscmModel {
  pub inner: crate::pricing::heston_stoch_corr::HestonStochCorrPricer,
}

#[pymethods]
impl PyHscmModel {
  #[new]
  #[allow(clippy::too_many_arguments)]
  fn new(
    v0: f64,
    kappa_v: f64,
    theta_v: f64,
    sigma_v: f64,
    rho0: f64,
    kappa_r: f64,
    mu_r: f64,
    sigma_r: f64,
    rho2: f64,
  ) -> PyResult<Self> {
    stochastic_rs_distributions::python::value_error_on_panic(|| -> PyResult<Self> {
      if v0 <= 0.0 || theta_v <= 0.0 || sigma_v <= 0.0 || sigma_r <= 0.0 {
        return Err(PyValueError::new_err(
          "v0, theta_v, sigma_v, sigma_r must be > 0",
        ));
      }
      if rho0.abs() >= 1.0 || rho2.abs() >= 1.0 || mu_r.abs() >= 1.0 {
        return Err(PyValueError::new_err("|rho0|, |rho2|, |mu_r| must be < 1"));
      }
      Ok(Self {
        inner: crate::pricing::heston_stoch_corr::HestonStochCorrPricer {
          v0,
          kappa_v,
          theta_v,
          sigma_v,
          rho0,
          kappa_r,
          mu_r,
          sigma_r,
          rho2,
        },
      })
    })?
  }

  /// Carr-Madan FFT call price.
  fn price_call(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    self.inner.price_call(s, k, r, q, tau)
  }
}

/// Market option for HSCM calibration: strike, maturity, observed call price and the rate it was
/// quoted at, which every option given to one `HscmCalibrator` shares.
#[pyclass(name = "HscmMarketOption", from_py_object, unsendable)]
#[derive(Clone)]
pub struct PyHscmMarketOption {
  pub inner: crate::OptionQuote,
  pub rate: f64,
}

#[pymethods]
impl PyHscmMarketOption {
  #[new]
  fn new(strike: f64, maturity: f64, price: f64, rate: f64) -> Self {
    Self {
      inner: crate::OptionQuote::new(strike, maturity, price, crate::OptionType::Call),
      rate,
    }
  }

  #[getter]
  fn strike(&self) -> f64 {
    self.inner.strike
  }

  #[getter]
  fn maturity(&self) -> f64 {
    self.inner.tau
  }

  #[getter]
  fn price(&self) -> f64 {
    self.inner.price
  }

  #[getter]
  fn rate(&self) -> f64 {
    self.rate
  }
}

/// Heston-stochastic-correlation calibrator (SLSQP, 9 parameters).
/// Mirrors the Rust `Calibrator` trait surface — pass `initial_guess` to
/// override the default mild-skew SPX-style start.
#[pyclass(name = "HscmCalibrator", unsendable)]
pub struct PyHscmCalibrator {
  inner: crate::calibration::heston_stoch_corr::HscmCalibrator,
}

#[pymethods]
impl PyHscmCalibrator {
  /// `max_iter` caps SLSQP's evaluations; every option must carry the same `rate`.
  #[new]
  #[pyo3(signature = (s0, options, max_iter=500))]
  fn new(s0: f64, options: Vec<PyHscmMarketOption>, max_iter: usize) -> pyo3::PyResult<Self> {
    stochastic_rs_distributions::python::value_error_on_panic(|| -> PyResult<Self> {
      use crate::calibration::CalibrationOptions;
      use crate::calibration::heston_stoch_corr::HscmCalibrator;
      let rate = options.first().map_or(0.0, |o| o.rate);
      let quotes = options.iter().map(|o| o.inner).collect();
      let settings = CalibrationOptions::default().with_max_iterations(max_iter);
      let inner = HscmCalibrator::new(s0, rate, 0.0, quotes).with_options(settings);
      if let Some(other) = options.iter().find(|o| o.rate != rate) {
        return Err(PyValueError::new_err(format!(
          "HSCM quotes must share one rate; got {rate} and {}",
          other.rate
        )));
      }
      Ok(Self { inner })
    })?
  }

  /// Returns `(kappa_v, theta_v, sigma_v, v0, kappa_r, mu_r, sigma_r, rho0,
  /// rho2, rmse, converged)`. The 9-vector matches `HscmCalibrator::calibrate`'s
  /// internal parameter ordering (NOT the `HscmModel` constructor order).
  #[pyo3(signature = (initial_guess=None))]
  #[allow(clippy::type_complexity)]
  fn calibrate(
    &self,
    initial_guess: Option<[f64; 9]>,
  ) -> PyResult<(f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, bool)> {
    use crate::traits::CalibrationResult;
    use crate::traits::Calibrator;
    let fit = self
      .inner
      .calibrate(initial_guess.map(hscm_start))
      .map_err(|e| PyValueError::new_err(format!("HSCM calibration failed: {e}")))?;
    let p = &fit.params;
    Ok((
      p.kappa_v,
      p.theta_v,
      p.sigma_v,
      p.v0,
      p.kappa_r,
      p.mu_r,
      p.sigma_r,
      p.rho0,
      p.rho2,
      fit.rmse(),
      fit.converged,
    ))
  }

  /// Calibrate then return the resulting `HscmModel` for direct pricing.
  #[pyo3(signature = (initial_guess=None))]
  fn calibrate_to_model(&self, initial_guess: Option<[f64; 9]>) -> PyResult<PyHscmModel> {
    use crate::traits::Calibrator;
    use crate::traits::ToModel;
    let res = self
      .inner
      .calibrate(initial_guess.map(hscm_start))
      .map_err(|e| PyValueError::new_err(format!("HSCM calibration failed: {e}")))?;
    Ok(PyHscmModel {
      inner: res.to_model(),
    })
  }
}
