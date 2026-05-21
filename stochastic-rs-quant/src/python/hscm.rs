use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::traits::ModelPricer;

/// Heston-stochastic-correlation model wrapper. Implements the Carr-Madan
/// pricer through `ModelPricer`. 9 parameters: `(v0, kappa_v, theta_v,
/// sigma_v, rho0, kappa_r, mu_r, sigma_r, rho2)`.
#[pyclass(name = "HscmModel", from_py_object, unsendable)]
#[derive(Clone)]
pub struct PyHscmModel {
  pub inner: crate::pricing::heston_stoch_corr::HscmModel,
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
    if v0 <= 0.0 || theta_v <= 0.0 || sigma_v <= 0.0 || sigma_r <= 0.0 {
      return Err(PyValueError::new_err(
        "v0, theta_v, sigma_v, sigma_r must be > 0",
      ));
    }
    if rho0.abs() >= 1.0 || rho2.abs() >= 1.0 || mu_r.abs() >= 1.0 {
      return Err(PyValueError::new_err("|rho0|, |rho2|, |mu_r| must be < 1"));
    }
    Ok(Self {
      inner: crate::pricing::heston_stoch_corr::HscmModel {
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
  }

  /// Carr-Madan FFT call price.
  fn price_call(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    self.inner.price_call(s, k, r, q, tau)
  }
}

/// Market option for HSCM calibration: strike, maturity, observed price, and
/// instrument-specific risk-free rate.
#[pyclass(name = "HscmMarketOption", from_py_object, unsendable)]
#[derive(Clone)]
pub struct PyHscmMarketOption {
  pub inner: crate::calibration::heston_stoch_corr::MarketOption,
}

#[pymethods]
impl PyHscmMarketOption {
  #[new]
  fn new(strike: f64, maturity: f64, price: f64, rate: f64) -> Self {
    Self {
      inner: crate::calibration::heston_stoch_corr::MarketOption {
        strike,
        maturity,
        price,
        rate,
      },
    }
  }

  #[getter]
  fn strike(&self) -> f64 {
    self.inner.strike
  }
  #[getter]
  fn maturity(&self) -> f64 {
    self.inner.maturity
  }
  #[getter]
  fn price(&self) -> f64 {
    self.inner.price
  }
  #[getter]
  fn rate(&self) -> f64 {
    self.inner.rate
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
  #[new]
  #[pyo3(signature = (s0, options, max_iter=500))]
  fn new(s0: f64, options: Vec<PyHscmMarketOption>, max_iter: usize) -> Self {
    let inner_options: Vec<crate::calibration::heston_stoch_corr::MarketOption> =
      options.into_iter().map(|o| o.inner).collect();
    Self {
      inner: crate::calibration::heston_stoch_corr::HscmCalibrator::new(s0, inner_options)
        .with_max_iter(max_iter),
    }
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
    use crate::traits::Calibrator;
    let res = self
      .inner
      .calibrate(initial_guess)
      .map_err(|e| PyValueError::new_err(format!("HSCM calibration failed: {e}")))?;
    Ok((
      res.kappa_v,
      res.theta_v,
      res.sigma_v,
      res.v0,
      res.kappa_r,
      res.mu_r,
      res.sigma_r,
      res.rho0,
      res.rho2,
      res.rmse,
      res.converged,
    ))
  }

  /// Calibrate then return the resulting `HscmModel` for direct pricing.
  #[pyo3(signature = (initial_guess=None))]
  fn calibrate_to_model(&self, initial_guess: Option<[f64; 9]>) -> PyResult<PyHscmModel> {
    use crate::traits::Calibrator;
    let res = self
      .inner
      .calibrate(initial_guess)
      .map_err(|e| PyValueError::new_err(format!("HSCM calibration failed: {e}")))?;
    Ok(PyHscmModel {
      inner: res.to_model(),
    })
  }
}
