use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::parse_option_type;

#[pyclass(name = "MarketSlice", from_py_object, unsendable)]
#[derive(Clone)]
pub struct PyMarketSlice {
  pub inner: crate::calibration::levy::MarketSlice,
}

#[pymethods]
impl PyMarketSlice {
  #[new]
  fn new(strikes: Vec<f64>, prices: Vec<f64>, is_call: Vec<bool>, t: f64) -> Self {
    Self {
      inner: crate::calibration::levy::MarketSlice {
        strikes,
        prices,
        is_call,
        t,
      },
    }
  }
}

#[pyclass(name = "BSMCalibrator", unsendable)]
pub struct PyBSMCalibrator {
  inner: crate::calibration::bsm::BSMCalibrator,
}

#[pymethods]
impl PyBSMCalibrator {
  #[new]
  #[pyo3(signature = (slices, s, r, option_type="call", q=None, sigma_init=0.2))]
  fn new(
    slices: Vec<PyMarketSlice>,
    s: f64,
    r: f64,
    option_type: &str,
    q: Option<f64>,
    sigma_init: f64,
  ) -> PyResult<Self> {
    let ot = parse_option_type(option_type)?;
    let inner_slices: Vec<crate::calibration::levy::MarketSlice> =
      slices.into_iter().map(|s| s.inner).collect();
    let params = crate::calibration::bsm::BSMParams { v: sigma_init };
    Ok(Self {
      inner: crate::calibration::bsm::BSMCalibrator::from_slices(
        params,
        &inner_slices,
        s,
        r,
        None,
        None,
        q,
        ot,
      ),
    })
  }

  /// Run calibration. Returns `(sigma, converged, loss_rmse)`.
  fn calibrate(&self) -> PyResult<(f64, bool, f64)> {
    use crate::traits::Calibrator;
    let res = self
      .inner
      .calibrate(None)
      .map_err(|e| PyValueError::new_err(format!("BSM calibration failed: {e}")))?;
    Ok((
      res.v,
      res.converged,
      res.loss.get(crate::types::LossMetric::Rmse),
    ))
  }
}

#[pyclass(name = "HestonCalibrator", unsendable)]
pub struct PyHestonCalibrator {
  inner: crate::calibration::heston::HestonCalibrator,
}

#[pymethods]
impl PyHestonCalibrator {
  #[new]
  #[pyo3(signature = (slices, s, r, option_type="call", q=None))]
  fn new(
    slices: Vec<PyMarketSlice>,
    s: f64,
    r: f64,
    option_type: &str,
    q: Option<f64>,
  ) -> PyResult<Self> {
    let ot = parse_option_type(option_type)?;
    let inner_slices: Vec<crate::calibration::levy::MarketSlice> =
      slices.into_iter().map(|s| s.inner).collect();
    Ok(Self {
      inner: crate::calibration::heston::HestonCalibrator::from_slices(
        None,
        &inner_slices,
        s,
        r,
        q,
        ot,
        false,
      ),
    })
  }

  /// Returns `(v0, kappa, theta, sigma, rho, converged, loss_rmse)`.
  fn calibrate(&self) -> PyResult<(f64, f64, f64, f64, f64, bool, f64)> {
    use crate::traits::Calibrator;
    let res = self
      .inner
      .calibrate(None)
      .map_err(|e| PyValueError::new_err(format!("Heston calibration failed: {e}")))?;
    let p = &res.params;
    Ok((
      p.v0,
      p.kappa,
      p.theta,
      p.sigma,
      p.rho,
      res.converged,
      res.loss.get(crate::types::LossMetric::Rmse),
    ))
  }
}

#[pyclass(name = "SabrCalibrator", unsendable)]
pub struct PySabrCalibrator {
  inner: crate::calibration::sabr::SabrCalibrator,
}

#[pymethods]
impl PySabrCalibrator {
  /// Single-maturity SABR calibration (β fixed at 1.0 by default).
  #[new]
  #[pyo3(signature = (strikes, prices, s, r, tau, option_type="call", q=None))]
  fn new(
    strikes: Vec<f64>,
    prices: Vec<f64>,
    s: f64,
    r: f64,
    tau: f64,
    option_type: &str,
    q: Option<f64>,
  ) -> PyResult<Self> {
    use nalgebra::DVector;
    if strikes.len() != prices.len() {
      return Err(PyValueError::new_err(
        "strikes and prices must have equal length",
      ));
    }
    let n = strikes.len();
    let ot = parse_option_type(option_type)?;
    Ok(Self {
      inner: crate::calibration::sabr::SabrCalibrator::new(
        None,
        DVector::from_vec(prices),
        DVector::from_vec(vec![s; n]),
        DVector::from_vec(strikes),
        r,
        q,
        tau,
        ot,
        false,
      ),
    })
  }

  /// Returns `(alpha, beta, nu, rho, converged, loss_rmse)`.
  fn calibrate(&self) -> PyResult<(f64, f64, f64, f64, bool, f64)> {
    use crate::traits::Calibrator;
    let res = self
      .inner
      .calibrate(None)
      .map_err(|e| PyValueError::new_err(format!("SABR calibration failed: {e}")))?;
    Ok((
      res.alpha,
      res.beta,
      res.nu,
      res.rho,
      res.converged,
      res.loss.get(crate::types::LossMetric::Rmse),
    ))
  }
}

#[pyclass(name = "SabrCapletCalibrator", unsendable)]
pub struct PySabrCapletCalibrator {
  inner: crate::calibration::sabr_caplet::SabrCapletCalibrator,
}

#[pymethods]
impl PySabrCapletCalibrator {
  /// SABR caplet smile calibrator — fits `(α, ν, ρ)` for a single expiry,
  /// β held fixed.
  #[new]
  fn new(forward: f64, expiry: f64, beta: f64, strikes: Vec<f64>, market_vols: Vec<f64>) -> Self {
    Self {
      inner: crate::calibration::sabr_caplet::SabrCapletCalibrator::new(
        forward,
        expiry,
        beta,
        strikes,
        market_vols,
      ),
    }
  }

  /// Returns `(alpha, beta, nu, rho, rmse, converged)`.
  fn calibrate(&self) -> PyResult<(f64, f64, f64, f64, f64, bool)> {
    use crate::traits::Calibrator;
    let res = self
      .inner
      .calibrate(None)
      .map_err(|e| PyValueError::new_err(format!("SABR caplet calibration failed: {e}")))?;
    Ok((
      res.alpha,
      res.beta,
      res.nu,
      res.rho,
      res.rmse,
      res.converged,
    ))
  }
}
