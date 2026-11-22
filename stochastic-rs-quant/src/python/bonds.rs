use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::curves::PyDiscountCurve;
use crate::traits::ShortRatePricer;

#[pyclass(name = "VasicekBond", unsendable)]
pub struct PyVasicekBond {
  inner: crate::bonds::vasicek::Vasicek,
  r_t: f64,
  tau: f64,
}

#[pymethods]
impl PyVasicekBond {
  #[new]
  fn new(r_t: f64, theta: f64, mu: f64, sigma: f64, tau: f64) -> pyo3::PyResult<Self> {
    stochastic_rs_distributions::python::value_error_on_panic(|| Self {
      inner: crate::bonds::vasicek::Vasicek { theta, mu, sigma },
      r_t,
      tau,
    })
  }

  fn price(&self) -> f64 {
    self.inner.zero_coupon_price(self.r_t, self.tau)
  }
}

/// CIR zero-coupon bond pricer on `dR = theta·(mu − R)·dt + sigma·√R·dW`: `theta` is the reversion
/// speed (Brigo's κ, §3.2.3) and `mu` the long-run mean (Brigo's θ).
#[pyclass(name = "CIRBond", unsendable)]
pub struct PyCirBond {
  inner: crate::bonds::cir::Cir,
  r_t: f64,
  tau: f64,
}

#[pymethods]
impl PyCirBond {
  /// Construct CIR ZCB pricer. `theta` is the mean-reversion speed; `mu`
  /// is the long-run mean. See struct doc for the naming convention.
  #[new]
  fn new(r_t: f64, theta: f64, mu: f64, sigma: f64, tau: f64) -> pyo3::PyResult<Self> {
    stochastic_rs_distributions::python::value_error_on_panic(|| Self {
      inner: crate::bonds::cir::Cir { theta, mu, sigma },
      r_t,
      tau,
    })
  }

  fn price(&self) -> f64 {
    self.inner.zero_coupon_price(self.r_t, self.tau)
  }
}

/// Hull-White (1990) zero-coupon bond on a market curve: `HullWhiteBond.from_curve(curve, ...)`, or
/// `__init__` with the projected `P^M(0, t)`, `P^M(0, T)` and `f^M(0, t)`.
#[pyclass(name = "HullWhiteBond", unsendable)]
pub struct PyHullWhiteBond {
  inner: crate::bonds::hull_white::HullWhite,
  r_t: f64,
  tau: f64,
}

#[pymethods]
impl PyHullWhiteBond {
  /// Direct constructor — pass the curve projection numerics if you
  /// already have them. Most users want `from_curve` instead.
  #[new]
  #[allow(clippy::too_many_arguments)]
  #[pyo3(signature = (r_t, alpha, sigma, t, tau, p0_at_t, p0_at_maturity, f0_at_t))]
  fn new(
    r_t: f64,
    alpha: f64,
    sigma: f64,
    t: f64,
    tau: f64,
    p0_at_t: f64,
    p0_at_maturity: f64,
    f0_at_t: f64,
  ) -> PyResult<Self> {
    stochastic_rs_distributions::python::value_error_on_panic(|| -> PyResult<Self> {
      if alpha <= 0.0 || sigma <= 0.0 {
        return Err(PyValueError::new_err("alpha and sigma must be > 0"));
      }
      if tau < 0.0 {
        return Err(PyValueError::new_err("tau must be >= 0"));
      }
      Ok(Self {
        inner: crate::bonds::hull_white::HullWhite {
          alpha,
          sigma,
          t,
          p0_at_t,
          p0_at_maturity,
          f0_at_t,
          pinned_tau: tau,
          curve: None,
        },
        r_t,
        tau,
      })
    })?
  }

  /// Projects a calibrated [`DiscountCurve`] onto the closed form's time points, the instantaneous
  /// forward `f^M(0, t)` by central differences.
  #[staticmethod]
  #[pyo3(signature = (curve, r_t, alpha, sigma, t, tau))]
  fn from_curve(
    curve: &PyDiscountCurve,
    r_t: f64,
    alpha: f64,
    sigma: f64,
    t: f64,
    tau: f64,
  ) -> PyResult<Self> {
    if alpha <= 0.0 || sigma <= 0.0 {
      return Err(PyValueError::new_err("alpha and sigma must be > 0"));
    }
    if tau < 0.0 {
      return Err(PyValueError::new_err("tau must be >= 0"));
    }
    Ok(Self {
      inner: crate::bonds::hull_white::HullWhite::from_curve(&curve.inner, alpha, sigma, t, tau),
      r_t,
      tau,
    })
  }

  fn price(&self) -> f64 {
    self.inner.zero_coupon_price(self.r_t, self.tau)
  }
}
