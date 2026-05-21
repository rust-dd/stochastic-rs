use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::traits::PricerExt;

use super::curves::PyDiscountCurve;

#[pyclass(name = "VasicekBond", unsendable)]
pub struct PyVasicekBond {
  inner: crate::bonds::vasicek::Vasicek,
}

#[pymethods]
impl PyVasicekBond {
  #[new]
  fn new(r_t: f64, theta: f64, mu: f64, sigma: f64, tau: f64) -> Self {
    Self {
      inner: crate::bonds::vasicek::Vasicek {
        r_t,
        theta,
        mu,
        sigma,
        tau,
        eval: None,
        expiration: None,
      },
    }
  }

  fn price(&self) -> f64 {
    self.inner.calculate_price()
  }
}

/// CIR zero-coupon bond pricer. **Field naming convention** (matches the
/// workspace SDE `dR = theta·(mu - R)·dt + sigma·sqrt(R)·dW`):
/// - `theta`: mean-reversion **speed** (κ in Brigo §3.2.3).
/// - `mu`: **long-run mean** of the short rate (θ in Brigo).
/// Calibrators that follow Brigo's symbols (e.g. `kappa`, `theta`) need to
/// rename `kappa` → `theta` and `theta` → `mu` when constructing a `CIRBond`.
#[pyclass(name = "CIRBond", unsendable)]
pub struct PyCIRBond {
  inner: crate::bonds::cir::Cir,
}

#[pymethods]
impl PyCIRBond {
  /// Construct CIR ZCB pricer. `theta` is the mean-reversion speed; `mu`
  /// is the long-run mean. See struct doc for the naming convention.
  #[new]
  fn new(r_t: f64, theta: f64, mu: f64, sigma: f64, tau: f64) -> Self {
    Self {
      inner: crate::bonds::cir::Cir {
        r_t,
        theta,
        mu,
        sigma,
        tau,
        eval: None,
        expiration: None,
      },
    }
  }

  fn price(&self) -> f64 {
    self.inner.calculate_price()
  }
}

/// Hull-White (1990) ZCB closed-form pricer projected onto a calibrated
/// `DiscountCurve`. Use `HullWhiteBond.from_curve(curve, ...)` to drop in
/// the market discount curve; or build directly by passing the projected
/// $P^M(0,t)$ / $P^M(0,T)$ / $f^M(0,t)$ values via `__init__`.
#[pyclass(name = "HullWhiteBond", unsendable)]
pub struct PyHullWhiteBond {
  inner: crate::bonds::hull_white::HullWhite,
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
    if alpha <= 0.0 || sigma <= 0.0 {
      return Err(PyValueError::new_err("alpha and sigma must be > 0"));
    }
    if tau < 0.0 {
      return Err(PyValueError::new_err("tau must be >= 0"));
    }
    Ok(Self {
      inner: crate::bonds::hull_white::HullWhite {
        r_t,
        alpha,
        sigma,
        tau,
        t,
        p0_at_t,
        p0_at_maturity,
        f0_at_t,
        eval: None,
        expiration: None,
      },
    })
  }

  /// Build by projecting a calibrated [`DiscountCurve`] onto the time
  /// points the closed form needs. The instantaneous forward
  /// $f^M(0, t)$ is estimated via centered finite differences.
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
      inner: crate::bonds::hull_white::HullWhite::from_curve(
        &curve.inner,
        r_t,
        alpha,
        sigma,
        t,
        tau,
        None,
        None,
      ),
    })
  }

  fn price(&self) -> f64 {
    self.inner.calculate_price()
  }
}
