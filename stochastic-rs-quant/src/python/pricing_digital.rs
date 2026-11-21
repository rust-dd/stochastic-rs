use pyo3::prelude::*;

use super::parse_option_type;
use crate::traits::ModelPricer;

/// Cash-or-nothing prices at the stored `(s, k, r, q, tau)` and option type; the carry `b` the
/// constructor takes is kept as `q = r − b`, since the model reads `b = r − q`.
#[pyclass(name = "CashOrNothingPricer", unsendable)]
pub struct PyCashOrNothingPricer {
  inner: crate::pricing::digital::CashOrNothingPricer,
  s: f64,
  k: f64,
  r: f64,
  q: f64,
  tau: f64,
  option_type: crate::OptionType,
}

#[pymethods]
impl PyCashOrNothingPricer {
  /// `b` is the cost-of-carry (typically `r - q`).
  #[new]
  #[pyo3(signature = (s, k, cash, r, b, sigma, t, option_type="call"))]
  fn new(
    s: f64,
    k: f64,
    cash: f64,
    r: f64,
    b: f64,
    sigma: f64,
    t: f64,
    option_type: &str,
  ) -> PyResult<Self> {
    stochastic_rs_distributions::python::value_error_on_panic(|| -> PyResult<Self> {
      let ot = parse_option_type(option_type)?;
      Ok(Self {
        inner: crate::pricing::digital::CashOrNothingPricer::new(cash, sigma),
        s,
        k,
        r,
        q: r - b,
        tau: t,
        option_type: ot,
      })
    })?
  }

  fn price(&self) -> f64 {
    self
      .inner
      .price_option(self.s, self.k, self.r, self.q, self.tau, self.option_type)
  }

  fn delta(&self) -> f64 {
    self
      .inner
      .delta(self.s, self.k, self.r, self.q, self.tau, self.option_type)
  }

  fn gamma(&self) -> f64 {
    self
      .inner
      .gamma(self.s, self.k, self.r, self.q, self.tau, self.option_type)
  }

  fn vega(&self) -> f64 {
    self
      .inner
      .vega(self.s, self.k, self.r, self.q, self.tau, self.option_type)
  }
}

/// Asset-or-nothing price at the stored `(s, k, r, q, tau)` and option type; the carry `b` is kept
/// as `q = r − b`.
#[pyclass(name = "AssetOrNothingPricer", unsendable)]
pub struct PyAssetOrNothingPricer {
  inner: crate::pricing::digital::AssetOrNothingPricer,
  s: f64,
  k: f64,
  r: f64,
  q: f64,
  tau: f64,
  option_type: crate::OptionType,
}

#[pymethods]
impl PyAssetOrNothingPricer {
  #[new]
  #[pyo3(signature = (s, k, r, b, sigma, t, option_type="call"))]
  fn new(s: f64, k: f64, r: f64, b: f64, sigma: f64, t: f64, option_type: &str) -> PyResult<Self> {
    stochastic_rs_distributions::python::value_error_on_panic(|| -> PyResult<Self> {
      let ot = parse_option_type(option_type)?;
      Ok(Self {
        inner: crate::pricing::digital::AssetOrNothingPricer::new(sigma),
        s,
        k,
        r,
        q: r - b,
        tau: t,
        option_type: ot,
      })
    })?
  }

  fn price(&self) -> f64 {
    self
      .inner
      .price_option(self.s, self.k, self.r, self.q, self.tau, self.option_type)
  }
}

/// Gap option price at the stored `(s, k1, r, q, tau)` and option type, the trigger `k1` being the
/// query strike; the carry `b` is kept as `q = r − b`.
#[pyclass(name = "GapPricer", unsendable)]
pub struct PyGapPricer {
  inner: crate::pricing::digital::GapPricer,
  s: f64,
  k1: f64,
  r: f64,
  q: f64,
  tau: f64,
  option_type: crate::OptionType,
}

#[pymethods]
impl PyGapPricer {
  #[new]
  #[pyo3(signature = (s, k1, k2, r, b, sigma, t, option_type="call"))]
  fn new(
    s: f64,
    k1: f64,
    k2: f64,
    r: f64,
    b: f64,
    sigma: f64,
    t: f64,
    option_type: &str,
  ) -> PyResult<Self> {
    stochastic_rs_distributions::python::value_error_on_panic(|| -> PyResult<Self> {
      let ot = parse_option_type(option_type)?;
      Ok(Self {
        inner: crate::pricing::digital::GapPricer::new(k2, sigma),
        s,
        k1,
        r,
        q: r - b,
        tau: t,
        option_type: ot,
      })
    })?
  }

  fn price(&self) -> f64 {
    self
      .inner
      .price_option(self.s, self.k1, self.r, self.q, self.tau, self.option_type)
  }
}

/// Supershare price at the stored `(s, x_low, r, q, tau)`, the lower trigger `x_low` being the
/// query strike; the carry `b` is kept as `q = r − b`.
#[pyclass(name = "SuperSharePricer", unsendable)]
pub struct PySuperSharePricer {
  inner: crate::pricing::digital::SuperSharePricer,
  s: f64,
  x_low: f64,
  r: f64,
  q: f64,
  tau: f64,
}

#[pymethods]
impl PySuperSharePricer {
  #[new]
  fn new(
    s: f64,
    x_low: f64,
    x_high: f64,
    r: f64,
    b: f64,
    sigma: f64,
    t: f64,
  ) -> pyo3::PyResult<Self> {
    stochastic_rs_distributions::python::value_error_on_panic(|| Self {
      inner: crate::pricing::digital::SuperSharePricer::new(x_high, sigma),
      s,
      x_low,
      r,
      q: r - b,
      tau: t,
    })
  }

  fn price(&self) -> f64 {
    self
      .inner
      .price_call(self.s, self.x_low, self.r, self.q, self.tau)
  }
}
