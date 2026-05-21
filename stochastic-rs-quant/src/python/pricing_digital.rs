use pyo3::prelude::*;

use super::parse_option_type;

#[pyclass(name = "CashOrNothingPricer", unsendable)]
pub struct PyCashOrNothingPricer {
  inner: crate::pricing::digital::CashOrNothingPricer,
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
    let ot = parse_option_type(option_type)?;
    Ok(Self {
      inner: crate::pricing::digital::CashOrNothingPricer {
        s,
        k,
        cash,
        r,
        b,
        sigma,
        t,
        option_type: ot,
      },
    })
  }

  fn price(&self) -> f64 {
    self.inner.price()
  }
  fn delta(&self) -> f64 {
    self.inner.delta()
  }
  fn gamma(&self) -> f64 {
    self.inner.gamma()
  }
  fn vega(&self) -> f64 {
    self.inner.vega()
  }
}

#[pyclass(name = "AssetOrNothingPricer", unsendable)]
pub struct PyAssetOrNothingPricer {
  inner: crate::pricing::digital::AssetOrNothingPricer,
}

#[pymethods]
impl PyAssetOrNothingPricer {
  #[new]
  #[pyo3(signature = (s, k, r, b, sigma, t, option_type="call"))]
  fn new(s: f64, k: f64, r: f64, b: f64, sigma: f64, t: f64, option_type: &str) -> PyResult<Self> {
    let ot = parse_option_type(option_type)?;
    Ok(Self {
      inner: crate::pricing::digital::AssetOrNothingPricer {
        s,
        k,
        r,
        b,
        sigma,
        t,
        option_type: ot,
      },
    })
  }

  fn price(&self) -> f64 {
    self.inner.price()
  }
}

#[pyclass(name = "GapPricer", unsendable)]
pub struct PyGapPricer {
  inner: crate::pricing::digital::GapPricer,
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
    let ot = parse_option_type(option_type)?;
    Ok(Self {
      inner: crate::pricing::digital::GapPricer {
        s,
        k1,
        k2,
        r,
        b,
        sigma,
        t,
        option_type: ot,
      },
    })
  }

  fn price(&self) -> f64 {
    self.inner.price()
  }
}

#[pyclass(name = "SuperSharePricer", unsendable)]
pub struct PySuperSharePricer {
  inner: crate::pricing::digital::SuperSharePricer,
}

#[pymethods]
impl PySuperSharePricer {
  #[new]
  fn new(s: f64, x_low: f64, x_high: f64, r: f64, b: f64, sigma: f64, t: f64) -> Self {
    Self {
      inner: crate::pricing::digital::SuperSharePricer {
        s,
        x_low,
        x_high,
        r,
        b,
        sigma,
        t,
      },
    }
  }

  fn price(&self) -> f64 {
    self.inner.price()
  }
}
