use pyo3::prelude::*;

use super::parse_option_type;
use crate::traits::PricerExt;

#[pyclass(name = "BSMPricer", unsendable)]
pub struct PyBSMPricer {
  inner: crate::pricing::bsm::BSMPricer,
}

#[pymethods]
impl PyBSMPricer {
  #[new]
  #[pyo3(signature = (s, v, k, r, tau, option_type="call", q=None))]
  fn new(
    s: f64,
    v: f64,
    k: f64,
    r: f64,
    tau: f64,
    option_type: &str,
    q: Option<f64>,
  ) -> PyResult<Self> {
    let ot = parse_option_type(option_type)?;
    let inner = crate::pricing::bsm::BSMPricer::new(
      s,
      v,
      k,
      r,
      None,
      None,
      q,
      Some(tau),
      None,
      None,
      ot,
      crate::pricing::bsm::BSMCoc::default(),
    );
    Ok(Self { inner })
  }

  fn price(&self) -> f64 {
    self.inner.calculate_price()
  }
  fn call_put(&self) -> (f64, f64) {
    self.inner.calculate_call_put()
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
  fn theta(&self) -> f64 {
    self.inner.theta()
  }
  fn rho(&self) -> f64 {
    self.inner.rho()
  }
  fn vanna(&self) -> f64 {
    self.inner.vanna()
  }
  fn charm(&self) -> f64 {
    self.inner.charm()
  }
  fn implied_volatility(&self, c_price: f64, option_type: &str) -> PyResult<f64> {
    let ot = parse_option_type(option_type)?;
    Ok(self.inner.implied_volatility(c_price, ot))
  }
}

#[pyclass(name = "HestonPricer", unsendable)]
pub struct PyHestonPricer {
  inner: crate::pricing::heston::HestonPricer,
}

#[pymethods]
impl PyHestonPricer {
  #[new]
  #[pyo3(signature = (s, v0, k, r, kappa, theta, sigma, rho, tau, q=None, lambda_=None))]
  fn new(
    s: f64,
    v0: f64,
    k: f64,
    r: f64,
    kappa: f64,
    theta: f64,
    sigma: f64,
    rho: f64,
    tau: f64,
    q: Option<f64>,
    lambda_: Option<f64>,
  ) -> Self {
    let inner = crate::pricing::heston::HestonPricer::new(
      s,
      v0,
      k,
      r,
      q,
      rho,
      kappa,
      theta,
      sigma,
      lambda_,
      Some(tau),
      None,
      None,
    );
    Self { inner }
  }

  fn price(&self) -> f64 {
    self.inner.calculate_price()
  }
  fn call_put(&self) -> (f64, f64) {
    self.inner.calculate_call_put()
  }
}

#[pyclass(name = "SabrPricer", unsendable)]
pub struct PySabrPricer {
  inner: crate::pricing::sabr::SabrPricer,
}

#[pymethods]
impl PySabrPricer {
  #[new]
  #[pyo3(signature = (s, k, r, alpha, beta, nu, rho, tau, q=None))]
  fn new(
    s: f64,
    k: f64,
    r: f64,
    alpha: f64,
    beta: f64,
    nu: f64,
    rho: f64,
    tau: f64,
    q: Option<f64>,
  ) -> Self {
    let inner = crate::pricing::sabr::SabrPricer::new(
      s,
      k,
      r,
      q,
      alpha,
      beta,
      nu,
      rho,
      Some(tau),
      None,
      None,
    );
    Self { inner }
  }

  fn price(&self) -> f64 {
    self.inner.calculate_price()
  }
  fn call_put(&self) -> (f64, f64) {
    self.inner.calculate_call_put()
  }
}

#[pyclass(name = "Merton1976Pricer", unsendable)]
pub struct PyMerton1976Pricer {
  inner: crate::pricing::merton_jump::Merton1976Pricer,
}

#[pymethods]
impl PyMerton1976Pricer {
  /// `m` is the Poisson-series truncation iteration limit (default 50).
  #[new]
  #[pyo3(signature = (s, v, k, r, lambda_, gamma, tau, option_type="call", q=None, m=50))]
  fn new(
    s: f64,
    v: f64,
    k: f64,
    r: f64,
    lambda_: f64,
    gamma: f64,
    tau: f64,
    option_type: &str,
    q: Option<f64>,
    m: usize,
  ) -> PyResult<Self> {
    let ot = parse_option_type(option_type)?;
    let mut builder =
      crate::pricing::merton_jump::Merton1976Pricer::builder(s, v, k, r, lambda_, gamma, m)
        .tau(tau)
        .option_type(ot);
    if let Some(qv) = q {
      builder = builder.q(qv);
    }
    Ok(Self {
      inner: builder.build(),
    })
  }

  fn price(&self) -> f64 {
    self.inner.calculate_price()
  }
  fn call_put(&self) -> (f64, f64) {
    self.inner.calculate_call_put()
  }
}
