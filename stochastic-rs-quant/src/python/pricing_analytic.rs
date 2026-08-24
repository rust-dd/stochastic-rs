use pyo3::prelude::*;

use super::parse_option_type;
use crate::traits::ModelPricer;
use crate::traits::PricerExt;

/// The Rust model holds `(v, coc)` only, so the wrapper carries the
/// `(s, k, r, q, tau)` query and the option type that the Python-visible
/// no-argument methods are defined at. Python's constructor signature is
/// unchanged.
#[pyclass(name = "BSMPricer", unsendable)]
pub struct PyBSMPricer {
  inner: crate::pricing::bsm::BSMPricer,
  s: f64,
  k: f64,
  r: f64,
  q: f64,
  tau: f64,
  option_type: crate::OptionType,
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
    let inner = crate::pricing::bsm::BSMPricer::new(v, crate::pricing::bsm::BSMCoc::default());
    Ok(Self {
      inner,
      s,
      k,
      r,
      q: q.unwrap_or(0.0),
      tau,
      option_type: ot,
    })
  }

  fn price(&self) -> f64 {
    self
      .inner
      .price_option(self.s, self.k, self.r, self.q, self.tau, self.option_type)
  }
  fn call_put(&self) -> (f64, f64) {
    self
      .inner
      .call_put(self.s, self.k, self.r, self.q, self.tau)
  }
  fn delta(&self) -> f64 {
    self
      .inner
      .delta(self.s, self.k, self.r, self.q, self.tau, self.option_type)
  }
  fn gamma(&self) -> f64 {
    self.inner.gamma(self.s, self.k, self.r, self.q, self.tau)
  }
  fn vega(&self) -> f64 {
    self.inner.vega(self.s, self.k, self.r, self.q, self.tau)
  }
  fn theta(&self) -> f64 {
    self
      .inner
      .theta(self.s, self.k, self.r, self.q, self.tau, self.option_type)
  }
  fn rho(&self) -> f64 {
    self
      .inner
      .rho(self.s, self.k, self.r, self.q, self.tau, self.option_type)
  }
  fn vanna(&self) -> f64 {
    self.inner.vanna(self.s, self.k, self.r, self.q, self.tau)
  }
  fn charm(&self) -> f64 {
    self
      .inner
      .charm(self.s, self.k, self.r, self.q, self.tau, self.option_type)
  }
  fn implied_volatility(&self, c_price: f64, option_type: &str) -> PyResult<f64> {
    let ot = parse_option_type(option_type)?;
    Ok(
      self
        .inner
        .implied_volatility(c_price, self.s, self.k, self.r, self.q, self.tau, ot),
    )
  }
}

/// The Rust model holds the six Heston parameters only, so the wrapper
/// carries the `(s, k, r, q, tau)` query the Python-visible no-argument
/// methods are defined at. Python's constructor signature is unchanged.
#[pyclass(name = "HestonPricer", unsendable)]
pub struct PyHestonPricer {
  inner: crate::pricing::heston::HestonPricer,
  s: f64,
  k: f64,
  r: f64,
  q: f64,
  tau: f64,
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
    let inner = crate::pricing::heston::HestonPricer::new(v0, rho, kappa, theta, sigma, lambda_);
    Self {
      inner,
      s,
      k,
      r,
      q: q.unwrap_or(0.0),
      tau,
    }
  }

  fn price(&self) -> f64 {
    self
      .inner
      .price_call(self.s, self.k, self.r, self.q, self.tau)
  }
  fn call_put(&self) -> (f64, f64) {
    self
      .inner
      .call_put(self.s, self.k, self.r, self.q, self.tau)
  }
}

/// The Rust model holds `(alpha, beta, nu, rho)` only, so the wrapper
/// carries the `(s, k, r, q, tau)` query the Python-visible no-argument
/// methods are defined at. Python's constructor signature is unchanged.
#[pyclass(name = "SabrPricer", unsendable)]
pub struct PySabrPricer {
  inner: crate::pricing::sabr::SabrPricer,
  s: f64,
  k: f64,
  r: f64,
  q: f64,
  tau: f64,
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
    let inner = crate::pricing::sabr::SabrPricer::new(alpha, beta, nu, rho);
    Self {
      inner,
      s,
      k,
      r,
      q: q.unwrap_or(0.0),
      tau,
    }
  }

  fn price(&self) -> f64 {
    self
      .inner
      .price_call(self.s, self.k, self.r, self.q, self.tau)
  }
  fn call_put(&self) -> (f64, f64) {
    self
      .inner
      .call_put(self.s, self.k, self.r, self.q, self.tau)
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
