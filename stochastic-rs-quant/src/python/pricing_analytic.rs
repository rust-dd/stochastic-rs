use pyo3::prelude::*;

use super::parse_option_type;
use crate::traits::ModelPricer;

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

/// The Rust model holds `(v, lambda, gamma, m, coc)` only, so the wrapper
/// carries the `(s, k, r, q, tau)` query and the option type the
/// Python-visible no-argument methods are defined at. Python's constructor
/// signature is unchanged.
///
/// `q` remains a no-op here, exactly as before: this class hardcodes
/// `BSMCoc::Bsm1973`, whose cost of carry is `r` and never reads the
/// dividend yield.
#[pyclass(name = "Merton1976Pricer", unsendable)]
pub struct PyMerton1976Pricer {
  inner: crate::pricing::merton_jump::Merton1976Pricer,
  s: f64,
  k: f64,
  r: f64,
  q: f64,
  tau: f64,
  option_type: crate::OptionType,
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
    let inner = crate::pricing::merton_jump::Merton1976Pricer::new(
      v,
      lambda_,
      gamma,
      m,
      crate::pricing::bsm::BSMCoc::Bsm1973,
    );
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
}

/// The Rust model holds `(σ_S, σ_E, ρ, r_f, E_p)` only, so the wrapper
/// carries the `(s, k, r, q, tau)` query and the option type that the
/// Python-visible no-argument methods are defined at; `r` is the domestic
/// rate.
#[pyclass(name = "QuantoPricer", unsendable)]
pub struct PyQuantoPricer {
  inner: crate::pricing::quanto::QuantoPricer,
  s: f64,
  k: f64,
  r: f64,
  q: f64,
  tau: f64,
  option_type: crate::OptionType,
}

#[pymethods]
impl PyQuantoPricer {
  #[new]
  #[pyo3(signature = (s, v, k, r, tau, r_f, v_fx, rho, fixed_rate, option_type="call", q=None))]
  #[allow(clippy::too_many_arguments)]
  fn new(
    s: f64,
    v: f64,
    k: f64,
    r: f64,
    tau: f64,
    r_f: f64,
    v_fx: f64,
    rho: f64,
    fixed_rate: f64,
    option_type: &str,
    q: Option<f64>,
  ) -> PyResult<Self> {
    let option_type = parse_option_type(option_type)?;
    Ok(Self {
      inner: crate::pricing::quanto::QuantoPricer::new(v, v_fx, rho, r_f, fixed_rate),
      s,
      k,
      r,
      q: q.unwrap_or(0.0),
      tau,
      option_type,
    })
  }

  /// Price of the configured option type in domestic currency.
  fn price(&self) -> f64 {
    self
      .inner
      .price_option(self.s, self.k, self.r, self.q, self.tau, self.option_type)
  }

  /// `(call, put)` in domestic currency.
  fn call_put(&self) -> (f64, f64) {
    self
      .inner
      .call_put(self.s, self.k, self.r, self.q, self.tau)
  }

  /// Quanto forward `E_p · S · exp((r_f − q − ρ σ_S σ_E) τ)`.
  fn forward(&self) -> f64 {
    self.inner.forward(self.s, self.q, self.tau)
  }
}

/// The Rust model holds the normal volatility only, so the wrapper carries
/// the `(s, k, r, q, tau)` query and the option type that the Python-visible
/// no-argument methods are defined at.
#[pyclass(name = "BachelierPricer", unsendable)]
pub struct PyBachelierPricer {
  inner: crate::pricing::bachelier::BachelierPricer,
  s: f64,
  k: f64,
  r: f64,
  q: f64,
  tau: f64,
  option_type: crate::OptionType,
}

#[pymethods]
impl PyBachelierPricer {
  /// `v` is the normal volatility in price units per √year.
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
    let option_type = parse_option_type(option_type)?;
    Ok(Self {
      inner: crate::pricing::bachelier::BachelierPricer::new(v),
      s,
      k,
      r,
      q: q.unwrap_or(0.0),
      tau,
      option_type,
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

  /// Forward `S e^{(r − q)τ}` the option is struck on.
  fn forward(&self) -> f64 {
    self.inner.forward(self.s, self.r, self.q, self.tau)
  }

  /// Normal vega `e^{−rτ} √τ φ(d)`.
  fn vega(&self) -> f64 {
    self.inner.vega(self.s, self.k, self.r, self.q, self.tau)
  }

  /// Normal volatility implied by a spot-quoted `price`; `NaN` below
  /// intrinsic value.
  fn implied_volatility(&self, price: f64, option_type: &str) -> PyResult<f64> {
    let option_type = parse_option_type(option_type)?;
    Ok(
      self
        .inner
        .implied_volatility(price, self.s, self.k, self.r, self.q, self.tau, option_type),
    )
  }
}

/// The Rust model holds the Heston parameters and the ADI numerics, so the
/// wrapper carries the `(s, k, r, q, tau)` query the Python-visible
/// no-argument methods are defined at; `r` is the domestic rate and `q` the
/// foreign rate / dividend yield.
#[pyclass(name = "HestonAdiPricer", unsendable)]
pub struct PyHestonAdiPricer {
  inner: crate::pricing::heston_adi::HestonAdiPricer,
  s: f64,
  k: f64,
  r: f64,
  q: f64,
  tau: f64,
}

#[pymethods]
impl PyHestonAdiPricer {
  /// `scheme` is one of `"douglas"`, `"cs"`, `"mcs"`, `"hv"`; `barrier` turns
  /// the call into a down-and-out call.
  #[new]
  #[pyo3(signature = (s, v0, k, r, kappa, theta, sigma, rho, tau, q=None, m1=100, m2=50, steps=50, scheme="mcs", damping=true, barrier=None))]
  #[allow(clippy::too_many_arguments)]
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
    m1: usize,
    m2: usize,
    steps: usize,
    scheme: &str,
    damping: bool,
    barrier: Option<f64>,
  ) -> PyResult<Self> {
    use crate::pricing::heston_adi::AdiScheme;
    let scheme = match scheme.to_ascii_lowercase().as_str() {
      "douglas" | "do" => AdiScheme::Douglas,
      "cs" | "craig_sneyd" | "craig-sneyd" => AdiScheme::CraigSneyd,
      "mcs" | "modified_craig_sneyd" => AdiScheme::ModifiedCraigSneyd,
      "hv" | "hundsdorfer_verwer" => AdiScheme::HundsdorferVerwer,
      other => {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
          "scheme must be douglas/cs/mcs/hv, got '{other}'"
        )));
      }
    };
    let mut inner = crate::pricing::heston_adi::HestonAdiPricer::new(v0, kappa, theta, sigma, rho)
      .with_grid(m1, m2, steps)
      .with_scheme(scheme)
      .with_damping(damping);
    if let Some(b) = barrier {
      inner = inner.with_barrier(b);
    }
    Ok(Self {
      inner,
      s,
      k,
      r,
      q: q.unwrap_or(0.0),
      tau,
    })
  }

  /// Call price (down-and-out call with a barrier).
  fn price(&self) -> f64 {
    self
      .inner
      .price_call(self.s, self.k, self.r, self.q, self.tau)
  }

  /// `(call, put)`; the put is `NaN` with a barrier.
  fn call_put(&self) -> (f64, f64) {
    (
      self
        .inner
        .price_call(self.s, self.k, self.r, self.q, self.tau),
      self
        .inner
        .price_put(self.s, self.k, self.r, self.q, self.tau),
    )
  }
}
