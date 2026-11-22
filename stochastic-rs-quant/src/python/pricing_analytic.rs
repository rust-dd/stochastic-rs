use pyo3::prelude::*;

use super::parse_option_type;
use crate::pricing::heston::HestonParams;
use crate::traits::ModelPricer;

/// Black-Scholes-Merton prices and Greeks at the stored `(s, k, r, q, tau)` and option type, which
/// the Rust model takes per query.
#[pyclass(name = "BSMPricer", unsendable)]
pub struct PyBsmPricer {
  inner: crate::pricing::bsm::BsmPricer,
  s: f64,
  k: f64,
  r: f64,
  q: f64,
  tau: f64,
  option_type: crate::OptionType,
}

#[pymethods]
impl PyBsmPricer {
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
    stochastic_rs_distributions::python::value_error_on_panic(|| -> PyResult<Self> {
      let ot = parse_option_type(option_type)?;
      let inner = crate::pricing::bsm::BsmPricer::new(v, crate::pricing::bsm::BsmCoc::Merton1973);
      Ok(Self {
        inner,
        s,
        k,
        r,
        q: q.unwrap_or(0.0),
        tau,
        option_type: ot,
      })
    })?
  }

  fn price(&self) -> f64 {
    self
      .inner
      .price_option(self.s, self.k, self.r, self.q, self.tau, self.option_type)
  }

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

/// Heston prices and Greeks at the stored `(s, k, r, q, tau)`, which the Rust model takes per
/// query.
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
  ) -> pyo3::PyResult<Self> {
    stochastic_rs_distributions::python::value_error_on_panic(|| {
      let inner = crate::pricing::heston::HestonPricer::new(HestonParams {
        v0,
        kappa,
        theta,
        sigma,
        rho,
      })
      .with_vol_risk_premium(lambda_.unwrap_or(0.0));
      Self {
        inner,
        s,
        k,
        r,
        q: q.unwrap_or(0.0),
        tau,
      }
    })
  }

  fn price(&self) -> f64 {
    self
      .inner
      .price_call(self.s, self.k, self.r, self.q, self.tau)
  }

  fn call_put(&self) -> (f64, f64) {
    let pair = self
      .inner
      .price_call_put(self.s, self.k, self.r, self.q, self.tau);
    (pair.call, pair.put)
  }
}

/// The Rust model holds one `SabrParams`, so the wrapper carries the `(s, k, r, q, tau)` query its
/// no-argument methods price at.
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
  ) -> pyo3::PyResult<Self> {
    stochastic_rs_distributions::python::value_error_on_panic(|| {
      let params = crate::pricing::sabr::SabrParams {
        alpha,
        beta,
        rho,
        nu,
      };
      let inner = crate::pricing::sabr::SabrPricer::new(params);
      Self {
        inner,
        s,
        k,
        r,
        q: q.unwrap_or(0.0),
        tau,
      }
    })
  }

  fn price(&self) -> f64 {
    self
      .inner
      .price_call(self.s, self.k, self.r, self.q, self.tau)
  }

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

/// Merton (1976) prices at the stored `(s, k, r, q, tau)` and option type; `q`, default 0, is the
/// dividend yield: this class carries at `BsmCoc::Merton1973`, `b = r − q`.
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
    stochastic_rs_distributions::python::value_error_on_panic(|| -> PyResult<Self> {
      let ot = parse_option_type(option_type)?;
      let inner = crate::pricing::merton_jump::Merton1976Pricer::new(
        v,
        lambda_,
        gamma,
        m,
        crate::pricing::bsm::BsmCoc::Merton1973,
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
    })?
  }

  fn price(&self) -> f64 {
    self
      .inner
      .price_option(self.s, self.k, self.r, self.q, self.tau, self.option_type)
  }

  fn call_put(&self) -> (f64, f64) {
    let pair = self
      .inner
      .price_call_put(self.s, self.k, self.r, self.q, self.tau);
    (pair.call, pair.put)
  }
}

/// Quanto prices at the stored `(s, k, r, q, tau)` and option type, `r` the domestic rate.
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
    stochastic_rs_distributions::python::value_error_on_panic(|| -> PyResult<Self> {
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
    })?
  }

  /// Price of the configured option type in domestic currency.
  fn price(&self) -> f64 {
    self
      .inner
      .price_option(self.s, self.k, self.r, self.q, self.tau, self.option_type)
  }

  /// `(call, put)` in domestic currency.
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

  /// Quanto forward `E_p · S · exp((r_f − q − ρ σ_S σ_E) τ)`.
  fn forward(&self) -> f64 {
    self.inner.forward(self.s, self.q, self.tau)
  }
}

/// Bachelier prices at the stored `(s, k, r, q, tau)` and option type.
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
    stochastic_rs_distributions::python::value_error_on_panic(|| -> PyResult<Self> {
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
    })?
  }

  fn price(&self) -> f64 {
    self
      .inner
      .price_option(self.s, self.k, self.r, self.q, self.tau, self.option_type)
  }

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

/// Heston ADI prices at the stored `(s, k, r, q, tau)`, `r` the domestic rate and `q` the foreign
/// rate or dividend yield.
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
    stochastic_rs_distributions::python::value_error_on_panic(|| -> PyResult<Self> {
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
      let mut inner = crate::pricing::heston_adi::HestonAdiPricer::new(HestonParams {
        v0,
        kappa,
        theta,
        sigma,
        rho,
      })
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
    })?
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
