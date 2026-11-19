use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::parse_option_type;
use crate::traits::ModelPricer;

/// The Rust model holds `sigma` only, so the wrapper carries the
/// `(s, k, r, q, tau)` query the Python-visible no-argument methods are
/// defined at. Python's constructor signature is unchanged.
#[pyclass(name = "AsianPricer", unsendable)]
pub struct PyAsianPricer {
  inner: crate::pricing::asian::AsianPricer,
  s: f64,
  k: f64,
  r: f64,
  q: f64,
  tau: f64,
}

#[pymethods]
impl PyAsianPricer {
  #[new]
  #[pyo3(signature = (s, v, k, r, tau, q=None))]
  fn new(s: f64, v: f64, k: f64, r: f64, tau: f64, q: Option<f64>) -> pyo3::PyResult<Self> {
    stochastic_rs_distributions::python::value_error_on_panic(|| Self {
      inner: crate::pricing::asian::AsianPricer::new(v),
      s,
      k,
      r,
      q: q.unwrap_or(0.0),
      tau,
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

/// The Rust model holds `sigma` only, so the wrapper carries the
/// `(s, k, r, q, tau)` query and the option type the Python-visible
/// no-argument methods are defined at. Python's constructor signature is
/// unchanged.
#[pyclass(name = "BjerksundStensland2002Pricer", unsendable)]
pub struct PyBjerksundStensland2002Pricer {
  inner: crate::pricing::bjerksund_stensland::BjerksundStensland2002Pricer,
  s: f64,
  k: f64,
  r: f64,
  q: f64,
  tau: f64,
  option_type: crate::OptionType,
}

#[pymethods]
impl PyBjerksundStensland2002Pricer {
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
      Ok(Self {
        inner: crate::pricing::bjerksund_stensland::BjerksundStensland2002Pricer::new(v),
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
}

/// Monte Carlo estimate: the mean, its standard error, and the sample
/// count, with the 95% confidence interval derived on demand.
#[pyclass(name = "McEstimate")]
pub struct PyMcEstimate {
  inner: stochastic_rs_stochastic::mc::McEstimate<f64>,
}

impl From<stochastic_rs_stochastic::mc::McEstimate<f64>> for PyMcEstimate {
  fn from(inner: stochastic_rs_stochastic::mc::McEstimate<f64>) -> Self {
    Self { inner }
  }
}

#[pymethods]
impl PyMcEstimate {
  #[getter]
  fn mean(&self) -> f64 {
    self.inner.mean
  }

  #[getter]
  fn std_err(&self) -> f64 {
    self.inner.std_err
  }

  #[getter]
  fn n_samples(&self) -> usize {
    self.inner.n_samples
  }

  /// Symmetric confidence interval `[mean ± z * std_err]`.
  #[pyo3(signature = (z = 1.96))]
  fn confidence_interval(&self, z: f64) -> (f64, f64) {
    self.inner.confidence_interval(z)
  }

  fn ci_95(&self) -> (f64, f64) {
    self.inner.ci_95()
  }

  fn __repr__(&self) -> String {
    format!("{}", self.inner)
  }
}

#[pyclass(name = "MCBarrierPricer", unsendable)]
pub struct PyMcBarrierPricer {
  inner: crate::pricing::barrier::McBarrierPricer,
}

#[pymethods]
impl PyMcBarrierPricer {
  #[new]
  #[pyo3(signature = (n_paths=10000, n_steps=252))]
  fn new(n_paths: usize, n_steps: usize) -> pyo3::PyResult<Self> {
    stochastic_rs_distributions::python::value_error_on_panic(|| Self {
      inner: crate::pricing::barrier::McBarrierPricer { n_paths, n_steps },
    })
  }

  /// `barrier_type`: one of "up_in" / "up_out" / "down_in" / "down_out".
  #[pyo3(signature = (s, k, h, r, sigma, t, barrier_type, option_type="call"))]
  fn price(
    &self,
    s: f64,
    k: f64,
    h: f64,
    r: f64,
    sigma: f64,
    t: f64,
    barrier_type: &str,
    option_type: &str,
  ) -> PyResult<PyMcEstimate> {
    use crate::pricing::barrier::BarrierType;
    let bt = match barrier_type.to_ascii_lowercase().as_str() {
      "up_in" | "ui" => BarrierType::UpAndIn,
      "up_out" | "uo" => BarrierType::UpAndOut,
      "down_in" | "di" => BarrierType::DownAndIn,
      "down_out" | "do" => BarrierType::DownAndOut,
      o => {
        return Err(PyValueError::new_err(format!(
          "barrier_type must be one of up_in/up_out/down_in/down_out, got '{o}'"
        )));
      }
    };
    let ot = parse_option_type(option_type)?;
    Ok(self.inner.price(s, k, h, r, sigma, t, bt, ot).into())
  }
}

/// The Rust model holds `(sigma1, sigma2, corr)`, so the wrapper carries the `(f1, f2, x, r, tau)`
/// query its no-argument `price()` reads.
#[pyclass(name = "KirkSpreadPricer", unsendable)]
pub struct PyKirkSpreadPricer {
  inner: crate::pricing::kirk::KirkSpreadPricer,
  f1: f64,
  f2: f64,
  x: f64,
  r: f64,
  tau: f64,
}

#[pymethods]
impl PyKirkSpreadPricer {
  #[new]
  fn new(
    f1: f64,
    f2: f64,
    x: f64,
    r: f64,
    v1: f64,
    v2: f64,
    corr: f64,
    tau: f64,
  ) -> pyo3::PyResult<Self> {
    stochastic_rs_distributions::python::value_error_on_panic(|| Self {
      inner: crate::pricing::kirk::KirkSpreadPricer::new(v1, v2, corr),
      f1,
      f2,
      x,
      r,
      tau,
    })
  }

  fn price(&self) -> f64 {
    self
      .inner
      .spread_call(self.f1, self.f2, self.x, self.r, self.tau)
  }
}
