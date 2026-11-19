use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::parse_option_type;
use crate::pricing::barrier::BarrierPricer;
use crate::pricing::barrier::BarrierType;
use crate::pricing::barrier::DoubleBarrierPricer;
use crate::pricing::chooser::ChooserPricer;
use crate::pricing::cliquet::CliquetPricer;
use crate::pricing::compound::CompoundPricer;
use crate::pricing::compound::CompoundType;
use crate::pricing::exotic::BarrierOption;
use crate::pricing::exotic::ChooserOption;
use crate::pricing::exotic::CliquetOption;
use crate::pricing::exotic::CompoundOption;
use crate::pricing::exotic::DoubleBarrierOption;
use crate::pricing::exotic::LookbackOption;
use crate::pricing::lookback::LookbackPricer;
use crate::pricing::variance_swap::VarianceSwapPricer;

/// A pricer with its contract and the `(s, r, q, tau)` query Python's no-argument `price()` reads.
struct Priced<P, C> {
  pricer: P,
  contract: C,
  s: f64,
  r: f64,
  q: f64,
  tau: f64,
}

#[pyclass(name = "BarrierPricer", unsendable)]
pub struct PyBarrierPricer {
  inner: Priced<BarrierPricer, BarrierOption>,
}

#[pymethods]
impl PyBarrierPricer {
  /// `barrier_type`: one of "up_in", "up_out", "down_in", "down_out".
  #[new]
  #[pyo3(signature = (s, k, h, r, q, sigma, t, barrier_type, option_type="call", rebate=0.0))]
  fn new(
    s: f64,
    k: f64,
    h: f64,
    r: f64,
    q: f64,
    sigma: f64,
    t: f64,
    barrier_type: &str,
    option_type: &str,
    rebate: f64,
  ) -> PyResult<Self> {
    stochastic_rs_distributions::python::value_error_on_panic(|| -> PyResult<Self> {
      let barrier_type = match barrier_type.to_ascii_lowercase().as_str() {
        "up_in" | "ui" | "upandin" => BarrierType::UpAndIn,
        "up_out" | "uo" | "upandout" => BarrierType::UpAndOut,
        "down_in" | "di" | "downandin" => BarrierType::DownAndIn,
        "down_out" | "do" | "downandout" => BarrierType::DownAndOut,
        o => {
          return Err(PyValueError::new_err(format!(
            "barrier_type must be one of up_in/up_out/down_in/down_out, got '{o}'"
          )));
        }
      };
      let contract = BarrierOption {
        strike: k,
        barrier: h,
        rebate,
        barrier_type,
        option_type: parse_option_type(option_type)?,
      };
      Ok(Self {
        inner: Priced {
          pricer: BarrierPricer::new(sigma),
          contract,
          s,
          r,
          q,
          tau: t,
        },
      })
    })?
  }

  fn price(&self) -> f64 {
    let p = &self.inner;
    p.pricer.price(&p.contract, p.s, p.r, p.q, p.tau)
  }
}

#[pyclass(name = "DoubleBarrierPricer", unsendable)]
pub struct PyDoubleBarrierPricer {
  inner: Priced<DoubleBarrierPricer, DoubleBarrierOption>,
}

#[pymethods]
impl PyDoubleBarrierPricer {
  #[new]
  #[pyo3(signature = (s, k, h_upper, h_lower, r, q, sigma, t, option_type="call"))]
  fn new(
    s: f64,
    k: f64,
    h_upper: f64,
    h_lower: f64,
    r: f64,
    q: f64,
    sigma: f64,
    t: f64,
    option_type: &str,
  ) -> PyResult<Self> {
    stochastic_rs_distributions::python::value_error_on_panic(|| -> PyResult<Self> {
      let contract = DoubleBarrierOption {
        strike: k,
        lower: h_lower,
        upper: h_upper,
        option_type: parse_option_type(option_type)?,
      };
      Ok(Self {
        inner: Priced {
          pricer: DoubleBarrierPricer::new(sigma),
          contract,
          s,
          r,
          q,
          tau: t,
        },
      })
    })?
  }

  fn price(&self) -> f64 {
    let p = &self.inner;
    p.pricer.price(&p.contract, p.s, p.r, p.q, p.tau)
  }
}

#[pyclass(name = "FloatingLookbackPricer", unsendable)]
pub struct PyFloatingLookbackPricer {
  inner: Priced<LookbackPricer, LookbackOption>,
}

#[pymethods]
impl PyFloatingLookbackPricer {
  #[new]
  #[pyo3(signature = (s, r, q, sigma, t, option_type="call", s_min=None, s_max=None))]
  fn new(
    s: f64,
    r: f64,
    q: f64,
    sigma: f64,
    t: f64,
    option_type: &str,
    s_min: Option<f64>,
    s_max: Option<f64>,
  ) -> PyResult<Self> {
    stochastic_rs_distributions::python::value_error_on_panic(|| -> PyResult<Self> {
      let contract = LookbackOption {
        strike: None,
        running_min: s_min,
        running_max: s_max,
        option_type: parse_option_type(option_type)?,
      };
      Ok(Self {
        inner: Priced {
          pricer: LookbackPricer::new(sigma),
          contract,
          s,
          r,
          q,
          tau: t,
        },
      })
    })?
  }

  fn price(&self) -> f64 {
    let p = &self.inner;
    p.pricer.price(&p.contract, p.s, p.r, p.q, p.tau)
  }
}

#[pyclass(name = "FixedLookbackPricer", unsendable)]
pub struct PyFixedLookbackPricer {
  inner: Priced<LookbackPricer, LookbackOption>,
}

#[pymethods]
impl PyFixedLookbackPricer {
  #[new]
  #[pyo3(signature = (s, k, r, q, sigma, t, option_type="call", s_min=None, s_max=None))]
  fn new(
    s: f64,
    k: f64,
    r: f64,
    q: f64,
    sigma: f64,
    t: f64,
    option_type: &str,
    s_min: Option<f64>,
    s_max: Option<f64>,
  ) -> PyResult<Self> {
    stochastic_rs_distributions::python::value_error_on_panic(|| -> PyResult<Self> {
      let contract = LookbackOption {
        strike: Some(k),
        running_min: s_min,
        running_max: s_max,
        option_type: parse_option_type(option_type)?,
      };
      Ok(Self {
        inner: Priced {
          pricer: LookbackPricer::new(sigma),
          contract,
          s,
          r,
          q,
          tau: t,
        },
      })
    })?
  }

  fn price(&self) -> f64 {
    let p = &self.inner;
    p.pricer.price(&p.contract, p.s, p.r, p.q, p.tau)
  }
}

#[pyclass(name = "SimpleChooserPricer", unsendable)]
pub struct PySimpleChooserPricer {
  inner: Priced<ChooserPricer, ChooserOption>,
}

#[pymethods]
impl PySimpleChooserPricer {
  #[new]
  fn new(s: f64, k: f64, r: f64, q: f64, sigma: f64, t1: f64, t: f64) -> PyResult<Self> {
    stochastic_rs_distributions::python::value_error_on_panic(|| Self {
      inner: Priced {
        pricer: ChooserPricer::new(sigma),
        contract: ChooserOption {
          strike: k,
          choice_time: t1,
        },
        s,
        r,
        q,
        tau: t,
      },
    })
  }

  fn price(&self) -> f64 {
    let p = &self.inner;
    p.pricer.price_simple(&p.contract, p.s, p.r, p.q, p.tau)
  }
}

#[pyclass(name = "CompoundPricer", unsendable)]
pub struct PyCompoundPricer {
  inner: Priced<CompoundPricer, CompoundOption>,
}

#[pymethods]
impl PyCompoundPricer {
  /// `compound_type`: one of "call_on_call" / "call_on_put" / "put_on_call" / "put_on_put".
  #[new]
  fn new(
    s: f64,
    k1: f64,
    k2: f64,
    t1: f64,
    t2: f64,
    r: f64,
    q: f64,
    sigma: f64,
    compound_type: &str,
  ) -> PyResult<Self> {
    stochastic_rs_distributions::python::value_error_on_panic(|| -> PyResult<Self> {
      let compound_type = match compound_type.to_ascii_lowercase().as_str() {
        "call_on_call" | "coc" => CompoundType::CallOnCall,
        "call_on_put" | "cop" => CompoundType::CallOnPut,
        "put_on_call" | "poc" => CompoundType::PutOnCall,
        "put_on_put" | "pop" => CompoundType::PutOnPut,
        o => {
          return Err(PyValueError::new_err(format!(
            "compound_type must be one of call_on_call/call_on_put/put_on_call/put_on_put, got '{o}'"
          )));
        }
      };
      let contract = CompoundOption {
        outer_strike: k1,
        outer_tau: t1,
        inner_strike: k2,
        compound_type,
      };
      Ok(Self {
        inner: Priced {
          pricer: CompoundPricer::new(sigma),
          contract,
          s,
          r,
          q,
          tau: t2,
        },
      })
    })?
  }

  fn price(&self) -> f64 {
    let p = &self.inner;
    p.pricer.price(&p.contract, p.s, p.r, p.q, p.tau)
  }
}

/// `s` stays a keyword; the closed-form cliquet's returns do not read it.
#[pyclass(name = "CliquetPricer", unsendable)]
pub struct PyCliquetPricer {
  inner: Priced<CliquetPricer, CliquetOption>,
}

#[pymethods]
impl PyCliquetPricer {
  #[new]
  #[pyo3(signature = (s, notional, m, t, r, q, sigma, local_floor=None, local_cap=None))]
  fn new(
    s: f64,
    notional: f64,
    m: usize,
    t: f64,
    r: f64,
    q: f64,
    sigma: f64,
    local_floor: Option<f64>,
    local_cap: Option<f64>,
  ) -> PyResult<Self> {
    stochastic_rs_distributions::python::value_error_on_panic(|| Self {
      inner: Priced {
        pricer: CliquetPricer::new(sigma),
        contract: CliquetOption {
          notional,
          periods: m,
          local_floor,
          local_cap,
          global_floor: None,
          global_cap: None,
        },
        s,
        r,
        q,
        tau: t,
      },
    })
  }

  fn price(&self) -> f64 {
    let p = &self.inner;
    p.pricer.price(&p.contract, p.r, p.q, p.tau)
  }
}

#[pyclass(name = "VarianceSwapPricer", unsendable)]
pub struct PyVarianceSwapPricer {
  s: f64,
  r: f64,
  q: f64,
  tau: f64,
}

#[pymethods]
impl PyVarianceSwapPricer {
  #[new]
  fn new(s: f64, r: f64, q: f64, t: f64) -> Self {
    Self { s, r, q, tau: t }
  }

  fn forward(&self) -> f64 {
    self.s * ((self.r - self.q) * self.tau).exp()
  }

  /// BSM fair-strike (`σ²`).
  fn fair_strike_bsm(&self, sigma: f64) -> f64 {
    VarianceSwapPricer::fair_strike_bsm(sigma)
  }

  /// Heston closed-form fair-variance strike (Brockhaus-Long 2000).
  fn fair_strike_heston(&self, v0: f64, kappa: f64, theta: f64) -> f64 {
    VarianceSwapPricer::fair_strike_heston(v0, kappa, theta, self.tau)
  }

  /// Demeterfi-Derman-Kamal-Zou static-replication fair strike.
  fn fair_strike_replication(&self, strikes: Vec<f64>, otm_prices: Vec<f64>) -> f64 {
    VarianceSwapPricer::fair_strike_replication(
      &strikes,
      &otm_prices,
      self.s,
      self.r,
      self.q,
      self.tau,
    )
  }
}
