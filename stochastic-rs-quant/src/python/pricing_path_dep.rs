use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::traits::PricerExt;

use super::parse_option_type;

#[pyclass(name = "AsianPricer", unsendable)]
pub struct PyAsianPricer {
  inner: crate::pricing::asian::AsianPricer,
}

#[pymethods]
impl PyAsianPricer {
  #[new]
  #[pyo3(signature = (s, v, k, r, tau, q=None))]
  fn new(s: f64, v: f64, k: f64, r: f64, tau: f64, q: Option<f64>) -> Self {
    Self {
      inner: crate::pricing::asian::AsianPricer::new(s, v, k, r, q, Some(tau), None, None),
    }
  }

  fn price(&self) -> f64 {
    self.inner.calculate_price()
  }
  fn call_put(&self) -> (f64, f64) {
    self.inner.calculate_call_put()
  }
}

#[pyclass(name = "BarrierPricer", unsendable)]
pub struct PyBarrierPricer {
  inner: crate::pricing::barrier::BarrierPricer,
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
    use crate::pricing::barrier::BarrierType;
    let bt = match barrier_type.to_ascii_lowercase().as_str() {
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
    let ot = parse_option_type(option_type)?;
    Ok(Self {
      inner: crate::pricing::barrier::BarrierPricer {
        s,
        k,
        h,
        r,
        q,
        sigma,
        t,
        rebate,
        barrier_type: bt,
        option_type: ot,
      },
    })
  }

  fn price(&self) -> f64 {
    self.inner.price()
  }
}

#[pyclass(name = "FloatingLookbackPricer", unsendable)]
pub struct PyFloatingLookbackPricer {
  inner: crate::pricing::lookback::FloatingLookbackPricer,
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
    let ot = parse_option_type(option_type)?;
    Ok(Self {
      inner: crate::pricing::lookback::FloatingLookbackPricer {
        s,
        s_min,
        s_max,
        r,
        q,
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

#[pyclass(name = "BjerksundStensland2002Pricer", unsendable)]
pub struct PyBjerksundStensland2002Pricer {
  inner: crate::pricing::bjerksund_stensland::BjerksundStensland2002Pricer,
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
    let ot = parse_option_type(option_type)?;
    Ok(Self {
      inner: crate::pricing::bjerksund_stensland::BjerksundStensland2002Pricer::new(
        s,
        v,
        k,
        r,
        q,
        Some(tau),
        None,
        None,
        ot,
      ),
    })
  }

  fn price(&self) -> f64 {
    self.inner.calculate_price()
  }
  fn call_put(&self) -> (f64, f64) {
    self.inner.calculate_call_put()
  }
}

#[pyclass(name = "VarianceSwapPricer", unsendable)]
pub struct PyVarianceSwapPricer {
  inner: crate::pricing::variance_swap::VarianceSwapPricer,
}

#[pymethods]
impl PyVarianceSwapPricer {
  #[new]
  fn new(s: f64, r: f64, q: f64, t: f64) -> Self {
    Self {
      inner: crate::pricing::variance_swap::VarianceSwapPricer { s, r, q, t },
    }
  }

  fn forward(&self) -> f64 {
    self.inner.forward()
  }
  /// BSM fair-strike (`σ²`).
  fn fair_strike_bsm(&self, sigma: f64) -> f64 {
    self.inner.fair_strike_bsm(sigma)
  }
  /// Heston closed-form fair-variance strike (Brockhaus-Long 2000).
  fn fair_strike_heston(&self, v0: f64, kappa: f64, theta: f64) -> f64 {
    self.inner.fair_strike_heston(v0, kappa, theta)
  }
  /// Demeterfi-Derman-Kamal-Zou static-replication fair strike.
  fn fair_strike_replication(&self, strikes: Vec<f64>, otm_prices: Vec<f64>) -> f64 {
    self.inner.fair_strike_replication(&strikes, &otm_prices)
  }
}

#[pyclass(name = "CompoundPricer", unsendable)]
pub struct PyCompoundPricer {
  inner: crate::pricing::compound::CompoundPricer,
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
    use crate::pricing::compound::CompoundType;
    let ct = match compound_type.to_ascii_lowercase().as_str() {
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
    Ok(Self {
      inner: crate::pricing::compound::CompoundPricer {
        s,
        k1,
        k2,
        t1,
        t2,
        r,
        q,
        sigma,
        compound_type: ct,
      },
    })
  }

  fn price(&self) -> f64 {
    self.inner.price()
  }
}

#[pyclass(name = "SimpleChooserPricer", unsendable)]
pub struct PySimpleChooserPricer {
  inner: crate::pricing::chooser::SimpleChooserPricer,
}

#[pymethods]
impl PySimpleChooserPricer {
  #[new]
  fn new(s: f64, k: f64, r: f64, q: f64, sigma: f64, t1: f64, t: f64) -> Self {
    Self {
      inner: crate::pricing::chooser::SimpleChooserPricer {
        s,
        k,
        r,
        q,
        sigma,
        t1,
        t,
      },
    }
  }

  fn price(&self) -> f64 {
    self.inner.price()
  }
}

#[pyclass(name = "CliquetPricer", unsendable)]
pub struct PyCliquetPricer {
  inner: crate::pricing::cliquet::CliquetPricer,
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
  ) -> Self {
    Self {
      inner: crate::pricing::cliquet::CliquetPricer {
        s,
        notional,
        m,
        t,
        r,
        q,
        sigma,
        local_floor,
        local_cap,
      },
    }
  }

  fn price(&self) -> f64 {
    self.inner.price()
  }
}

#[pyclass(name = "FixedLookbackPricer", unsendable)]
pub struct PyFixedLookbackPricer {
  inner: crate::pricing::lookback::FixedLookbackPricer,
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
    let ot = parse_option_type(option_type)?;
    Ok(Self {
      inner: crate::pricing::lookback::FixedLookbackPricer {
        s,
        k,
        s_min,
        s_max,
        r,
        q,
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

#[pyclass(name = "DoubleBarrierPricer", unsendable)]
pub struct PyDoubleBarrierPricer {
  inner: crate::pricing::barrier::DoubleBarrierPricer,
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
    let ot = parse_option_type(option_type)?;
    Ok(Self {
      inner: crate::pricing::barrier::DoubleBarrierPricer {
        s,
        k,
        h_upper,
        h_lower,
        r,
        q,
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

#[pyclass(name = "MCBarrierPricer", unsendable)]
pub struct PyMCBarrierPricer {
  inner: crate::pricing::barrier::MCBarrierPricer,
}

#[pymethods]
impl PyMCBarrierPricer {
  #[new]
  #[pyo3(signature = (n_paths=10000, n_steps=252))]
  fn new(n_paths: usize, n_steps: usize) -> Self {
    Self {
      inner: crate::pricing::barrier::MCBarrierPricer { n_paths, n_steps },
    }
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
  ) -> PyResult<f64> {
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
    Ok(self.inner.price(s, k, h, r, sigma, t, bt, ot))
  }
}

#[pyclass(name = "KirkSpreadPricer", unsendable)]
pub struct PyKirkSpreadPricer {
  inner: crate::pricing::kirk::KirkSpreadPricer,
}

#[pymethods]
impl PyKirkSpreadPricer {
  #[new]
  fn new(f1: f64, f2: f64, x: f64, r: f64, v1: f64, v2: f64, corr: f64, tau: f64) -> Self {
    Self {
      inner: crate::pricing::kirk::KirkSpreadPricer::new(
        f1,
        f2,
        x,
        r,
        v1,
        v2,
        corr,
        Some(tau),
        None,
        None,
      ),
    }
  }

  fn price(&self) -> f64 {
    self.inner.calculate_price()
  }
}
