use numpy::IntoPyArray;
use numpy::PyReadonlyArray1;
use numpy::PyReadonlyArray2;
use pyo3::IntoPyObjectExt;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use stochastic_rs_core::simd_rng::Deterministic;

use super::*;
use crate::traits::Grid2D;

/// The leverage a Python caller hands over: a callable `L(t, s)`, a
/// `(spots, times, values)` triple of arrays, or any object carrying those
/// three as attributes — the quant crate's `LeverageSurface` is one. A
/// callable runs under the GIL at every step; a grid is interpolated in
/// Rust.
fn leverage_from_py(obj: &Bound<'_, PyAny>) -> PyResult<Fn2D<f64>> {
  if obj.is_callable() {
    return Ok(Fn2D::Py(obj.clone().unbind()));
  }
  type Triple<'py> = (
    PyReadonlyArray1<'py, f64>,
    PyReadonlyArray1<'py, f64>,
    PyReadonlyArray2<'py, f64>,
  );
  let (spots, times, values) = match obj.extract::<Triple<'_>>() {
    Ok(triple) => triple,
    Err(_) => (
      obj.getattr("spots")?.extract()?,
      obj.getattr("times")?.extract()?,
      obj.getattr("values")?.extract()?,
    ),
  };
  let spots = spots.as_array().to_owned();
  let times = times.as_array().to_owned();
  let values = values.as_array().to_owned();
  if spots.is_empty() || times.is_empty() {
    return Err(PyValueError::new_err(
      "leverage: spots and times must not be empty",
    ));
  }
  if spots.windows(2).into_iter().any(|w| w[0] >= w[1])
    || times.windows(2).into_iter().any(|w| w[0] >= w[1])
  {
    return Err(PyValueError::new_err(
      "leverage: spots and times must be strictly ascending",
    ));
  }
  if values.dim() != (times.len(), spots.len()) {
    return Err(PyValueError::new_err(format!(
      "leverage: values must have shape (times, spots) = ({}, {}), got {:?}",
      times.len(),
      spots.len(),
      values.dim()
    )));
  }
  Ok(Fn2D::Grid(Grid2D::new(times, spots, values)))
}

#[pyclass]
pub struct PyHestonSlv {
  inner: Option<HestonSlv<f64>>,
  seeded: Option<HestonSlv<f64, Deterministic>>,
}

#[pymethods]
impl PyHestonSlv {
  /// `leverage` is `L(t, s)`: a Python callable, a `(spots, times, values)`
  /// triple of arrays, or an object with those three attributes such as the
  /// `LeverageSurface` a `HestonSlvCalibrator` returns. `eta` is the mixing
  /// fraction the vol-of-vol is scaled by.
  #[new]
  #[pyo3(signature = (kappa, theta, sigma, rho, mu, eta, leverage, n, s0=None, v0=None, t=None, seed=None))]
  fn new(
    kappa: f64,
    theta: f64,
    sigma: f64,
    rho: f64,
    mu: f64,
    eta: f64,
    leverage: &Bound<'_, PyAny>,
    n: usize,
    s0: Option<f64>,
    v0: Option<f64>,
    t: Option<f64>,
    seed: Option<u64>,
  ) -> PyResult<Self> {
    if n < 2 {
      return Err(PyValueError::new_err("n must be at least 2"));
    }
    if !(0.0..=1.0).contains(&eta) {
      return Err(PyValueError::new_err("eta must lie in [0, 1]"));
    }
    if !(-1.0..=1.0).contains(&rho) {
      return Err(PyValueError::new_err("rho must lie in [-1, 1]"));
    }
    if kappa < 0.0 || theta < 0.0 || sigma < 0.0 {
      return Err(PyValueError::new_err(
        "kappa, theta and sigma must be non-negative",
      ));
    }
    if s0.is_some_and(|s| s <= 0.0) || v0.is_some_and(|v| v < 0.0) {
      return Err(PyValueError::new_err(
        "s0 must be positive and v0 non-negative",
      ));
    }
    let leverage = leverage_from_py(leverage)?;
    Ok(match seed {
      Some(s) => Self {
        inner: None,
        seeded: Some(HestonSlv::new(
          s0,
          v0,
          kappa,
          theta,
          sigma,
          rho,
          mu,
          eta,
          leverage,
          n,
          t,
          Deterministic::new(s),
        )),
      },
      None => Self {
        inner: Some(HestonSlv::new(
          s0, v0, kappa, theta, sigma, rho, mu, eta, leverage, n, t, Unseeded,
        )),
        seeded: None,
      },
    })
  }

  /// The reason a device kernel cannot carry this configuration: from
  /// Python the leverage is a callable or a grid, neither of which a kernel
  /// evaluates, so the process samples on the host.
  fn device_fallback(&self) -> Option<&'static str> {
    crate::py_dispatch_f64!(self, |inner| inner.device_fallback())
  }

  fn device_ready(&self) -> bool {
    self.device_fallback().is_none()
  }

  /// The leverage at `(t, s)`, as the sampler reads it.
  fn leverage(&self, t: f64, s: f64) -> f64 {
    crate::py_dispatch_f64!(self, |inner| inner.leverage.call(t, s))
  }

  /// One path as the pair `(s, v)` of arrays.
  fn sample<'py>(&self, py: Python<'py>) -> (Py<PyAny>, Py<PyAny>) {
    crate::py_dispatch_f64!(self, |inner| {
      let [s, v] = inner.sample();
      (
        s.into_pyarray(py).into_py_any(py).unwrap(),
        v.into_pyarray(py).into_py_any(py).unwrap(),
      )
    })
  }

  /// `m` paths as a pair of `(m, n)` arrays. A callable leverage is called
  /// back under the GIL at every step, so the parallel speed-up is bounded
  /// by those calls; a grid is interpolated in Rust.
  fn sample_par<'py>(&self, py: Python<'py>, m: usize) -> (Py<PyAny>, Py<PyAny>) {
    crate::py_dispatch_f64!(self, |inner| {
      // The callbacks re-attach to the interpreter from the worker threads, so
      // the GIL must be released here or the parallel sampler deadlocks.
      let samples = py.detach(|| inner.sample_par(m));
      let mut ss = ndarray::Array2::<f64>::zeros((m, inner.n));
      let mut vs = ndarray::Array2::<f64>::zeros((m, inner.n));
      for (i, [s, v]) in samples.iter().enumerate() {
        ss.row_mut(i).assign(s);
        vs.row_mut(i).assign(v);
      }
      (
        ss.into_pyarray(py).into_py_any(py).unwrap(),
        vs.into_pyarray(py).into_py_any(py).unwrap(),
      )
    })
  }
}
