use std::panic::AssertUnwindSafe;
use std::panic::catch_unwind;

use numpy::IntoPyArray;
use numpy::PyReadonlyArray1;
use numpy::PyReadonlyArray2;
use pyo3::IntoPyObjectExt;
use pyo3::exceptions::PyRuntimeError;
use pyo3::exceptions::PyTypeError;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use stochastic_rs_core::simd_rng::Deterministic;

use super::*;
use crate::traits::Grid2D;

/// Fn2D's Rust API is infallible, so a failed Python callback unwinds.
/// Translate it at the Python boundary, including panics relayed by Rayon.
fn callback_result<R>(operation: impl FnOnce() -> R) -> PyResult<R> {
  catch_unwind(AssertUnwindSafe(operation)).map_err(|payload| {
    let message = payload
      .downcast_ref::<String>()
      .map(String::as_str)
      .or_else(|| payload.downcast_ref::<&str>().copied())
      .unwrap_or("unknown sampler failure");
    PyRuntimeError::new_err(format!("HestonSlv evaluation failed: {message}"))
  })
}

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
  let (spots, times, values) = if obj.is_instance_of::<PyTuple>() {
    obj.extract::<Triple<'_>>().map_err(|e| {
      PyValueError::new_err(format!(
        "leverage: a (spots, times, values) triple must hold float64 arrays of one, one and two \
         dimensions: {e}"
      ))
    })?
  } else if obj.hasattr("spots")? && obj.hasattr("times")? && obj.hasattr("values")? {
    (
      obj.getattr("spots")?.extract()?,
      obj.getattr("times")?.extract()?,
      obj.getattr("values")?.extract()?,
    )
  } else {
    return Err(PyTypeError::new_err(
      "leverage must be a callable L(t, s), a (spots, times, values) triple of arrays, or an \
       object with spots, times and values attributes such as LeverageSurface",
    ));
  };
  let spots = spots.as_array().to_owned();
  let times = times.as_array().to_owned();
  let values = values.as_array().to_owned();
  if spots.is_empty() || times.is_empty() {
    return Err(PyValueError::new_err(
      "leverage: spots and times must not be empty",
    ));
  }
  if spots
    .iter()
    .chain(times.iter())
    .chain(values.iter())
    .any(|x| !x.is_finite())
  {
    return Err(PyValueError::new_err(
      "leverage: spots, times and values must be finite",
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

#[doc(hidden)]
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
    stochastic_rs_distributions::python::value_error_on_panic(|| -> PyResult<Self> {
      if n < 2 {
        return Err(PyValueError::new_err("n must be at least 2"));
      }
      if !(0.0..=1.0).contains(&eta) {
        return Err(PyValueError::new_err("eta must lie in [0, 1]"));
      }
      if !(-1.0..=1.0).contains(&rho) {
        return Err(PyValueError::new_err("rho must lie in [-1, 1]"));
      }
      if [kappa, theta, sigma]
        .iter()
        .any(|p| !(p.is_finite() && *p >= 0.0))
      {
        return Err(PyValueError::new_err(
          "kappa, theta and sigma must be finite and non-negative",
        ));
      }
      if !mu.is_finite() {
        return Err(PyValueError::new_err("mu must be finite"));
      }
      if s0.is_some_and(|s| !(s.is_finite() && s > 0.0))
        || v0.is_some_and(|v| !(v.is_finite() && v >= 0.0))
      {
        return Err(PyValueError::new_err(
          "s0 must be finite and positive, v0 finite and non-negative",
        ));
      }
      if t.is_some_and(|t| !(t.is_finite() && t > 0.0)) {
        return Err(PyValueError::new_err("t must be finite and positive"));
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
    })?
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
  fn leverage(&self, t: f64, s: f64) -> PyResult<f64> {
    callback_result(|| crate::py_dispatch_f64!(self, |inner| inner.leverage.call(t, s)))
  }

  /// One path as the pair `(s, v)` of arrays.
  /// A failed leverage callback raises `RuntimeError` with its error message.
  fn sample<'py>(&self, py: Python<'py>) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
    stochastic_rs_distributions::python::runtime_error_on_panic(
      || -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        let [s, v] = callback_result(|| crate::py_dispatch_f64!(self, |inner| inner.sample()))?;
        Ok((
          s.into_pyarray(py).into_py_any(py)?,
          v.into_pyarray(py).into_py_any(py)?,
        ))
      },
    )?
  }

  /// `m` paths as a pair of `(m, n)` arrays. A callable leverage is called
  /// back under the GIL at every step, so the parallel speed-up is bounded
  /// by those calls; a grid is interpolated in Rust.
  /// A failed leverage callback raises `RuntimeError` with its error message.
  fn sample_par<'py>(&self, py: Python<'py>, m: usize) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
    stochastic_rs_distributions::python::runtime_error_on_panic(
      || -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        crate::py_dispatch_f64!(self, |inner| {
          // The callbacks re-attach to the interpreter from the worker threads, so
          // the GIL must be released here or the parallel sampler deadlocks.
          let samples = callback_result(|| py.detach(|| inner.sample_par(m)))?;
          let mut ss = ndarray::Array2::<f64>::zeros((m, inner.n));
          let mut vs = ndarray::Array2::<f64>::zeros((m, inner.n));
          for (i, [s, v]) in samples.iter().enumerate() {
            ss.row_mut(i).assign(s);
            vs.row_mut(i).assign(v);
          }
          Ok((
            ss.into_pyarray(py).into_py_any(py)?,
            vs.into_pyarray(py).into_py_any(py)?,
          ))
        })
      },
    )?
  }
}
