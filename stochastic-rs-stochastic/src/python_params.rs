//! The Python classes' optional kwargs as Rust values: the macros' `=> map` functions and parsers.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::params::Drift;
use crate::params::HoLeeDrift;
use crate::params::Horizon;
use crate::params::VarianceBoundary;
use crate::traits::Fn1D;

/// `"full_truncation"`, `"absorption"` or `"reflect"`; any other value is a `ValueError`.
pub struct PyVarianceBoundary(pub VarianceBoundary);

impl<'a, 'py> FromPyObject<'a, 'py> for PyVarianceBoundary {
  type Error = PyErr;

  fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
    match ob.extract::<String>().as_deref() {
      Ok("full_truncation") => Ok(Self(VarianceBoundary::FullTruncation)),
      Ok("absorption") => Ok(Self(VarianceBoundary::Absorption)),
      Ok("reflect") => Ok(Self(VarianceBoundary::Reflect)),
      _ => Err(PyValueError::new_err(format!(
        "boundary must be \"full_truncation\", \"absorption\" or \"reflect\", got {}",
        ob.repr()?
      ))),
    }
  }
}

/// `boundary=None` is full truncation.
pub fn boundary(boundary: Option<PyVarianceBoundary>) -> VarianceBoundary {
  boundary.map_or(VarianceBoundary::default(), |b| b.0)
}

/// `clamp=None` clamps.
pub fn clamp(clamp: Option<bool>) -> bool {
  clamp.unwrap_or(true)
}

/// Exactly one of `mu`, `b`, or the `(r, r_f)` pair.
pub fn drift_from_kwargs(
  mu: Option<f64>,
  b: Option<f64>,
  r: Option<f64>,
  r_f: Option<f64>,
) -> PyResult<Drift<f64>> {
  match (mu, b, r, r_f) {
    (Some(mu), None, None, None) => Ok(Drift::Mu(mu)),
    (None, Some(b), None, None) => Ok(Drift::CostOfCarry(b)),
    (None, None, Some(r), Some(r_f)) => Ok(Drift::Rates { r, r_f }),
    _ => Err(PyValueError::new_err(
      "give exactly one drift: mu, or b, or both r and r_f",
    )),
  }
}

/// Exactly one of `n` (an event count) and `t_max` (a time horizon).
pub fn horizon_from_kwargs(n: Option<usize>, t_max: Option<f64>) -> PyResult<Horizon<f64>> {
  match (n, t_max) {
    (Some(n), None) => Ok(Horizon::Events(n)),
    (None, Some(t)) => Ok(Horizon::Until(t)),
    _ => Err(PyValueError::new_err("give exactly one of n and t_max")),
  }
}

/// Exactly one of `theta` and the `forward_curve`.
pub fn ho_lee_drift_from_kwargs(
  theta: Option<f64>,
  forward_curve: Option<Py<PyAny>>,
) -> PyResult<HoLeeDrift<f64>> {
  match (theta, forward_curve) {
    (Some(theta), None) => Ok(HoLeeDrift::Constant(theta)),
    (None, Some(f)) => Ok(HoLeeDrift::ForwardCurve(Fn1D::Py(f))),
    _ => Err(PyValueError::new_err(
      "give exactly one drift: theta or forward_curve",
    )),
  }
}

/// Each rate rounded to `f32` on its own, as the scalar kwargs are.
pub fn drift_f32(drift: Drift<f64>) -> Drift<f32> {
  match drift {
    Drift::Mu(mu) => Drift::Mu(mu as f32),
    Drift::CostOfCarry(b) => Drift::CostOfCarry(b as f32),
    Drift::Rates { r, r_f } => Drift::Rates {
      r: r as f32,
      r_f: r_f as f32,
    },
  }
}

/// The horizon rounded to `f32`; an event count is unchanged.
pub fn horizon_f32(horizon: Horizon<f64>) -> Horizon<f32> {
  match horizon {
    Horizon::Events(n) => Horizon::Events(n),
    Horizon::Until(t) => Horizon::Until(t as f32),
  }
}
