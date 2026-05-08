//! PyO3 wrappers for `stochastic-rs-copulas`.
//!
//! Bivariate Archimedean copulas (Clayton, Gumbel, Frank, Independence) wrapped
//! as `#[pyclass]`. Each exposes `theta` / `tau` setters, `pdf` / `cdf` /
//! `sample` over numpy arrays.
//!
//! The multivariate Gaussian copula is intentionally **not** wrapped because it
//! depends on `ndarray-linalg` (openblas), which is not part of the default
//! Python build.

#![cfg(feature = "python")]
#![allow(clippy::too_many_arguments)]

use ndarray::Array2;
use numpy::IntoPyArray;
use numpy::PyReadonlyArray2;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::traits::BivariateExt;

fn err_to_py(e: Box<dyn std::error::Error>) -> PyErr {
  PyValueError::new_err(e.to_string())
}

macro_rules! py_bivariate {
  ($py_name:ident, $inner_path:path, $name_str:literal, $ctor:expr) => {
    #[pyclass(name = $name_str, unsendable)]
    pub struct $py_name {
      inner: $inner_path,
    }

    #[pymethods]
    impl $py_name {
      #[new]
      #[pyo3(signature = (theta=None, tau=None))]
      fn new(theta: Option<f64>, tau: Option<f64>) -> Self {
        let mut inner = ($ctor)();
        if let Some(t) = theta {
          BivariateExt::set_theta(&mut inner, t);
        }
        if let Some(t) = tau {
          BivariateExt::set_tau(&mut inner, t);
        }
        Self { inner }
      }

      fn theta(&self) -> Option<f64> {
        BivariateExt::theta(&self.inner)
      }

      fn tau(&self) -> Option<f64> {
        BivariateExt::tau(&self.inner)
      }

      fn set_theta(&mut self, theta: f64) {
        BivariateExt::set_theta(&mut self.inner, theta);
      }

      fn set_tau(&mut self, tau: f64) {
        BivariateExt::set_tau(&mut self.inner, tau);
      }

      /// Solve for theta from the currently-set tau (Kendall inversion).
      fn compute_theta(&mut self) -> f64 {
        let t = BivariateExt::compute_theta(&self.inner);
        BivariateExt::set_theta(&mut self.inner, t);
        t
      }

      fn fit<'py>(&mut self, x: PyReadonlyArray2<'py, f64>) -> PyResult<()> {
        let arr = x.as_array().to_owned();
        BivariateExt::fit(&mut self.inner, &arr).map_err(err_to_py)
      }

      fn pdf<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
      ) -> PyResult<pyo3::Bound<'py, numpy::PyArray1<f64>>> {
        let arr = x.as_array().to_owned();
        let out = BivariateExt::pdf(&self.inner, &arr).map_err(err_to_py)?;
        Ok(out.into_pyarray(py))
      }

      fn cdf<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
      ) -> PyResult<pyo3::Bound<'py, numpy::PyArray1<f64>>> {
        let arr = x.as_array().to_owned();
        let out = BivariateExt::cdf(&self.inner, &arr).map_err(err_to_py)?;
        Ok(out.into_pyarray(py))
      }

      #[pyo3(signature = (n, seed=None))]
      fn sample<'py>(
        &mut self,
        py: Python<'py>,
        n: usize,
        seed: Option<u64>,
      ) -> PyResult<pyo3::Bound<'py, numpy::PyArray2<f64>>> {
        let arr: Array2<f64> = match seed {
          Some(s) => BivariateExt::sample_with_seed(&mut self.inner, n, s).map_err(err_to_py)?,
          None => BivariateExt::sample(&mut self.inner, n).map_err(err_to_py)?,
        };
        Ok(arr.into_pyarray(py))
      }
    }
  };
}

py_bivariate!(
  PyClayton,
  crate::bivariate::clayton::Clayton,
  "Clayton",
  crate::bivariate::clayton::Clayton::new
);
py_bivariate!(PyGumbel, crate::bivariate::gumbel::Gumbel, "Gumbel", || {
  crate::bivariate::gumbel::Gumbel::new(None, None)
});
py_bivariate!(PyFrank, crate::bivariate::frank::Frank, "Frank", || {
  crate::bivariate::frank::Frank::new(None, None)
});
py_bivariate!(
  PyIndependence,
  crate::bivariate::independence::Independence,
  "Independence",
  crate::bivariate::independence::Independence::new
);

#[pyclass(name = "EmpiricalCopula2D", unsendable)]
pub struct PyEmpiricalCopula2D {
  inner: crate::empirical::EmpiricalCopula2D,
}

#[pymethods]
impl PyEmpiricalCopula2D {
  /// Build a 2D empirical copula from two equal-length series via rank-transform.
  #[new]
  fn new<'py>(
    x: numpy::PyReadonlyArray1<'py, f64>,
    y: numpy::PyReadonlyArray1<'py, f64>,
  ) -> Self {
    let x_arr = x.as_array().to_owned();
    let y_arr = y.as_array().to_owned();
    Self {
      inner: crate::empirical::EmpiricalCopula2D::new_from_two_series(&x_arr, &y_arr),
    }
  }

  fn rank_data<'py>(&self, py: Python<'py>) -> pyo3::Bound<'py, numpy::PyArray2<f64>> {
    self.inner.rank_data.clone().into_pyarray(py)
  }

  fn sample<'py>(&self, py: Python<'py>, n: usize) -> pyo3::Bound<'py, numpy::PyArray2<f64>> {
    self.inner.sample(n).into_pyarray(py)
  }
}

/// Kendall's τ pairwise matrix from an `(n, k)` data matrix.
#[pyfunction]
pub fn kendall_tau_matrix<'py>(
  py: Python<'py>,
  data: numpy::PyReadonlyArray2<'py, f64>,
) -> pyo3::Bound<'py, numpy::PyArray2<f64>> {
  let arr = data.as_array().to_owned();
  let out = crate::correlation::kendall_tau(&arr);
  out.into_pyarray(py)
}

/// Convert a Kendall τ matrix to a Gaussian copula correlation matrix
/// elementwise via $\rho_{ij} = \sin(\pi \tau_{ij} / 2)$.
#[pyfunction]
pub fn tau_matrix_to_corr_matrix<'py>(
  py: Python<'py>,
  tau: numpy::PyReadonlyArray2<'py, f64>,
) -> pyo3::Bound<'py, numpy::PyArray2<f64>> {
  let arr = tau.as_array().to_owned();
  let out = crate::correlation::tau_matrix_to_corr_matrix(&arr);
  out.into_pyarray(py)
}

#[pyfunction]
pub fn tau_to_corr(tau: f64) -> f64 {
  crate::correlation::tau_to_corr(tau)
}

#[pyfunction]
pub fn corr_to_tau(rho: f64) -> f64 {
  crate::correlation::corr_to_tau(rho)
}
