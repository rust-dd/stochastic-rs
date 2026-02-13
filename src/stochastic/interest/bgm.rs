//! A simple single-factor BGM (Brace–Gatarek–Musiela) model for forward rates.
//!
//! Produces a 2D array (`(xn, n)`) of forward rate paths. Each row represents a separate
//! forward rate, evolving over `n` time steps.
//!
//! # Parameters
//! - `lambda`: The drift/volatility multiplier for each forward rate.
//! - `x0`: Initial forward rates for each path.
//! - `xn`: Number of forward rates (rows) to simulate.
//! - `t`: Total time horizon.
//! - `n`: Number of time steps in the simulation.
//! - `m`: Batch size for parallel sampling (if used).

use ndarray::Array1;
use ndarray::Array2;

use crate::stochastic::noise::gn::Gn;
use crate::traits::FloatExt;
use crate::traits::ProcessExt;

pub struct BGM<T: FloatExt> {
  /// Drift/volatility multiplier for each forward rate.
  pub lambda: Array1<T>,
  /// Initial forward rates for each path.
  pub x0: Array1<T>,
  /// Number of forward rates (rows) to simulate.
  pub xn: usize,
  /// Total time horizon.
  pub t: Option<T>,
  /// Number of time steps in the simulation.
  pub n: usize,
  gn: Gn<T>,
}

impl<T: FloatExt> BGM<T> {
  pub fn new(lambda: Array1<T>, x0: Array1<T>, xn: usize, t: Option<T>, n: usize) -> Self {
    Self {
      lambda,
      x0,
      xn,
      t,
      n,
      gn: Gn::new(n - 1, t),
    }
  }
}

impl<T: FloatExt> ProcessExt<T> for BGM<T> {
  type Output = Array2<T>;

  fn sample(&self) -> Self::Output {
    let mut fwd = Array2::<T>::zeros((self.xn, self.n));

    for i in 0..self.xn {
      fwd[(i, 0)] = self.x0[i];
    }

    for i in 0..self.xn {
      let gn = &self.gn.sample();

      for j in 1..self.n {
        let f_old = fwd[(i, j - 1)];
        fwd[(i, j)] = f_old + f_old * self.lambda[i] * gn[j - 1];
      }
    }

    fwd
  }
}

#[cfg(feature = "python")]
#[pyo3::prelude::pyclass]
pub struct PyBGM {
  inner_f32: Option<BGM<f32>>,
  inner_f64: Option<BGM<f64>>,
}

#[cfg(feature = "python")]
#[pyo3::prelude::pymethods]
impl PyBGM {
  #[new]
  #[pyo3(signature = (lambda_, x0, xn, n, t=None, dtype=None))]
  fn new(
    lambda_: Vec<f64>, x0: Vec<f64>, xn: usize, n: usize, t: Option<f64>,
    dtype: Option<&str>,
  ) -> Self {
    match dtype.unwrap_or("f64") {
      "f32" => {
        let lambda_f32 = ndarray::Array1::from_vec(lambda_.iter().map(|&v| v as f32).collect());
        let x0_f32 = ndarray::Array1::from_vec(x0.iter().map(|&v| v as f32).collect());
        Self {
          inner_f32: Some(BGM::new(lambda_f32, x0_f32, xn, t.map(|v| v as f32), n)),
          inner_f64: None,
        }
      }
      _ => {
        let lambda_arr = ndarray::Array1::from_vec(lambda_);
        let x0_arr = ndarray::Array1::from_vec(x0);
        Self {
          inner_f32: None,
          inner_f64: Some(BGM::new(lambda_arr, x0_arr, xn, t, n)),
        }
      }
    }
  }

  fn sample<'py>(&self, py: pyo3::Python<'py>) -> pyo3::Py<pyo3::PyAny> {
    use numpy::IntoPyArray;
    use crate::traits::ProcessExt;
    use pyo3::IntoPyObjectExt;
    if let Some(ref inner) = self.inner_f64 {
      inner.sample().into_pyarray(py).into_py_any(py).unwrap()
    } else if let Some(ref inner) = self.inner_f32 {
      inner.sample().into_pyarray(py).into_py_any(py).unwrap()
    } else { unreachable!() }
  }

  fn sample_par<'py>(&self, py: pyo3::Python<'py>, m: usize) -> pyo3::Py<pyo3::PyAny> {
    use numpy::IntoPyArray;
    use crate::traits::ProcessExt;
    use pyo3::IntoPyObjectExt;
    if let Some(ref inner) = self.inner_f64 {
      let samples = inner.sample_par(m);
      pyo3::types::PyList::new(
        py,
        samples.iter().map(|s| s.clone().into_pyarray(py).into_py_any(py).unwrap()),
      ).unwrap().into_py_any(py).unwrap()
    } else if let Some(ref inner) = self.inner_f32 {
      let samples = inner.sample_par(m);
      pyo3::types::PyList::new(
        py,
        samples.iter().map(|s| s.clone().into_pyarray(py).into_py_any(py).unwrap()),
      ).unwrap().into_py_any(py).unwrap()
    } else { unreachable!() }
  }
}
