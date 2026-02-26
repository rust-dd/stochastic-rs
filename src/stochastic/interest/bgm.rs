//! # BGM
//!
//! $$
//! dL_i(t)=\mu_i(t)L_i(t)dt+\sigma_i(t)L_i(t)dW_t
//! $$
//!
use ndarray::Array1;
use ndarray::Array2;

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
}

impl<T: FloatExt> BGM<T> {
  pub fn new(lambda: Array1<T>, x0: Array1<T>, xn: usize, t: Option<T>, n: usize) -> Self {
    assert_eq!(
      lambda.len(),
      xn,
      "lambda length ({}) must match xn ({})",
      lambda.len(),
      xn
    );
    assert_eq!(
      x0.len(),
      xn,
      "x0 length ({}) must match xn ({})",
      x0.len(),
      xn
    );
    Self {
      lambda,
      x0,
      xn,
      t,
      n,
    }
  }
}

impl<T: FloatExt> ProcessExt<T> for BGM<T> {
  type Output = Array2<T>;

  fn sample(&self) -> Self::Output {
    let mut fwd = Array2::<T>::zeros((self.xn, self.n));
    if self.n == 0 {
      return fwd;
    }

    for i in 0..self.xn {
      fwd[(i, 0)] = self.x0[i];
    }

    if self.n == 1 {
      return fwd;
    }

    let n_increments = self.n - 1;
    let sqrt_dt = (self.t.unwrap_or(T::one()) / T::from_usize_(n_increments)).sqrt();

    for i in 0..self.xn {
      let mut row = fwd.row_mut(i);
      let row_slice = row
        .as_slice_mut()
        .expect("BGM row must be contiguous in memory");
      let tail = &mut row_slice[1..];
      T::fill_standard_normal_slice(tail);
      for z in tail.iter_mut() {
        *z = *z * sqrt_dt;
      }

      for j in 1..self.n {
        let f_old = row_slice[j - 1];
        row_slice[j] = f_old + f_old * self.lambda[i] * row_slice[j];
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
    lambda_: Vec<f64>,
    x0: Vec<f64>,
    xn: usize,
    n: usize,
    t: Option<f64>,
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
    use pyo3::IntoPyObjectExt;

    use crate::traits::ProcessExt;
    if let Some(ref inner) = self.inner_f64 {
      inner.sample().into_pyarray(py).into_py_any(py).unwrap()
    } else if let Some(ref inner) = self.inner_f32 {
      inner.sample().into_pyarray(py).into_py_any(py).unwrap()
    } else {
      unreachable!()
    }
  }

  fn sample_par<'py>(&self, py: pyo3::Python<'py>, m: usize) -> pyo3::Py<pyo3::PyAny> {
    use numpy::IntoPyArray;
    use pyo3::IntoPyObjectExt;

    use crate::traits::ProcessExt;
    if let Some(ref inner) = self.inner_f64 {
      let samples = inner.sample_par(m);
      pyo3::types::PyList::new(
        py,
        samples
          .iter()
          .map(|s| s.clone().into_pyarray(py).into_py_any(py).unwrap()),
      )
      .unwrap()
      .into_py_any(py)
      .unwrap()
    } else if let Some(ref inner) = self.inner_f32 {
      let samples = inner.sample_par(m);
      pyo3::types::PyList::new(
        py,
        samples
          .iter()
          .map(|s| s.clone().into_pyarray(py).into_py_any(py).unwrap()),
      )
      .unwrap()
      .into_py_any(py)
      .unwrap()
    } else {
      unreachable!()
    }
  }
}
