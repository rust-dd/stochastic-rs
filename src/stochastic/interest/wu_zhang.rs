//! # Wu Zhang
//!
//! $$
//! dX_t=K(\Theta-X_t)dt+\sqrt{A+BX_t}\,dW_t,\quad r_t=\ell_0+\ell^\top X_t
//! $$
//!
use ndarray::Array1;
use ndarray::Array2;
use ndarray::Axis;

use crate::traits::FloatExt;
use crate::traits::ProcessExt;

pub struct WuZhangD<T: FloatExt> {
  /// Mean reversion level for each dimension's volatility.
  pub alpha: Array1<T>,
  /// Mean reversion speed for each dimension's volatility.
  pub beta: Array1<T>,
  /// Volatility of volatility for each dimension.
  pub nu: Array1<T>,
  /// Parameter controlling the impact of volatility on the forward rate.
  pub lambda: Array1<T>,
  /// Initial forward rates for each dimension.
  pub x0: Array1<T>,
  /// Initial volatilities for each dimension.
  pub v0: Array1<T>,
  /// Number of (rate, vol) pairs.
  pub xn: usize,
  /// Total time horizon.
  pub t: Option<T>,
  /// Number of time steps in the simulation.
  pub n: usize,
}

impl<T: FloatExt> WuZhangD<T> {
  pub fn new(
    alpha: Array1<T>,
    beta: Array1<T>,
    nu: Array1<T>,
    lambda: Array1<T>,
    x0: Array1<T>,
    v0: Array1<T>,
    xn: usize,
    t: Option<T>,
    n: usize,
  ) -> Self {
    assert_eq!(
      alpha.len(),
      xn,
      "alpha length ({}) must match xn ({})",
      alpha.len(),
      xn
    );
    assert_eq!(
      beta.len(),
      xn,
      "beta length ({}) must match xn ({})",
      beta.len(),
      xn
    );
    assert_eq!(
      nu.len(),
      xn,
      "nu length ({}) must match xn ({})",
      nu.len(),
      xn
    );
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
    assert_eq!(
      v0.len(),
      xn,
      "v0 length ({}) must match xn ({})",
      v0.len(),
      xn
    );
    assert!(
      alpha.iter().all(|&x| x >= T::zero()),
      "alpha entries must be non-negative"
    );
    assert!(
      beta.iter().all(|&x| x >= T::zero()),
      "beta entries must be non-negative"
    );
    assert!(
      nu.iter().all(|&x| x >= T::zero()),
      "nu entries must be non-negative"
    );
    assert!(
      v0.iter().all(|&x| x >= T::zero()),
      "v0 entries must be non-negative"
    );
    Self {
      alpha,
      beta,
      nu,
      lambda,
      x0,
      v0,
      xn,
      t,
      n,
    }
  }
}

impl<T: FloatExt> ProcessExt<T> for WuZhangD<T> {
  type Output = Array2<T>;

  fn sample(&self) -> Self::Output {
    let dt = if self.n > 1 {
      self.t.unwrap_or(T::one()) / T::from_usize_(self.n - 1)
    } else {
      T::zero()
    };
    let sqrt_dt = dt.sqrt();
    let mut fv = Array2::<T>::zeros((2 * self.xn, self.n));
    let (mut f_rows, mut v_rows) = fv.view_mut().split_at(Axis(0), self.xn);
    for i in 0..self.xn {
      let mut f_row = f_rows.row_mut(i);
      let mut v_row = v_rows.row_mut(i);
      let f_slice = f_row
        .as_slice_mut()
        .expect("WuZhang forward row must be contiguous in memory");
      let v_slice = v_row
        .as_slice_mut()
        .expect("WuZhang volatility row must be contiguous in memory");

      f_slice[0] = self.x0[i];
      v_slice[0] = self.v0[i];

      if self.n <= 1 {
        continue;
      }

      {
        let f_tail = &mut f_slice[1..];
        T::fill_standard_normal_scaled_slice(f_tail, sqrt_dt);
      }
      {
        let v_tail = &mut v_slice[1..];
        T::fill_standard_normal_scaled_slice(v_tail, sqrt_dt);
      }

      for j in 1..self.n {
        let v_old = v_slice[j - 1].max(T::zero());
        let f_old = f_slice[j - 1].max(T::zero());
        let d_w_v = v_slice[j];
        let d_w_f = f_slice[j];

        let dv = self.beta[i] * (self.alpha[i] - v_old) * dt + self.nu[i] * v_old.sqrt() * d_w_v;
        let v_new = (v_old + dv).max(T::zero());
        v_slice[j] = v_new;

        let df = f_old * self.lambda[i] * v_new.sqrt() * d_w_f;
        f_slice[j] = f_old + df;
      }
    }

    fv
  }
}

#[cfg(test)]
mod tests {
  use ndarray::array;

  use super::*;
  use crate::traits::ProcessExt;

  #[test]
  fn cir_drift_uses_mean_reversion_level_directly() {
    let model = WuZhangD::new(
      array![2.0f64],
      array![0.5f64],
      array![0.0f64],
      array![0.0f64],
      array![1.0f64],
      array![0.0f64],
      1,
      Some(2.0),
      3,
    );
    let fv = model.sample();
    let v = fv.row(1);
    assert!((v[1] - 1.0).abs() < 1e-12);
    assert!((v[2] - 1.5).abs() < 1e-12);
  }

  #[test]
  #[should_panic(expected = "v0 entries must be non-negative")]
  fn negative_initial_volatility_panics() {
    let _ = WuZhangD::new(
      array![1.0f64],
      array![1.0f64],
      array![0.1f64],
      array![0.0f64],
      array![1.0f64],
      array![-0.1f64],
      1,
      Some(1.0),
      16,
    );
  }
}

#[cfg(feature = "python")]
#[pyo3::prelude::pyclass]
pub struct PyWuZhangD {
  inner_f32: Option<WuZhangD<f32>>,
  inner_f64: Option<WuZhangD<f64>>,
}

#[cfg(feature = "python")]
#[pyo3::prelude::pymethods]
impl PyWuZhangD {
  #[new]
  #[pyo3(signature = (alpha, beta, nu, lambda_, x0, v0, xn, n, t=None, dtype=None))]
  fn new(
    alpha: Vec<f64>,
    beta: Vec<f64>,
    nu: Vec<f64>,
    lambda_: Vec<f64>,
    x0: Vec<f64>,
    v0: Vec<f64>,
    xn: usize,
    n: usize,
    t: Option<f64>,
    dtype: Option<&str>,
  ) -> Self {
    match dtype.unwrap_or("f64") {
      "f32" => {
        let to_f32_arr =
          |v: Vec<f64>| ndarray::Array1::from_vec(v.iter().map(|&x| x as f32).collect());
        Self {
          inner_f32: Some(WuZhangD::new(
            to_f32_arr(alpha),
            to_f32_arr(beta),
            to_f32_arr(nu),
            to_f32_arr(lambda_),
            to_f32_arr(x0),
            to_f32_arr(v0),
            xn,
            t.map(|v| v as f32),
            n,
          )),
          inner_f64: None,
        }
      }
      _ => {
        let to_arr = |v: Vec<f64>| ndarray::Array1::from_vec(v);
        Self {
          inner_f32: None,
          inner_f64: Some(WuZhangD::new(
            to_arr(alpha),
            to_arr(beta),
            to_arr(nu),
            to_arr(lambda_),
            to_arr(x0),
            to_arr(v0),
            xn,
            t,
            n,
          )),
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
