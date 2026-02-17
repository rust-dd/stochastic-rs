//! A multi-dimensional Wu–Zhang-style model, combining a forward rate and a stochastic
//! volatility process for each dimension.
//!
//! Produces a 2D array (`(2*xn, n)`) of forward rates and volatilities. The first `xn` rows
//! are the forward rates; the next `xn` rows are the volatilities, each evolving over `n` steps.
//!
//! # Parameters
//! - `alpha`: Mean reversion level for each dimension's volatility.
//! - `beta`: Mean reversion speed for each dimension's volatility.
//! - `nu`: Volatility-of-volatility parameter for each dimension.
//! - `lambda`: Parameter controlling how volatility impacts the forward rate.
//! - `x0`: Initial forward rates for each dimension.
//! - `v0`: Initial volatilities for each dimension.
//! - `xn`: Number of `(rate, vol)` pairs.
//! - `t`: Total time horizon.
//! - `n`: Number of time steps in the simulation.
//! - `m`: Batch size for parallel sampling (if used).

use ndarray::Array1;
use ndarray::Array2;

use crate::stochastic::noise::gn::Gn;
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
  gn: Gn<T>,
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
      gn: Gn::new(n - 1, t),
    }
  }
}

impl<T: FloatExt> ProcessExt<T> for WuZhangD<T> {
  type Output = Array2<T>;

  fn sample(&self) -> Self::Output {
    let dt = self.gn.dt();
    let mut fv = Array2::<T>::zeros((2 * self.xn, self.n));

    for i in 0..self.xn {
      fv[(i, 0)] = self.x0[i];
      fv[(i + self.xn, 0)] = self.v0[i];
    }

    for i in 0..self.xn {
      let gn_f = &self.gn.sample();
      let gn_v = &self.gn.sample();

      for j in 1..self.n {
        let v_old = fv[(i + self.xn, j - 1)].max(T::zero());
        let f_old = fv[(i, j - 1)].max(T::zero());

        let dv =
          (self.alpha[i] - self.beta[i] * v_old) * dt + self.nu[i] * v_old.sqrt() * gn_v[j - 1];

        let v_new = (v_old + dv).max(T::zero());
        fv[(i + self.xn, j)] = v_new;

        let df = f_old * self.lambda[i] * v_new.sqrt() * gn_f[j - 1];
        fv[(i, j)] = f_old + df;
      }
    }

    fv
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
