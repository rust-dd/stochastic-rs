//! # Bgm
//!
//! $$
//! dL_i(t)=\mu_i(t)L_i(t)dt+\sigma_i(t)L_i(t)dW_t
//! $$
//!
use ndarray::Array1;
use ndarray::Array2;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::normal::SimdNormal;

use crate::traits::FloatExt;
use crate::traits::ProcessExt;

pub struct Bgm<T: FloatExt, S: SeedExt = Unseeded> {
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
  /// Seed strategy (compile-time: [`Unseeded`] or [`Deterministic`]).
  pub seed: S,
}

impl<T: FloatExt> Bgm<T> {
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
      seed: Unseeded,
    }
  }
}

impl<T: FloatExt> Bgm<T, Deterministic> {
  pub fn seeded(
    lambda: Array1<T>,
    x0: Array1<T>,
    xn: usize,
    t: Option<T>,
    n: usize,
    seed: u64,
  ) -> Self {
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
      seed: Deterministic::new(seed),
    }
  }
}

impl<T: FloatExt, S: SeedExt> ProcessExt<T> for Bgm<T, S> {
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
        .expect("Bgm row must be contiguous in memory");
      let tail = &mut row_slice[1..];
      let normal = SimdNormal::<T>::from_seed_source(T::zero(), sqrt_dt, &self.seed);
      normal.fill_slice_fast(tail);

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
pub struct PyBgm {
  inner_f32: Option<Bgm<f32>>,
  inner_f64: Option<Bgm<f64>>,
  seeded_f32: Option<Bgm<f32, crate::simd_rng::Deterministic>>,
  seeded_f64: Option<Bgm<f64, crate::simd_rng::Deterministic>>,
}

#[cfg(feature = "python")]
#[pyo3::prelude::pymethods]
impl PyBgm {
  #[new]
  #[pyo3(signature = (lambda_, x0, xn, n, t=None, seed=None, dtype=None))]
  fn new(
    lambda_: Vec<f64>,
    x0: Vec<f64>,
    xn: usize,
    n: usize,
    t: Option<f64>,
    seed: Option<u64>,
    dtype: Option<&str>,
  ) -> Self {
    match (seed, dtype.unwrap_or("f64")) {
      (Some(s), "f32") => {
        let lambda_f32 = ndarray::Array1::from_vec(lambda_.iter().map(|&v| v as f32).collect());
        let x0_f32 = ndarray::Array1::from_vec(x0.iter().map(|&v| v as f32).collect());
        Self {
          inner_f32: None,
          inner_f64: None,
          seeded_f32: Some(Bgm::seeded(
            lambda_f32,
            x0_f32,
            xn,
            t.map(|v| v as f32),
            n,
            s,
          )),
          seeded_f64: None,
        }
      }
      (Some(s), _) => {
        let lambda_arr = ndarray::Array1::from_vec(lambda_);
        let x0_arr = ndarray::Array1::from_vec(x0);
        Self {
          inner_f32: None,
          inner_f64: None,
          seeded_f32: None,
          seeded_f64: Some(Bgm::seeded(lambda_arr, x0_arr, xn, t, n, s)),
        }
      }
      (None, "f32") => {
        let lambda_f32 = ndarray::Array1::from_vec(lambda_.iter().map(|&v| v as f32).collect());
        let x0_f32 = ndarray::Array1::from_vec(x0.iter().map(|&v| v as f32).collect());
        Self {
          inner_f32: Some(Bgm::new(lambda_f32, x0_f32, xn, t.map(|v| v as f32), n)),
          inner_f64: None,
          seeded_f32: None,
          seeded_f64: None,
        }
      }
      (None, _) => {
        let lambda_arr = ndarray::Array1::from_vec(lambda_);
        let x0_arr = ndarray::Array1::from_vec(x0);
        Self {
          inner_f32: None,
          inner_f64: Some(Bgm::new(lambda_arr, x0_arr, xn, t, n)),
          seeded_f32: None,
          seeded_f64: None,
        }
      }
    }
  }

  fn sample<'py>(&self, py: pyo3::Python<'py>) -> pyo3::Py<pyo3::PyAny> {
    use numpy::IntoPyArray;
    use pyo3::IntoPyObjectExt;

    use crate::traits::ProcessExt;
    py_dispatch!(self, |inner| inner
      .sample()
      .into_pyarray(py)
      .into_py_any(py)
      .unwrap())
  }

  fn sample_par<'py>(&self, py: pyo3::Python<'py>, m: usize) -> pyo3::Py<pyo3::PyAny> {
    use numpy::IntoPyArray;
    use pyo3::IntoPyObjectExt;

    use crate::traits::ProcessExt;
    py_dispatch!(self, |inner| {
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
    })
  }
}

#[cfg(test)]
mod tests {
  use ndarray::Array1;

  use super::*;

  #[test]
  fn bgm_sample_runs() {
    let lambda = Array1::<f64>::from_vec(vec![0.2, 0.2, 0.2]);
    let x0 = Array1::<f64>::from_vec(vec![0.03, 0.035, 0.04]);
    let bgm = Bgm::<f64>::new(lambda, x0, 3, Some(1.0), 50);
    let path = bgm.sample();
    // Bgm produces a 2D matrix (n_rates × n_steps)
    assert_eq!(path.nrows(), 3);
    assert_eq!(path.ncols(), 50);
  }
}
