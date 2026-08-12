//! `PyBates`, split out of `bates.rs` to keep that file under the project's
//! 600-line cap (same reason `bates_tests.rs` was split out earlier). Pulled
//! in via `use super::*;`, so it shares `bates.rs`'s own imports
//! (`Bates1996`, `Unseeded`, the `#[cfg(feature = "python")]`-gated
//! `Deterministic`) exactly as `bates_tests.rs` does.

use super::*;

#[pyo3::prelude::pyclass]
pub struct PyBates {
  inner_f32: Option<Bates1996<f32, crate::traits::CallableDist<f32>>>,
  inner_f64: Option<Bates1996<f64, crate::traits::CallableDist<f64>>>,
  seeded_f32:
    Option<Bates1996<f32, crate::traits::CallableDist<f32>, crate::simd_rng::Deterministic>>,
  seeded_f64:
    Option<Bates1996<f64, crate::traits::CallableDist<f64>, crate::simd_rng::Deterministic>>,
}

#[pyo3::prelude::pymethods]
impl PyBates {
  #[new]
  #[pyo3(signature = (lambda_, k, alpha, beta, sigma, rho, distribution, n, mu=None, b=None, r=None, r_f=None, s0=None, v0=None, t=None, use_sym=None, seed=None, dtype=None))]
  fn new(
    lambda_: f64,
    k: f64,
    alpha: f64,
    beta: f64,
    sigma: f64,
    rho: f64,
    distribution: pyo3::Py<pyo3::PyAny>,
    n: usize,
    mu: Option<f64>,
    b: Option<f64>,
    r: Option<f64>,
    r_f: Option<f64>,
    s0: Option<f64>,
    v0: Option<f64>,
    t: Option<f64>,
    use_sym: Option<bool>,
    seed: Option<u64>,
    dtype: Option<&str>,
  ) -> Self {
    let mut s = Self {
      inner_f32: None,
      inner_f64: None,
      seeded_f32: None,
      seeded_f64: None,
    };
    match dtype.unwrap_or("f64") {
      "f32" => {
        let jump_dist = crate::traits::CallableDist::new(distribution);
        match seed {
          Some(sd) => {
            s.seeded_f32 = Some(Bates1996::new(
              mu.map(|v| v as f32),
              b.map(|v| v as f32),
              r.map(|v| v as f32),
              r_f.map(|v| v as f32),
              lambda_ as f32,
              k as f32,
              alpha as f32,
              beta as f32,
              sigma as f32,
              rho as f32,
              jump_dist,
              n,
              s0.map(|v| v as f32),
              v0.map(|v| v as f32),
              t.map(|v| v as f32),
              use_sym,
              Deterministic::new(sd),
            ));
          }
          None => {
            s.inner_f32 = Some(Bates1996::new(
              mu.map(|v| v as f32),
              b.map(|v| v as f32),
              r.map(|v| v as f32),
              r_f.map(|v| v as f32),
              lambda_ as f32,
              k as f32,
              alpha as f32,
              beta as f32,
              sigma as f32,
              rho as f32,
              jump_dist,
              n,
              s0.map(|v| v as f32),
              v0.map(|v| v as f32),
              t.map(|v| v as f32),
              use_sym,
              Unseeded,
            ));
          }
        }
      }
      _ => {
        let jump_dist = crate::traits::CallableDist::new(distribution);
        match seed {
          Some(sd) => {
            s.seeded_f64 = Some(Bates1996::new(
              mu,
              b,
              r,
              r_f,
              lambda_,
              k,
              alpha,
              beta,
              sigma,
              rho,
              jump_dist,
              n,
              s0,
              v0,
              t,
              use_sym,
              Deterministic::new(sd),
            ));
          }
          None => {
            s.inner_f64 = Some(Bates1996::new(
              mu, b, r, r_f, lambda_, k, alpha, beta, sigma, rho, jump_dist, n, s0, v0, t, use_sym,
              Unseeded,
            ));
          }
        }
      }
    }
    s
  }

  fn sample<'py>(&self, py: pyo3::Python<'py>) -> (pyo3::Py<pyo3::PyAny>, pyo3::Py<pyo3::PyAny>) {
    use numpy::IntoPyArray;
    use pyo3::IntoPyObjectExt;

    use crate::traits::ProcessExt;
    py_dispatch!(self, |inner| {
      let [a, b] = inner.sample();
      (
        a.into_pyarray(py).into_py_any(py).unwrap(),
        b.into_pyarray(py).into_py_any(py).unwrap(),
      )
    })
  }

  fn sample_par<'py>(
    &self,
    py: pyo3::Python<'py>,
    m: usize,
  ) -> (pyo3::Py<pyo3::PyAny>, pyo3::Py<pyo3::PyAny>) {
    use numpy::IntoPyArray;
    use numpy::ndarray::Array2;
    use pyo3::IntoPyObjectExt;

    use crate::traits::ProcessExt;
    py_dispatch!(self, |inner| {
      let samples = inner.sample_par(m);
      let n = samples[0][0].len();
      let mut r0 = Array2::zeros((m, n));
      let mut r1 = Array2::zeros((m, n));
      for (i, [a, b]) in samples.iter().enumerate() {
        r0.row_mut(i).assign(a);
        r1.row_mut(i).assign(b);
      }
      (
        r0.into_pyarray(py).into_py_any(py).unwrap(),
        r1.into_pyarray(py).into_py_any(py).unwrap(),
      )
    })
  }
}
