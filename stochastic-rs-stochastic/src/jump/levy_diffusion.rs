//! # Levy Diffusion
//!
//! $$
//! dX_t=\mu_tdt+\sigma_t dW_t+dL_t
//! $$
//!
use ndarray::Array1;
use rand_distr::Distribution;
#[cfg(feature = "python")]
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::normal::SimdNormal;

use crate::buffer::array1_from_fill;
use crate::process::cpoisson::CompoundPoisson;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

pub struct LevyDiffusion<T, D, S: SeedExt = Unseeded>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
{
  pub gamma: T,
  pub sigma: T,
  pub n: usize,
  pub x0: Option<T>,
  pub t: Option<T>,
  pub cpoisson: CompoundPoisson<T, D>,
  pub seed: S,
}

impl<T, D, S: SeedExt> LevyDiffusion<T, D, S>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
{
  pub fn new(
    gamma: T,
    sigma: T,
    n: usize,
    x0: Option<T>,
    t: Option<T>,
    cpoisson: CompoundPoisson<T, D>,
    seed: S,
  ) -> Self {
    Self {
      gamma,
      sigma,
      n,
      x0,
      t,
      cpoisson,
      seed,
    }
  }
}

impl<T, D, S: SeedExt> ProcessExt<T> for LevyDiffusion<T, D, S>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
{
  type Output = Array1<T>;
  type Sampler<'s>
    = LevyDiffusionSampler<'s, T, D>
  where
    Self: 's;

  fn sampler(&self) -> LevyDiffusionSampler<'_, T, D> {
    // The diffusion source is owned and derived from `self.seed`; the
    // compound-Poisson jump driver is borrowed and re-drawn per fill exactly
    // as the legacy `sample()` did (it rebuilds its own RNG from
    // `cpoisson.seed` each call). The two seed sources are independent, so the
    // first fill reproduces the legacy stream bit-for-bit.
    let dt = if self.n > 1 {
      self.t.unwrap_or(T::one()) / T::from_usize_(self.n - 1)
    } else {
      T::zero()
    };
    LevyDiffusionSampler {
      n: self.n,
      sigma: self.sigma,
      x0: self.x0.unwrap_or(T::zero()),
      dt,
      drift_dt: self.gamma * dt,
      cpoisson: &self.cpoisson,
      normal: SimdNormal::<T>::new(T::zero(), dt.sqrt(), &self.seed),
    }
  }
}

/// Reusable [`LevyDiffusion`] sampling state: owns the Gaussian diffusion
/// source and borrows the compound-Poisson jump driver, so a Monte-Carlo loop
/// pays the `SimdNormal` setup once.
#[doc(hidden)]
pub struct LevyDiffusionSampler<'a, T, D>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
{
  n: usize,
  sigma: T,
  x0: T,
  dt: T,
  drift_dt: T,
  cpoisson: &'a CompoundPoisson<T, D>,
  normal: SimdNormal<T>,
}

impl<T, D> LevyDiffusionSampler<'_, T, D>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
{
  fn fill_path(&mut self, out: &mut [T]) {
    if out.is_empty() {
      return;
    }

    let jump_increments = self.cpoisson.sample_grid_increments(out.len(), self.dt);
    let mut gn = Array1::<T>::zeros(out.len() - 1);
    if let Some(gn_slice) = gn.as_slice_mut() {
      self.normal.fill_slice_fast(gn_slice);
    }

    out[0] = self.x0;

    for i in 1..out.len() {
      out[i] = out[i - 1] + self.drift_dt + self.sigma * gn[i - 1] + jump_increments[i];
    }
  }
}

impl<T, D> PathSampler<T> for LevyDiffusionSampler<'_, T, D>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
{
  type Output = Array1<T>;

  fn sample_into(&mut self, out: &mut Array1<T>) {
    self.fill_path(
      out
        .as_slice_mut()
        .expect("LevyDiffusion output must be contiguous"),
    );
  }

  fn sample(&mut self) -> Array1<T> {
    let n = self.n;
    array1_from_fill(n, |out| self.fill_path(out))
  }
}

#[cfg(feature = "python")]
#[pyo3::prelude::pyclass]
pub struct PyLevyDiffusion {
  inner_f32: Option<LevyDiffusion<f32, crate::traits::CallableDist<f32>>>,
  inner_f64: Option<LevyDiffusion<f64, crate::traits::CallableDist<f64>>>,
  seeded_f32:
    Option<LevyDiffusion<f32, crate::traits::CallableDist<f32>, crate::simd_rng::Deterministic>>,
  seeded_f64:
    Option<LevyDiffusion<f64, crate::traits::CallableDist<f64>, crate::simd_rng::Deterministic>>,
}

#[cfg(feature = "python")]
#[pyo3::prelude::pymethods]
impl PyLevyDiffusion {
  #[new]
  #[pyo3(signature = (gamma_, sigma, distribution, lambda_, n, x0=None, t=None, seed=None, dtype=None))]
  fn new(
    gamma_: f64,
    sigma: f64,
    distribution: pyo3::Py<pyo3::PyAny>,
    lambda_: f64,
    n: usize,
    x0: Option<f64>,
    t: Option<f64>,
    seed: Option<u64>,
    dtype: Option<&str>,
  ) -> Self {
    use crate::process::poisson::Poisson;
    let mut s = Self {
      inner_f32: None,
      inner_f64: None,
      seeded_f32: None,
      seeded_f64: None,
    };
    match dtype.unwrap_or("f64") {
      "f32" => {
        let cpoisson = CompoundPoisson::new(
          crate::traits::CallableDist::new(distribution),
          Poisson::new(lambda_ as f32, Some(n), t.map(|v| v as f32), Unseeded),
          Unseeded,
        );
        match seed {
          Some(sd) => {
            s.seeded_f32 = Some(LevyDiffusion::new(
              gamma_ as f32,
              sigma as f32,
              n,
              x0.map(|v| v as f32),
              t.map(|v| v as f32),
              cpoisson,
              Deterministic::new(sd),
            ));
          }
          None => {
            s.inner_f32 = Some(LevyDiffusion::new(
              gamma_ as f32,
              sigma as f32,
              n,
              x0.map(|v| v as f32),
              t.map(|v| v as f32),
              cpoisson,
              Unseeded,
            ));
          }
        }
      }
      _ => {
        let cpoisson = CompoundPoisson::new(
          crate::traits::CallableDist::new(distribution),
          Poisson::new(lambda_, Some(n), t, Unseeded),
          Unseeded,
        );
        match seed {
          Some(sd) => {
            s.seeded_f64 = Some(LevyDiffusion::new(
              gamma_,
              sigma,
              n,
              x0,
              t,
              cpoisson,
              Deterministic::new(sd),
            ));
          }
          None => {
            s.inner_f64 = Some(LevyDiffusion::new(
              gamma_, sigma, n, x0, t, cpoisson, Unseeded,
            ));
          }
        }
      }
    }
    s
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
    use numpy::ndarray::Array2;
    use pyo3::IntoPyObjectExt;

    use crate::traits::ProcessExt;
    py_dispatch!(self, |inner| {
      let paths = inner.sample_par(m);
      let n = paths[0].len();
      let mut result = Array2::zeros((m, n));
      for (i, path) in paths.iter().enumerate() {
        result.row_mut(i).assign(path);
      }
      result.into_pyarray(py).into_py_any(py).unwrap()
    })
  }
}
