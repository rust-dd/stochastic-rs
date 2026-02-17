use ndarray::Array1;
use ndarray::Axis;
use rand::rng;
use rand::Rng;
use rand_distr::Distribution;

use super::poisson::Poisson;
use crate::distributions::poisson::SimdPoisson;
use crate::traits::FloatExt;
use crate::traits::ProcessExt;

pub struct CompoundPoisson<T, D>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
{
  pub distribution: D,
  pub poisson: Poisson<T>,
}

impl<T, D> CompoundPoisson<T, D>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
{
  pub fn new(distribution: D, poisson: Poisson<T>) -> Self {
    Self {
      distribution,
      poisson,
    }
  }

  /// Draw compound-Poisson jump increments on a fixed simulation grid.
  ///
  /// The returned array has length `n`, with `increments[0] = 0` and
  /// `increments[i]` holding the sum of jump sizes over `((i-1)dt, i*dt]`.
  pub fn sample_grid_increments(&self, n: usize, dt: T) -> Array1<T> {
    let mut increments = Array1::<T>::zeros(n);
    if n <= 1 {
      return increments;
    }

    let lambda_dt = (self.poisson.lambda * dt).to_f64().unwrap();
    if !(lambda_dt.is_finite()) {
      panic!("lambda * dt must be finite");
    }
    if lambda_dt <= 0.0 {
      return increments;
    }

    let poisson = SimdPoisson::<u32>::new(lambda_dt);
    let mut rng = rng();
    for i in 1..n {
      let jump_count = poisson.sample(&mut rng);
      let mut jump_sum = T::zero();
      for _ in 0..jump_count {
        jump_sum += self.distribution.sample(&mut rng);
      }
      increments[i] = jump_sum;
    }

    increments
  }

  #[inline]
  fn relative_jump_from_count<R: Rng + ?Sized>(&self, jump_count: u32, rng: &mut R) -> T {
    let mut factor = T::one();
    for _ in 0..jump_count {
      let y = self.distribution.sample(rng);
      factor = factor * (T::one() + y);
    }
    factor - T::one()
  }

  /// Draw multiplicative jump increments on a fixed simulation grid.
  ///
  /// If the per-jump return is `Y`, then for each interval this returns
  /// `prod_k(1 + Y_k) - 1`, preserving multiple jumps in one `dt` step.
  pub fn sample_grid_relative_increments(&self, n: usize, dt: T) -> Array1<T> {
    let mut increments = Array1::<T>::zeros(n);
    if n <= 1 {
      return increments;
    }

    let lambda_dt = (self.poisson.lambda * dt).to_f64().unwrap();
    if !(lambda_dt.is_finite()) {
      panic!("lambda * dt must be finite");
    }
    if lambda_dt <= 0.0 {
      return increments;
    }

    let poisson = SimdPoisson::<u32>::new(lambda_dt);
    let mut rng = rng();
    for i in 1..n {
      let jump_count = poisson.sample(&mut rng);
      increments[i] = self.relative_jump_from_count(jump_count, &mut rng);
    }

    increments
  }
}

impl<T, D> ProcessExt<T> for CompoundPoisson<T, D>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
{
  type Output = [Array1<T>; 3];

  fn sample(&self) -> Self::Output {
    let poisson = self.poisson.sample();
    let mut jumps = Array1::<T>::zeros(poisson.len());
    for i in 1..poisson.len() {
      jumps[i] = self.distribution.sample(&mut rng());
    }

    let mut cum_jupms = jumps.clone();
    cum_jupms.accumulate_axis_inplace(Axis(0), |&prev, curr| *curr += prev);

    [poisson, cum_jupms, jumps]
  }
}

#[cfg(test)]
mod tests {
  use rand_distr::Distribution;

  use super::*;

  #[derive(Clone, Copy)]
  struct ConstJump<T: FloatExt>(T);

  impl<T: FloatExt> Distribution<T> for ConstJump<T> {
    fn sample<R: rand::Rng + ?Sized>(&self, _rng: &mut R) -> T {
      self.0
    }
  }

  #[test]
  fn grid_increments_zero_for_zero_intensity() {
    let cp = CompoundPoisson::new(ConstJump(1.0f64), Poisson::new(0.0, Some(16), Some(1.0)));
    let inc = cp.sample_grid_increments(16, 1.0 / 15.0);
    assert_eq!(inc.len(), 16);
    assert!(inc.iter().all(|&x| x == 0.0));
  }

  #[test]
  fn grid_increments_start_at_zero() {
    let cp = CompoundPoisson::new(ConstJump(1.0f64), Poisson::new(2.0, Some(16), Some(1.0)));
    let inc = cp.sample_grid_increments(16, 1.0 / 15.0);
    assert_eq!(inc[0], 0.0);
  }

  #[test]
  fn relative_increment_compounds_multiple_jumps() {
    let cp = CompoundPoisson::new(ConstJump(0.1f64), Poisson::new(1.0, Some(4), Some(1.0)));
    let mut rng = rand::rng();
    let rel = cp.relative_jump_from_count(3, &mut rng);
    let expected = 1.1f64.powi(3) - 1.0;
    assert!((rel - expected).abs() < 1e-12);
  }
}

#[cfg(feature = "python")]
#[pyo3::prelude::pyclass]
pub struct PyCompoundPoisson {
  inner_f32: Option<CompoundPoisson<f32, crate::traits::CallableDist<f32>>>,
  inner_f64: Option<CompoundPoisson<f64, crate::traits::CallableDist<f64>>>,
}

#[cfg(feature = "python")]
#[pyo3::prelude::pymethods]
impl PyCompoundPoisson {
  #[new]
  #[pyo3(signature = (distribution, lambda_, n=None, t_max=None, dtype=None))]
  fn new(
    distribution: pyo3::Py<pyo3::PyAny>,
    lambda_: f64,
    n: Option<usize>,
    t_max: Option<f64>,
    dtype: Option<&str>,
  ) -> Self {
    match dtype.unwrap_or("f64") {
      "f32" => Self {
        inner_f32: Some(CompoundPoisson::new(
          crate::traits::CallableDist::new(distribution),
          Poisson::new(lambda_ as f32, n, t_max.map(|v| v as f32)),
        )),
        inner_f64: None,
      },
      _ => Self {
        inner_f32: None,
        inner_f64: Some(CompoundPoisson::new(
          crate::traits::CallableDist::new(distribution),
          Poisson::new(lambda_, n, t_max),
        )),
      },
    }
  }

  fn sample<'py>(
    &self,
    py: pyo3::Python<'py>,
  ) -> (
    pyo3::Py<pyo3::PyAny>,
    pyo3::Py<pyo3::PyAny>,
    pyo3::Py<pyo3::PyAny>,
  ) {
    use numpy::IntoPyArray;
    use pyo3::IntoPyObjectExt;

    use crate::traits::ProcessExt;
    if let Some(ref inner) = self.inner_f64 {
      let [p, cum, j] = inner.sample();
      (
        p.into_pyarray(py).into_py_any(py).unwrap(),
        cum.into_pyarray(py).into_py_any(py).unwrap(),
        j.into_pyarray(py).into_py_any(py).unwrap(),
      )
    } else if let Some(ref inner) = self.inner_f32 {
      let [p, cum, j] = inner.sample();
      (
        p.into_pyarray(py).into_py_any(py).unwrap(),
        cum.into_pyarray(py).into_py_any(py).unwrap(),
        j.into_pyarray(py).into_py_any(py).unwrap(),
      )
    } else {
      unreachable!()
    }
  }
}
