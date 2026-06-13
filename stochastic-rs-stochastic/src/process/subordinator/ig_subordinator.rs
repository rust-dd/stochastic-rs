use ndarray::Array1;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::inverse_gauss::SimdInverseGauss;

use crate::buffer::array1_from_fill;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

/// Inverse-Gaussian subordinator with BNS parameterization:
/// `phi(lambda) = delta (sqrt(gamma^2 + 2 lambda) - gamma)`.
pub struct IGSubordinator<T: FloatExt, S: SeedExt = Unseeded> {
  /// Scale `delta`.
  pub delta: T,
  /// Shape `gamma`.
  pub gamma: T,
  /// Number of grid points.
  pub n: usize,
  /// Initial level.
  pub x0: Option<T>,
  /// Horizon.
  pub t: Option<T>,
  /// Seed strategy (compile-time: [`Unseeded`] or [`Deterministic`]).
  pub seed: S,
}

impl<T: FloatExt, S: SeedExt> IGSubordinator<T, S> {
  pub fn new(delta: T, gamma: T, n: usize, x0: Option<T>, t: Option<T>, seed: S) -> Self {
    assert!(delta > T::zero(), "delta must be positive");
    assert!(gamma > T::zero(), "gamma must be positive");
    Self {
      delta,
      gamma,
      n,
      x0,
      t,
      seed,
    }
  }
}

impl<T: FloatExt, S: SeedExt> ProcessExt<T> for IGSubordinator<T, S> {
  type Output = Array1<T>;
  type Sampler<'s>
    = IGSubordinatorSampler<T>
  where
    Self: 's;

  fn sampler(&self) -> IGSubordinatorSampler<T> {
    let x0 = self.x0.unwrap_or(T::zero());
    let n_increments = self.n.saturating_sub(1).max(1);
    let dt = self.t.unwrap_or(T::one()) / T::from_usize_(n_increments);
    let mu = (self.delta * dt) / self.gamma;
    let lambda = (self.delta * dt).powi(2);
    IGSubordinatorSampler {
      n: self.n,
      x0,
      ig: SimdInverseGauss::<T>::new(mu, lambda, &self.seed),
    }
  }
}

/// Reusable [`IGSubordinator`] sampling state: the owned inverse-Gaussian
/// increment source. The path is `x0` followed by the running sum.
#[doc(hidden)]
pub struct IGSubordinatorSampler<T: FloatExt> {
  n: usize,
  x0: T,
  ig: SimdInverseGauss<T>,
}

impl<T: FloatExt> IGSubordinatorSampler<T> {
  fn fill_path(&mut self, out: &mut [T]) {
    if out.is_empty() {
      return;
    }
    out[0] = self.x0;
    if out.len() == 1 {
      return;
    }
    let tail = &mut out[1..];
    self.ig.fill_slice_fast(tail);
    let mut acc = self.x0;
    for x in tail.iter_mut() {
      acc += *x;
      *x = acc;
    }
  }
}

impl<T: FloatExt> PathSampler<T> for IGSubordinatorSampler<T> {
  type Output = Array1<T>;

  fn sample_into(&mut self, out: &mut Array1<T>) {
    let slice = out
      .as_slice_mut()
      .expect("IGSubordinator output must be contiguous");
    self.fill_path(slice);
  }

  fn sample(&mut self) -> Array1<T> {
    let n = self.n;
    array1_from_fill(n, |out| self.fill_path(out))
  }
}

py_process_1d!(PyIGSubordinator, IGSubordinator,
  sig: (delta, gamma_, n, x0=None, t=None, seed=None, dtype=None),
  params: (delta: f64, gamma_: f64, n: usize, x0: Option<f64>, t: Option<f64>)
);
