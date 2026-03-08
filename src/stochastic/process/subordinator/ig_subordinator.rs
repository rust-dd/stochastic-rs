use ndarray::Array1;

use crate::distributions::inverse_gauss::SimdInverseGauss;
use crate::simd_rng::Deterministic;
use crate::simd_rng::SeedExt;
use crate::simd_rng::Unseeded;
use crate::traits::FloatExt;
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

impl<T: FloatExt> IGSubordinator<T> {
  pub fn new(delta: T, gamma: T, n: usize, x0: Option<T>, t: Option<T>) -> Self {
    assert!(delta > T::zero(), "delta must be positive");
    assert!(gamma > T::zero(), "gamma must be positive");
    Self {
      delta,
      gamma,
      n,
      x0,
      t,
      seed: Unseeded,
    }
  }
}

impl<T: FloatExt> IGSubordinator<T, Deterministic> {
  pub fn seeded(delta: T, gamma: T, n: usize, x0: Option<T>, t: Option<T>, seed: u64) -> Self {
    assert!(delta > T::zero(), "delta must be positive");
    assert!(gamma > T::zero(), "gamma must be positive");
    Self {
      delta,
      gamma,
      n,
      x0,
      t,
      seed: Deterministic(seed),
    }
  }
}

impl<T: FloatExt, S: SeedExt> ProcessExt<T> for IGSubordinator<T, S> {
  type Output = Array1<T>;

  fn sample(&self) -> Self::Output {
    let mut out = Array1::<T>::zeros(self.n);
    if self.n == 0 {
      return out;
    }
    out[0] = self.x0.unwrap_or(T::zero());
    if self.n == 1 {
      return out;
    }

    let dt = self.t.unwrap_or(T::one()) / T::from_usize_(self.n - 1);
    let mu = (self.delta * dt) / self.gamma;
    let lambda = (self.delta * dt).powi(2);
    let mut seed = self.seed;
    let ig = SimdInverseGauss::from_seed_source(mu, lambda, &mut seed);
    let mut inc = Array1::<T>::zeros(self.n - 1);
    ig.fill_slice_fast(inc.as_slice_mut().unwrap());
    for i in 1..self.n {
      out[i] = out[i - 1] + inc[i - 1];
    }
    out
  }
}

py_process_1d!(PyIGSubordinator, IGSubordinator,
  sig: (delta, gamma_, n, x0=None, t=None, seed=None, dtype=None),
  params: (delta: f64, gamma_: f64, n: usize, x0: Option<f64>, t: Option<f64>)
);
