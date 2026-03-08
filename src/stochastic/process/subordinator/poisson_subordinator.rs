use ndarray::Array1;
use rand_distr::Distribution;

use crate::distributions::poisson::SimdPoisson;
use crate::simd_rng::Deterministic;
use crate::simd_rng::Seed;
use crate::simd_rng::Unseeded;
use crate::traits::FloatExt;
use crate::traits::ProcessExt;

/// Poisson subordinator with unit jumps:
/// `N_t` with independent increments `Poisson(lambda * dt)`.
pub struct PoissonSubordinator<T: FloatExt, S: Seed = Unseeded> {
  /// Intensity parameter.
  pub lambda: T,
  /// Number of grid points.
  pub n: usize,
  /// Initial level.
  pub x0: Option<T>,
  /// Horizon.
  pub t: Option<T>,
  /// Seed strategy (compile-time: [`Unseeded`] or [`Deterministic`]).
  pub seed: S,
}

impl<T: FloatExt> PoissonSubordinator<T> {
  pub fn new(lambda: T, n: usize, x0: Option<T>, t: Option<T>) -> Self {
    assert!(lambda > T::zero(), "lambda must be positive");
    Self {
      lambda,
      n,
      x0,
      t,
      seed: Unseeded,
    }
  }
}

impl<T: FloatExt> PoissonSubordinator<T, Deterministic> {
  pub fn seeded(lambda: T, n: usize, x0: Option<T>, t: Option<T>, seed: u64) -> Self {
    assert!(lambda > T::zero(), "lambda must be positive");
    Self {
      lambda,
      n,
      x0,
      t,
      seed: Deterministic(seed),
    }
  }
}

impl<T: FloatExt, S: Seed> ProcessExt<T> for PoissonSubordinator<T, S> {
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
    let t_max = self.t.unwrap_or(T::one());
    let dt = t_max / T::from_usize_(self.n - 1);
    let lambda_dt = (self.lambda * dt).to_f64().unwrap();
    let poisson = SimdPoisson::<u32>::new(lambda_dt);
    let mut seed = self.seed;
    let mut rng = seed.rng();
    for i in 1..self.n {
      let k = poisson.sample(&mut rng) as usize;
      out[i] = out[i - 1] + T::from_usize_(k);
    }
    out
  }
}

py_process_1d!(PyPoissonSubordinator, PoissonSubordinator,
  sig: (lambda_, n, x0=None, t=None, dtype=None),
  params: (lambda_: f64, n: usize, x0: Option<f64>, t: Option<f64>)
);
