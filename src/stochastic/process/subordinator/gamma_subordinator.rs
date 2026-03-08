use ndarray::Array1;
use ndarray_rand::RandomExt;

use crate::distributions::gamma::SimdGamma;
use crate::simd_rng::Deterministic;
use crate::simd_rng::SeedExt;
use crate::simd_rng::Unseeded;
use crate::traits::FloatExt;
use crate::traits::ProcessExt;

/// Gamma subordinator where `G_t ~ Gamma(nu * t, rate)`.
pub struct GammaSubordinator<T: FloatExt, S: SeedExt = Unseeded> {
  /// Shape intensity `nu`.
  pub nu: T,
  /// Rate parameter (>0).
  pub rate: T,
  /// Number of grid points.
  pub n: usize,
  /// Initial level.
  pub x0: Option<T>,
  /// Horizon.
  pub t: Option<T>,
  /// Seed strategy (compile-time: [`Unseeded`] or [`Deterministic`]).
  pub seed: S,
}

impl<T: FloatExt> GammaSubordinator<T> {
  pub fn new(nu: T, rate: T, n: usize, x0: Option<T>, t: Option<T>) -> Self {
    assert!(nu > T::zero(), "nu must be positive");
    assert!(rate > T::zero(), "rate must be positive");
    Self {
      nu,
      rate,
      n,
      x0,
      t,
      seed: Unseeded,
    }
  }
}

impl<T: FloatExt> GammaSubordinator<T, Deterministic> {
  pub fn seeded(nu: T, rate: T, n: usize, x0: Option<T>, t: Option<T>, seed: u64) -> Self {
    assert!(nu > T::zero(), "nu must be positive");
    assert!(rate > T::zero(), "rate must be positive");
    Self {
      nu,
      rate,
      n,
      x0,
      t,
      seed: Deterministic(seed),
    }
  }
}

impl<T: FloatExt, S: SeedExt> ProcessExt<T> for GammaSubordinator<T, S> {
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
    let shape = self.nu * dt;
    let scale = T::one() / self.rate;
    let mut seed = self.seed;
    let gamma = SimdGamma::from_seed_source(shape, scale, &mut seed);
    let mut rng = seed.rng();
    let inc = Array1::random_using(self.n - 1, &gamma, &mut rng);
    for i in 1..self.n {
      out[i] = out[i - 1] + inc[i - 1];
    }
    out
  }
}

py_process_1d!(PyGammaSubordinator, GammaSubordinator,
  sig: (nu, rate, n, x0=None, t=None, dtype=None),
  params: (nu: f64, rate: f64, n: usize, x0: Option<f64>, t: Option<f64>)
);
