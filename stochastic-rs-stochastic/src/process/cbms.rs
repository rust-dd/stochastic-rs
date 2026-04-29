//! # Cbms
//!
//! $$
//! dX_t=L\,dW_t,\quad LL^\top=\Sigma
//! $$
//!
use ndarray::Array1;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;

use crate::noise::cgns::Cgns;
use crate::traits::FloatExt;
use crate::traits::ProcessExt;

pub struct Cbms<T: FloatExt, S: SeedExt = Unseeded> {
  /// Instantaneous correlation between the two Brownian components.
  pub rho: T,
  /// Number of discrete time points in each path.
  pub n: usize,
  /// Total simulation horizon (defaults to `1` if `None`).
  pub t: Option<T>,
  /// Seed strategy (compile-time: [`Unseeded`] or [`Deterministic`]).
  pub seed: S,
  cgns: Cgns<T>,
}

impl<T: FloatExt> Cbms<T> {
  pub fn new(rho: T, n: usize, t: Option<T>) -> Self {
    assert!(
      (-T::one()..=T::one()).contains(&rho),
      "Correlation coefficient must be in [-1, 1]"
    );

    Self {
      rho,
      n,
      t,
      seed: Unseeded,
      cgns: Cgns::new(rho, n - 1, t),
    }
  }
}

impl<T: FloatExt> Cbms<T, Deterministic> {
  pub fn seeded(rho: T, n: usize, t: Option<T>, seed: u64) -> Self {
    assert!(
      (-T::one()..=T::one()).contains(&rho),
      "Correlation coefficient must be in [-1, 1]"
    );

    Self {
      rho,
      n,
      t,
      seed: Deterministic::new(seed),
      cgns: Cgns::new(rho, n - 1, t),
    }
  }
}

impl<T: FloatExt, S: SeedExt> Cbms<T, S> {
  #[inline]
  fn cumsum_noise(&self, noise: [Array1<T>; 2]) -> [Array1<T>; 2] {
    let [cgn1, cgn2] = &noise;
    let mut bm1 = Array1::<T>::zeros(self.n);
    let mut bm2 = Array1::<T>::zeros(self.n);
    for i in 1..self.n {
      bm1[i] = bm1[i - 1] + cgn1[i - 1];
      bm2[i] = bm2[i - 1] + cgn2[i - 1];
    }
    [bm1, bm2]
  }
}

impl<T: FloatExt, S: SeedExt> ProcessExt<T> for Cbms<T, S> {
  type Output = [Array1<T>; 2];

  fn sample(&self) -> Self::Output {
    let noise = self.cgns.sample_impl(&self.seed.derive());
    self.cumsum_noise(noise)
  }
}

py_process_2x1d!(PyCbms, Cbms,
  sig: (rho, n, t=None, seed=None, dtype=None),
  params: (rho: f64, n: usize, t: Option<f64>)
);
