//! # Cfbms
//!
//! $$
//! dX_t=L\,dB_t^H,\quad LL^\top=\Sigma
//! $$
//!
use ndarray::Array1;

use crate::simd_rng::Deterministic;
use crate::simd_rng::Seed;
use crate::simd_rng::Unseeded;
use crate::stochastic::noise::cfgns::CFGNS;
use crate::traits::FloatExt;
use crate::traits::ProcessExt;

pub struct CFBMS<T: FloatExt, S: Seed = Unseeded> {
  /// Hurst parameter (`0 < H < 1`) shared by both components.
  pub hurst: T,
  /// Instantaneous correlation between the two fractional-noise drivers.
  pub rho: T,
  /// Number of discrete time points in each path.
  pub n: usize,
  /// Total simulation horizon (defaults to `1` if `None`).
  pub t: Option<T>,
  /// Seed strategy (compile-time: [`Unseeded`] or [`Deterministic`]).
  pub seed: S,
  cfgns: CFGNS<T>,
}

impl<T: FloatExt> CFBMS<T> {
  pub fn new(hurst: T, rho: T, n: usize, t: Option<T>) -> Self {
    assert!(
      (T::zero()..=T::one()).contains(&hurst),
      "Hurst parameter must be in (0, 1)"
    );
    assert!(
      (-T::one()..=T::one()).contains(&rho),
      "Correlation coefficient must be in [-1, 1]"
    );

    Self {
      hurst,
      rho,
      n,
      t,
      seed: Unseeded,
      cfgns: CFGNS::new(hurst, rho, n - 1, t),
    }
  }
}

impl<T: FloatExt> CFBMS<T, Deterministic> {
  pub fn seeded(hurst: T, rho: T, n: usize, t: Option<T>, seed: u64) -> Self {
    assert!(
      (T::zero()..=T::one()).contains(&hurst),
      "Hurst parameter must be in (0, 1)"
    );
    assert!(
      (-T::one()..=T::one()).contains(&rho),
      "Correlation coefficient must be in [-1, 1]"
    );

    Self {
      hurst,
      rho,
      n,
      t,
      seed: Deterministic(seed),
      cfgns: CFGNS::new(hurst, rho, n - 1, t),
    }
  }
}

impl<T: FloatExt, S: Seed> ProcessExt<T> for CFBMS<T, S> {
  type Output = [Array1<T>; 2];

  fn sample(&self) -> Self::Output {
    let mut seed = self.seed;
    let [fgn1, fgn2] = self.cfgns.sample_impl(seed.derive());

    let mut fbm1 = Array1::<T>::zeros(self.n);
    let mut fbm2 = Array1::<T>::zeros(self.n);

    for i in 1..self.n {
      fbm1[i] = fbm1[i - 1] + fgn1[i - 1];
      fbm2[i] = fbm2[i - 1] + fgn2[i - 1];
    }

    [fbm1, fbm2]
  }
}

py_process_2x1d!(PyCFBMS, CFBMS,
  sig: (hurst, rho, n, t=None, dtype=None),
  params: (hurst: f64, rho: f64, n: usize, t: Option<f64>)
);
