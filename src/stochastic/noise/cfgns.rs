//! # Cfgns
//!
//! $$
//! Z_t=L\eta_t^H,\quad \operatorname{Cov}(\eta_i^H,\eta_j^H)=\gamma_H(i-j)
//! $$
//!
use ndarray::Array1;

use super::fgn::FGN;
use crate::simd_rng::Deterministic;
use crate::simd_rng::SeedExt;
use crate::simd_rng::Unseeded;
use crate::traits::FloatExt;
use crate::traits::ProcessExt;

pub struct CFGNS<T: FloatExt, S: SeedExt = Unseeded> {
  /// Hurst exponent controlling roughness and long-memory.
  pub hurst: T,
  /// Instantaneous correlation parameter.
  pub rho: T,
  /// Number of discrete simulation points (or samples).
  pub n: usize,
  /// Total simulation horizon (defaults to 1 when omitted).
  pub t: Option<T>,
  /// Seed strategy (compile-time: [`Unseeded`] or [`Deterministic`]).
  pub seed: S,
  fgn: FGN<T>,
}

impl<T: FloatExt> CFGNS<T> {
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
      fgn: FGN::new(hurst, n, t),
    }
  }
}

impl<T: FloatExt> CFGNS<T, Deterministic> {
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
      fgn: FGN::new(hurst, n, t),
    }
  }
}

impl<T: FloatExt, S: SeedExt> CFGNS<T, S> {
  /// Sample with an explicit seed, used by callers like CFBMS.
  pub fn sample_with_seed(&self, seed: u64) -> [Array1<T>; 2] {
    self.sample_impl(Deterministic(seed))
  }

  /// Core sampling — monomorphised per seed strategy, zero runtime branching.
  #[inline]
  pub(crate) fn sample_impl<S2: SeedExt>(&self, mut seed: S2) -> [Array1<T>; 2] {
    let child1 = seed.derive();
    let child2 = seed.derive();
    let fgn1 = self.fgn.sample_cpu_impl(child1);
    let z = self.fgn.sample_cpu_impl(child2);
    let c = (T::one() - self.rho.powi(2)).sqrt();
    let mut fgn2 = Array1::zeros(self.n);
    for i in 0..self.n {
      fgn2[i] = self.rho * fgn1[i] + c * z[i];
    }
    [fgn1, fgn2]
  }
}

impl<T: FloatExt, S: SeedExt> ProcessExt<T> for CFGNS<T, S> {
  type Output = [Array1<T>; 2];

  fn sample(&self) -> Self::Output {
    self.sample_impl(self.seed)
  }
}

py_process_2x1d!(PyCFGNS, CFGNS,
  sig: (hurst, rho, n, t=None, seed=None, dtype=None),
  params: (hurst: f64, rho: f64, n: usize, t: Option<f64>)
);
