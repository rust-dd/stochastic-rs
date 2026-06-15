//! # Cfbms
//!
//! $$
//! dX_t=L\,dB_t^H,\quad LL^\top=\Sigma
//! $$
//!
use ndarray::Array1;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;

use crate::noise::cfgns::Cfgns;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

pub struct Cfbms<T: FloatExt, S: SeedExt = Unseeded> {
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
  cfgns: Cfgns<T>,
}

impl<T: FloatExt, S: SeedExt> Cfbms<T, S> {
  pub fn new(hurst: T, rho: T, n: usize, t: Option<T>, seed: S) -> Self {
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
      seed,
      cfgns: Cfgns::new(hurst, rho, n - 1, t, Unseeded),
    }
  }
}

impl<T: FloatExt, S: SeedExt> ProcessExt<T> for Cfbms<T, S> {
  type Output = [Array1<T>; 2];
  type Sampler<'s>
    = CfbmsSampler<'s, T, S>
  where
    Self: 's;

  fn sampler(&self) -> CfbmsSampler<'_, T, S> {
    CfbmsSampler {
      n: self.n,
      cfgns: &self.cfgns,
      seed: self.seed.clone(),
    }
  }
}

/// Reusable [`Cfbms`] sampling state: borrows the fractional correlated-Gaussian
/// generator (which owns non-`Copy` FFT scratch) and owns the seed source so a
/// Monte-Carlo loop reuses both output buffers.
#[doc(hidden)]
pub struct CfbmsSampler<'a, T: FloatExt, S: SeedExt> {
  n: usize,
  cfgns: &'a Cfgns<T>,
  seed: S,
}

impl<T: FloatExt, S: SeedExt> CfbmsSampler<'_, T, S> {
  fn fill_paths(&mut self, fbm1: &mut [T], fbm2: &mut [T]) {
    if self.n == 0 {
      return;
    }
    let [fgn1, fgn2] = self.cfgns.sample_impl(&self.seed.derive());
    fbm1[0] = T::zero();
    fbm2[0] = T::zero();
    for i in 1..self.n {
      fbm1[i] = fbm1[i - 1] + fgn1[i - 1];
      fbm2[i] = fbm2[i - 1] + fgn2[i - 1];
    }
  }
}

impl<T: FloatExt, S: SeedExt> PathSampler<T> for CfbmsSampler<'_, T, S> {
  type Output = [Array1<T>; 2];

  fn sample_into(&mut self, out: &mut [Array1<T>; 2]) {
    let [fbm1, fbm2] = out;
    self.fill_paths(
      fbm1
        .as_slice_mut()
        .expect("Cfbms output must be contiguous"),
      fbm2
        .as_slice_mut()
        .expect("Cfbms output must be contiguous"),
    );
  }

  fn sample(&mut self) -> [Array1<T>; 2] {
    let mut fbm1 = Array1::<T>::zeros(self.n);
    let mut fbm2 = Array1::<T>::zeros(self.n);
    self.fill_paths(
      fbm1.as_slice_mut().expect("contiguous"),
      fbm2.as_slice_mut().expect("contiguous"),
    );
    [fbm1, fbm2]
  }
}

py_process_2x1d!(PyCfbms, Cfbms,
  sig: (hurst, rho, n, t=None, seed=None, dtype=None),
  params: (hurst: f64, rho: f64, n: usize, t: Option<f64>)
);
