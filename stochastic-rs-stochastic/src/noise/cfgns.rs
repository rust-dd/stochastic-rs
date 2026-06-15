//! # Cfgns
//!
//! $$
//! Z_t=L\eta_t^H,\quad \operatorname{Cov}(\eta_i^H,\eta_j^H)=\gamma_H(i-j)
//! $$
//!
use ndarray::Array1;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;

use super::fgn::Fgn;
use crate::device::Backend;
use crate::device::Cpu;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

pub struct Cfgns<T: FloatExt, S: SeedExt = Unseeded, B = Cpu> {
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
  fgn: Fgn<T, Unseeded, B>,
}

impl<T: FloatExt, S: SeedExt> Cfgns<T, S, Cpu> {
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
      fgn: Fgn::new(hurst, n, t, Unseeded),
    }
  }
}

impl<T: FloatExt, S: SeedExt, B: Backend> Cfgns<T, S, B> {
  /// Sample with an explicit seed, used by callers like Cfbms.
  pub fn sample_with_seed(&self, seed: u64) -> [Array1<T>; 2] {
    self.sample_impl(&Deterministic::new(seed))
  }

  /// Core sampling — monomorphised per seed strategy, zero runtime branching.
  /// Uses one paired fGN pass (real/imag of a single circulant FFT) for the two
  /// independent fields; on a GPU backend they come from a batch of two.
  #[inline]
  pub(crate) fn sample_impl<S2: SeedExt>(&self, seed: &S2) -> [Array1<T>; 2] {
    let (fgn1, z) = self.fgn.noise_pair(seed);
    let c = (T::one() - self.rho.powi(2)).sqrt();
    let mut fgn2 = Array1::zeros(self.n);
    for i in 0..self.n {
      fgn2[i] = self.rho * fgn1[i] + c * z[i];
    }
    [fgn1, fgn2]
  }
}

impl<T: FloatExt, S: SeedExt, B: Backend> ProcessExt<T> for Cfgns<T, S, B> {
  type Output = [Array1<T>; 2];
  type Sampler<'s>
    = CfgnsSampler<'s, T, S, B>
  where
    Self: 's;

  /// A CPU sampler borrowing the process for its inner [`Fgn`] (`Arc`-shared
  /// FFT plan + eigenvalues) and seed source. The first `sample` reproduces the
  /// legacy `sample_impl(&seed)` stream bit-for-bit; each subsequent call
  /// advances the seed for an independent correlated pair.
  fn sampler(&self) -> CfgnsSampler<'_, T, S, B> {
    CfgnsSampler { cfgns: self }
  }
}

/// Reusable [`Cfgns`] sampling state: borrows the process for its inner [`Fgn`]
/// (one paired fGN pass per call) and seed source.
#[doc(hidden)]
pub struct CfgnsSampler<'a, T: FloatExt, S: SeedExt, B> {
  cfgns: &'a Cfgns<T, S, B>,
}

impl<T: FloatExt, S: SeedExt, B: Backend> CfgnsSampler<'_, T, S, B> {
  fn fill_paths(&mut self, fgn1_out: &mut [T], fgn2_out: &mut [T]) {
    let [fgn1, fgn2] = self.cfgns.sample_impl(&self.cfgns.seed);
    fgn1_out.copy_from_slice(fgn1.as_slice().expect("Cfgns noise 1 must be contiguous"));
    fgn2_out.copy_from_slice(fgn2.as_slice().expect("Cfgns noise 2 must be contiguous"));
  }
}

impl<T: FloatExt, S: SeedExt, B: Backend> PathSampler<T> for CfgnsSampler<'_, T, S, B> {
  type Output = [Array1<T>; 2];

  fn sample_into(&mut self, out: &mut [Array1<T>; 2]) {
    let [a, b] = out;
    let fgn1 = a.as_slice_mut().expect("Cfgns output must be contiguous");
    let fgn2 = b.as_slice_mut().expect("Cfgns output must be contiguous");
    self.fill_paths(fgn1, fgn2);
  }

  fn sample(&mut self) -> [Array1<T>; 2] {
    self.cfgns.sample_impl(&self.cfgns.seed)
  }
}

backend_switch!([T: FloatExt, S: SeedExt] Cfgns<T, S> { hurst, rho, n, t, seed } via fgn);

py_process_2x1d!(PyCfgns, Cfgns,
  sig: (hurst, rho, n, t=None, seed=None, dtype=None),
  params: (hurst: f64, rho: f64, n: usize, t: Option<f64>)
);
