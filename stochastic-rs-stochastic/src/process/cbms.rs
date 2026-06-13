//! # Cbms
//!
//! $$
//! dX_t=L\,dW_t,\quad LL^\top=\Sigma
//! $$
//!
use ndarray::Array1;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;

use crate::noise::cgns::Cgns;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
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

impl<T: FloatExt, S: SeedExt> Cbms<T, S> {
  pub fn new(rho: T, n: usize, t: Option<T>, seed: S) -> Self {
    assert!(
      (-T::one()..=T::one()).contains(&rho),
      "Correlation coefficient must be in [-1, 1]"
    );

    Self {
      rho,
      n,
      t,
      seed,
      cgns: Cgns::new(rho, n - 1, t, Unseeded),
    }
  }
}

impl<T: FloatExt, S: SeedExt> ProcessExt<T> for Cbms<T, S> {
  type Output = [Array1<T>; 2];
  type Sampler<'s>
    = CbmsSampler<T, S>
  where
    Self: 's;

  fn sampler(&self) -> CbmsSampler<T, S> {
    CbmsSampler {
      n: self.n,
      cgns: self.cgns,
      seed: self.seed.clone(),
    }
  }
}

/// Reusable [`Cbms`] sampling state: owns the correlated-Gaussian generator and
/// the seed source so a Monte-Carlo loop reuses both output buffers.
#[doc(hidden)]
pub struct CbmsSampler<T: FloatExt, S: SeedExt> {
  n: usize,
  cgns: Cgns<T>,
  seed: S,
}

impl<T: FloatExt, S: SeedExt> CbmsSampler<T, S> {
  fn fill_paths(&mut self, bm1: &mut [T], bm2: &mut [T]) {
    if self.n == 0 {
      return;
    }
    let [cgn1, cgn2] = &self.cgns.sample_impl(&self.seed.derive());
    bm1[0] = T::zero();
    bm2[0] = T::zero();
    for i in 1..self.n {
      bm1[i] = bm1[i - 1] + cgn1[i - 1];
      bm2[i] = bm2[i - 1] + cgn2[i - 1];
    }
  }
}

impl<T: FloatExt, S: SeedExt> PathSampler<T> for CbmsSampler<T, S> {
  type Output = [Array1<T>; 2];

  fn sample_into(&mut self, out: &mut [Array1<T>; 2]) {
    let [bm1, bm2] = out;
    self.fill_paths(
      bm1.as_slice_mut().expect("Cbms output must be contiguous"),
      bm2.as_slice_mut().expect("Cbms output must be contiguous"),
    );
  }

  fn sample(&mut self) -> [Array1<T>; 2] {
    let mut bm1 = Array1::<T>::zeros(self.n);
    let mut bm2 = Array1::<T>::zeros(self.n);
    self.fill_paths(
      bm1.as_slice_mut().expect("contiguous"),
      bm2.as_slice_mut().expect("contiguous"),
    );
    [bm1, bm2]
  }
}

py_process_2x1d!(PyCbms, Cbms,
  sig: (rho, n, t=None, seed=None, dtype=None),
  params: (rho: f64, n: usize, t: Option<f64>)
);
