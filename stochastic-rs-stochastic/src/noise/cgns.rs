//! # Cgns
//!
//! $$
//! Z_t=L\varepsilon_t,\quad \varepsilon_t\sim\mathcal N(0,I),\ LL^\top=\Sigma
//! $$
//!
use ndarray::Array1;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;

use crate::traits::FloatExt;
use crate::traits::ProcessExt;

#[derive(Copy, Clone)]
pub struct Cgns<T: FloatExt, S: SeedExt = Unseeded> {
  /// Instantaneous correlation parameter.
  pub rho: T,
  /// Number of discrete simulation points (or samples).
  pub n: usize,
  /// Total simulation horizon (defaults to 1 when omitted).
  pub t: Option<T>,
  /// Seed strategy (compile-time: [`Unseeded`] or [`Deterministic`]).
  pub seed: S,
}

impl<T: FloatExt> Cgns<T> {
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
    }
  }
}

impl<T: FloatExt> Cgns<T, Deterministic> {
  pub fn seeded(rho: T, n: usize, t: Option<T>, seed: u64) -> Self {
    assert!(
      (-T::one()..=T::one()).contains(&rho),
      "Correlation coefficient must be in [-1, 1]"
    );

    Self {
      rho,
      n,
      t,
      seed: Deterministic(seed),
    }
  }
}

impl<T: FloatExt, S: SeedExt> Cgns<T, S> {
  /// Sample with an explicit seed, used by callers like Cbms.
  pub fn sample_with_seed(&self, seed: u64) -> [Array1<T>; 2] {
    self.sample_impl(Deterministic(seed))
  }

  /// Core sampling — monomorphised per seed strategy, zero runtime branching.
  #[inline]
  pub(crate) fn sample_impl<S2: SeedExt>(&self, mut seed: S2) -> [Array1<T>; 2] {
    let mut gn1 = Array1::<T>::zeros(self.n);
    let mut z = Array1::<T>::zeros(self.n);
    if self.n == 0 {
      return [gn1, z];
    }

    let sqrt_dt = (self.t.unwrap_or(T::one()) / T::from_usize_(self.n)).sqrt();
    let gn1_slice = gn1.as_slice_mut().expect("Cgns noise 1 must be contiguous");
    let z_slice = z.as_slice_mut().expect("Cgns noise 2 must be contiguous");
    let n1 = stochastic_rs_distributions::normal::SimdNormal::<T>::from_seed_source(
      T::zero(),
      sqrt_dt,
      &mut seed,
    );
    let n2 = stochastic_rs_distributions::normal::SimdNormal::<T>::from_seed_source(
      T::zero(),
      sqrt_dt,
      &mut seed,
    );
    n1.fill_slice_fast(gn1_slice);
    n2.fill_slice_fast(z_slice);
    let c = (T::one() - self.rho.powi(2)).sqrt();
    let mut gn2 = Array1::zeros(self.n);

    for i in 0..self.n {
      gn2[i] = self.rho * gn1[i] + c * z[i];
    }

    [gn1, gn2]
  }

  pub fn dt(&self) -> T {
    self.t.unwrap_or(T::one()) / T::from_usize_(self.n)
  }
}

impl<T: FloatExt, S: SeedExt> ProcessExt<T> for Cgns<T, S> {
  type Output = [Array1<T>; 2];

  fn sample(&self) -> Self::Output {
    self.sample_impl(self.seed)
  }
}

py_process_2x1d!(PyCgns, Cgns,
  sig: (rho, n, t=None, seed=None, dtype=None),
  params: (rho: f64, n: usize, t: Option<f64>)
);
