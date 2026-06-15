use ndarray::Array1;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::gamma::SimdGamma;

use crate::buffer::array1_from_fill;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
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

impl<T: FloatExt, S: SeedExt> GammaSubordinator<T, S> {
  pub fn new(nu: T, rate: T, n: usize, x0: Option<T>, t: Option<T>, seed: S) -> Self {
    assert!(nu > T::zero(), "nu must be positive");
    assert!(rate > T::zero(), "rate must be positive");
    Self {
      nu,
      rate,
      n,
      x0,
      t,
      seed,
    }
  }
}

impl<T: FloatExt, S: SeedExt> ProcessExt<T> for GammaSubordinator<T, S> {
  type Output = Array1<T>;
  type Sampler<'s>
    = GammaSubordinatorSampler<T>
  where
    Self: 's;

  fn sampler(&self) -> GammaSubordinatorSampler<T> {
    let x0 = self.x0.unwrap_or(T::zero());
    let n_increments = self.n.saturating_sub(1).max(1);
    let dt = self.t.unwrap_or(T::one()) / T::from_usize_(n_increments);
    let shape = self.nu * dt;
    let scale = T::one() / self.rate;
    GammaSubordinatorSampler {
      n: self.n,
      x0,
      gamma: SimdGamma::<T>::new(shape, scale, &self.seed),
    }
  }
}

/// Reusable [`GammaSubordinator`] sampling state: the owned Gamma increment
/// source. The path is `x0` followed by the running sum of the increments.
#[doc(hidden)]
pub struct GammaSubordinatorSampler<T: FloatExt> {
  n: usize,
  x0: T,
  gamma: SimdGamma<T>,
}

impl<T: FloatExt> GammaSubordinatorSampler<T> {
  fn fill_path(&mut self, out: &mut [T]) {
    if out.is_empty() {
      return;
    }
    out[0] = self.x0;
    if out.len() == 1 {
      return;
    }
    let tail = &mut out[1..];
    self.gamma.fill_slice_fast(tail);
    let mut acc = self.x0;
    for x in tail.iter_mut() {
      acc += *x;
      *x = acc;
    }
  }
}

impl<T: FloatExt> PathSampler<T> for GammaSubordinatorSampler<T> {
  type Output = Array1<T>;

  fn sample_into(&mut self, out: &mut Array1<T>) {
    let slice = out
      .as_slice_mut()
      .expect("GammaSubordinator output must be contiguous");
    self.fill_path(slice);
  }

  fn sample(&mut self) -> Array1<T> {
    let n = self.n;
    array1_from_fill(n, |out| self.fill_path(out))
  }
}

py_process_1d!(PyGammaSubordinator, GammaSubordinator,
  sig: (nu, rate, n, x0=None, t=None, seed=None, dtype=None),
  params: (nu: f64, rate: f64, n: usize, x0: Option<f64>, t: Option<f64>)
);
