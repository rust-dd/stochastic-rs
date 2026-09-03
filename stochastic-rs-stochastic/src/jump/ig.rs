//! # Ig
//!
//! $$
//! L_t\sim\mathrm{Ig}(\mu t,\lambda t),\quad X_t=L_t\text{ or time-change driver}
//! $$
//!
use std::marker::PhantomData;

use ndarray::Array1;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::inverse_gauss::SimdInverseGauss;

use crate::buffer::array1_from_fill;
use crate::device::Cpu;
use crate::device::HostBackend;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

pub struct Ig<T: FloatExt, S: SeedExt = Unseeded, B = Cpu> {
  /// Rate γ of the inverse-Gaussian subordinator: sets the per-step mean
  /// (`gamma * dt`) and, via the mean squared, the per-step shape of the
  /// IG(mean, shape) increments — not an asymmetry/skew parameter.
  pub gamma: T,
  /// Number of points sampled along the IG-subordinator path.
  pub n: usize,
  /// Initial value X₀ of the IG-subordinator path.
  pub x0: Option<T>,
  /// Simulation horizon [0, t] for the path (defaults to 1 when omitted).
  pub t: Option<T>,
  /// Seed strategy (compile-time: `Unseeded` or `Deterministic`).
  pub seed: S,
  /// Sampling backend marker (compile-time): [`Cpu`] by default, a device
  /// marker after [`on`](Self::on). Public so `..Default::default()` struct
  /// updates keep working; it carries no data.
  pub backend: PhantomData<B>,
}

impl<T: FloatExt, S: SeedExt> Ig<T, S> {
  pub fn new(gamma: T, n: usize, x0: Option<T>, t: Option<T>, seed: S) -> Self {
    assert!(gamma > T::zero(), "gamma must be positive");
    Self {
      backend: PhantomData,
      gamma,
      n,
      x0,
      t,
      seed,
    }
  }
}

impl<T: FloatExt, S: SeedExt, B> Ig<T, S, B> {}

impl<T: FloatExt, S: SeedExt, B> Ig<T, S, B> {
  #[inline]
  fn dt(&self) -> T {
    self.t.unwrap_or(T::one()) / T::from_usize_(self.n - 1)
  }
}

backend_switch!([T: FloatExt, S: SeedExt] Ig<T, S> { gamma, n, x0, t, seed } via host);

impl<T: FloatExt, S: SeedExt, B: HostBackend> ProcessExt<T> for Ig<T, S, B> {
  type Output = Array1<T>;
  type Sampler<'s>
    = IgSampler<T>
  where
    Self: 's;

  fn sampler(&self) -> IgSampler<T> {
    // Single-parameter Ig subordinator: increments are strictly positive and
    // independent over grid steps. The IG source is derived from `self.seed`
    // exactly as the legacy `sample()`, so the first fill matches bit-for-bit
    // and the owned source advances on reuse for independent paths.
    let dt = self.dt();
    let mean = self.gamma * dt;
    let shape = mean * mean;
    IgSampler {
      n: self.n,
      x0: self.x0.unwrap_or(T::zero()),
      ig_dist: SimdInverseGauss::<T>::new(mean, shape, &self.seed),
    }
  }
}

/// Reusable [`Ig`] sampling state: owns the inverse-Gaussian subordinator so a
/// Monte-Carlo loop pays its setup once.
#[doc(hidden)]
pub struct IgSampler<T: FloatExt> {
  n: usize,
  x0: T,
  ig_dist: SimdInverseGauss<T>,
}

impl<T: FloatExt> IgSampler<T> {
  fn fill_path(&mut self, out: &mut [T]) {
    if out.is_empty() {
      return;
    }
    out[0] = self.x0;
    if out.len() == 1 {
      return;
    }

    let mut inc = Array1::<T>::zeros(out.len() - 1);
    self.ig_dist.fill_slice(inc.as_slice_mut().unwrap());

    for i in 1..out.len() {
      out[i] = out[i - 1] + inc[i - 1];
    }
  }
}

impl<T: FloatExt> PathSampler<T> for IgSampler<T> {
  type Output = Array1<T>;

  fn sample_into(&mut self, out: &mut Array1<T>) {
    self.fill_path(out.as_slice_mut().expect("Ig output must be contiguous"));
  }

  fn sample(&mut self) -> Array1<T> {
    let n = self.n;
    array1_from_fill(n, |out| self.fill_path(out))
  }
}

py_process_1d!(PyIg, Ig,
  sig: (gamma_, n, x0=None, t=None, seed=None, dtype=None),
  params: (gamma_: f64, n: usize, x0: Option<f64>, t: Option<f64>)
);

#[cfg(test)]
mod tests {
  use super::*;
  use crate::traits::ProcessExt;

  #[test]
  fn ig_path_is_non_decreasing() {
    let p = Ig::new(1.0_f64, 256, Some(0.0), Some(1.0), Unseeded);
    let x = p.sample();
    assert_eq!(x.len(), 256);
    assert!(x.windows(2).into_iter().all(|w| w[1] >= w[0]));
  }

  #[test]
  fn n_eq_1_keeps_initial_value() {
    let p = Ig::new(1.0_f64, 1, Some(3.5), Some(1.0), Unseeded);
    let x = p.sample();
    assert_eq!(x.len(), 1);
    assert_eq!(x[0], 3.5);
  }
}
