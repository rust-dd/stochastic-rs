//! Van Emmerich / Jacobi-type stochastic correlation (Eq. 15).
//!
//! $$
//! d\rho_t = \kappa(\mu - \rho_t)\,dt + \sigma\sqrt{1 - \rho_t^2}\,dW_t
//! $$

use ndarray::Array1;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::normal::SimdNormal;

use crate::buffer::array1_from_fill;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

/// Van Emmerich stochastic correlation process (Eq. 15 in Teng et al. 2016).
///
/// The diffusion coefficient vanishes *linearly* at ±1, keeping the
/// process inside (−1, 1) when κ ≥ σ²/(1 ± μ).
#[derive(Debug, Clone)]
pub struct VanEmmerich<T: FloatExt, S: SeedExt = Unseeded> {
  pub kappa: T,
  pub mu: T,
  pub sigma: T,
  pub rho0: T,
  pub n: usize,
  pub t: Option<T>,
  pub seed: S,
}

impl<T: FloatExt, S: SeedExt> VanEmmerich<T, S> {
  pub fn new(kappa: T, mu: T, sigma: T, rho0: T, n: usize, t: Option<T>, seed: S) -> Self {
    Self {
      kappa,
      mu,
      sigma,
      rho0,
      n,
      t,
      seed,
    }
  }
}

impl<T: FloatExt, S: SeedExt> ProcessExt<T> for VanEmmerich<T, S> {
  type Output = Array1<T>;
  type Sampler<'s>
    = VanEmmerichSampler<T>
  where
    Self: 's;

  fn sampler(&self) -> VanEmmerichSampler<T> {
    let n_steps = self.n.saturating_sub(1);
    let dt = if n_steps > 0 {
      self.t.unwrap_or(T::one()) / T::from_usize_(n_steps)
    } else {
      T::zero()
    };
    VanEmmerichSampler {
      n: self.n,
      kappa: self.kappa,
      mu: self.mu,
      sigma: self.sigma,
      rho0: self.rho0,
      dt,
      normal: SimdNormal::<T>::new(T::zero(), dt.sqrt(), &self.seed),
    }
  }
}

/// Reusable [`VanEmmerich`] sampling state: owns the Gaussian source and the
/// precomputed step size. `fill_path` Euler-steps the Jacobi-type diffusion
/// with the linear `√(1−ρ²)` coefficient and clamps to (−1, 1) in place; the
/// owned source advances each call for independent paths.
#[doc(hidden)]
pub struct VanEmmerichSampler<T: FloatExt> {
  n: usize,
  kappa: T,
  mu: T,
  sigma: T,
  rho0: T,
  dt: T,
  normal: SimdNormal<T>,
}

impl<T: FloatExt> VanEmmerichSampler<T> {
  fn fill_path(&mut self, out: &mut [T]) {
    if out.is_empty() {
      return;
    }
    let n_steps = out.len() - 1;
    let mut gn = Array1::<T>::zeros(n_steps);
    if let Some(slice) = gn.as_slice_mut() {
      self.normal.fill_slice_fast(slice);
    }

    out[0] = self.rho0;
    let clamp_lo = T::from_f64_fast(-0.9999);
    let clamp_hi = T::from_f64_fast(0.9999);

    for i in 1..out.len() {
      let r = out[i - 1];
      let one_minus_r2 = (T::one() - r * r).max(T::zero());

      let drift = self.kappa * (self.mu - r) * self.dt;
      let diffusion = self.sigma * one_minus_r2.sqrt() * gn[i - 1];

      out[i] = (r + drift + diffusion).clamp(clamp_lo, clamp_hi);
    }
  }
}

impl<T: FloatExt> PathSampler<T> for VanEmmerichSampler<T> {
  type Output = Array1<T>;

  fn sample_into(&mut self, out: &mut Array1<T>) {
    let slice = out
      .as_slice_mut()
      .expect("VanEmmerich output must be contiguous");
    self.fill_path(slice);
  }

  fn sample(&mut self) -> Array1<T> {
    let n = self.n;
    array1_from_fill(n, |out| self.fill_path(out))
  }
}

#[cfg(test)]
mod tests {
  use stochastic_rs_core::simd_rng::Deterministic;

  use super::*;

  #[test]
  fn stays_bounded() {
    let scp = VanEmmerich::new(
      5.0_f64,
      -0.3,
      0.8,
      -0.3,
      1000,
      Some(1.0),
      Deterministic::new(42),
    );
    let path = scp.sample();
    assert_eq!(path.len(), 1000);
    assert!(path.iter().all(|&r| r > -1.0 && r < 1.0));
  }
}
