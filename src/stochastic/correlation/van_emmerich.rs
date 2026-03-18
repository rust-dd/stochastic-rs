//! Van Emmerich / Jacobi-type stochastic correlation (Eq. 15).
//!
//! $$
//! d\rho_t = \kappa(\mu - \rho_t)\,dt + \sigma\sqrt{1 - \rho_t^2}\,dW_t
//! $$

use ndarray::Array1;

use crate::distributions::normal::SimdNormal;
use crate::simd_rng::Deterministic;
use crate::simd_rng::SeedExt;
use crate::simd_rng::Unseeded;
use crate::traits::FloatExt;
use crate::traits::ProcessExt;

/// Van Emmerich stochastic correlation process (Eq. 15 in Teng et al. 2016).
///
/// The diffusion coefficient vanishes *linearly* at ±1, keeping the
/// process inside (−1, 1) when κ ≥ σ²/(1 ± μ).
pub struct VanEmmerich<T: FloatExt, S: SeedExt = Unseeded> {
  pub kappa: T,
  pub mu: T,
  pub sigma: T,
  pub rho0: T,
  pub n: usize,
  pub t: Option<T>,
  pub seed: S,
}

impl<T: FloatExt> VanEmmerich<T> {
  pub fn new(kappa: T, mu: T, sigma: T, rho0: T, n: usize, t: Option<T>) -> Self {
    Self { kappa, mu, sigma, rho0, n, t, seed: Unseeded }
  }
}

impl<T: FloatExt> VanEmmerich<T, Deterministic> {
  pub fn seeded(kappa: T, mu: T, sigma: T, rho0: T, n: usize, t: Option<T>, seed: u64) -> Self {
    Self { kappa, mu, sigma, rho0, n, t, seed: Deterministic(seed) }
  }
}

impl<T: FloatExt, S: SeedExt> ProcessExt<T> for VanEmmerich<T, S> {
  type Output = Array1<T>;

  fn sample(&self) -> Self::Output {
    let mut seed = self.seed;
    let n_steps = self.n.saturating_sub(1);
    if self.n == 0 {
      return Array1::zeros(0);
    }
    let dt = if n_steps > 0 {
      self.t.unwrap_or(T::one()) / T::from_usize_(n_steps)
    } else {
      T::zero()
    };
    let sqrt_dt = dt.sqrt();

    let mut gn = Array1::<T>::zeros(n_steps);
    if let Some(slice) = gn.as_slice_mut() {
      let normal = SimdNormal::<T>::from_seed_source(T::zero(), sqrt_dt, &mut seed);
      normal.fill_slice_fast(slice);
    }

    let mut rho = Array1::zeros(self.n);
    rho[0] = self.rho0;
    let clamp_lo = T::from_f64_fast(-0.9999);
    let clamp_hi = T::from_f64_fast(0.9999);

    for i in 1..self.n {
      let r = rho[i - 1];
      let one_minus_r2 = (T::one() - r * r).max(T::zero());

      let drift = self.kappa * (self.mu - r) * dt;
      let diffusion = self.sigma * one_minus_r2.sqrt() * gn[i - 1];

      rho[i] = (r + drift + diffusion).clamp(clamp_lo, clamp_hi);
    }

    rho
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn stays_bounded() {
    let scp = VanEmmerich::seeded(5.0_f64, -0.3, 0.8, -0.3, 1000, Some(1.0), 42);
    let path = scp.sample();
    assert_eq!(path.len(), 1000);
    assert!(path.iter().all(|&r| r > -1.0 && r < 1.0));
  }
}
