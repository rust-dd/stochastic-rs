//! Teng Modified Ou stochastic correlation (Eq. 19/20, Lemma 1).
//!
//! $$
//! dX_t = \kappa(\mu - \tanh(X_t))\,dt + \sigma\,dW_t, \quad \rho_t = \tanh(X_t)
//! $$
//!
//! The SCP ρ_t = tanh(X_t) satisfies:
//!
//! $$
//! d\rho_t = (1-\rho_t^2)\bigl(\kappa(\mu-\rho_t) - \sigma^2\rho_t\bigr)\,dt
//!         + (1-\rho_t^2)\sigma\,dW_t
//! $$
//!
//! Diffusion vanishes *quadratically* at ±1 (stronger confinement).
//! Closed-form stationary density: f(ρ̃) ∝ (1+ρ̃)^{a+b}(1−ρ̃)^{a−b}.

use ndarray::Array1;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::normal::SimdNormal;

use crate::traits::FloatExt;
use crate::traits::ProcessExt;

/// Teng modified-Ou stochastic correlation process.
///
/// Simulates in X-space via the modified Ou (Eq. 19):
/// dX_t = κ(μ − tanh(X_t)) dt + σ dW_t
///
/// Output ρ_t = tanh(X_t) ∈ (−1, 1).
#[derive(Debug, Clone)]
pub struct TengSCP<T: FloatExt, S: SeedExt = Unseeded> {
  /// Mean-reversion speed (κ > 0).
  pub kappa: T,
  /// Long-run correlation level (μ ∈ (−1, 1)).
  pub mu: T,
  /// Correlation volatility (σ > 0).
  pub sigma: T,
  /// Initial correlation (ρ₀ ∈ (−1, 1)).
  pub rho0: T,
  /// Number of discrete simulation points.
  pub n: usize,
  /// Total simulation horizon (defaults to 1).
  pub t: Option<T>,
  /// Seed strategy.
  pub seed: S,
}

impl<T: FloatExt> TengSCP<T> {
  pub fn new(kappa: T, mu: T, sigma: T, rho0: T, n: usize, t: Option<T>) -> Self {
    Self {
      kappa,
      mu,
      sigma,
      rho0,
      n,
      t,
      seed: Unseeded,
    }
  }
}

impl<T: FloatExt> TengSCP<T, Deterministic> {
  pub fn seeded(kappa: T, mu: T, sigma: T, rho0: T, n: usize, t: Option<T>, seed: u64) -> Self {
    Self {
      kappa,
      mu,
      sigma,
      rho0,
      n,
      t,
      seed: Deterministic(seed),
    }
  }
}

impl<T: FloatExt, S: SeedExt> TengSCP<T, S> {
  /// Reparametrised coefficients (Eq. 21):
  /// κ* = κ + σ², μ* = κμ/(κ+σ²), σ* = σ.
  pub fn kappa_star(&self) -> T {
    self.kappa + self.sigma * self.sigma
  }

  pub fn mu_star(&self) -> T {
    self.kappa * self.mu / self.kappa_star()
  }

  /// Stationary density exponents (Eq. 33/37):
  /// a = (κ − σ²)/σ², b = κμ/σ².
  pub fn density_a(&self) -> T {
    let s2 = self.sigma * self.sigma;
    (self.kappa - s2) / s2
  }

  pub fn density_b(&self) -> T {
    let s2 = self.sigma * self.sigma;
    self.kappa * self.mu / s2
  }

  /// Evaluate the (unnormalised) stationary density at ρ̃ ∈ (−1, 1).
  ///
  /// f(ρ̃) ∝ (1+ρ̃)^{a+b} (1−ρ̃)^{a−b}   (Eq. 39)
  pub fn stationary_density_unnorm(&self, rho: T) -> T {
    let a = self.density_a();
    let b = self.density_b();
    if rho <= -T::one() || rho >= T::one() {
      return T::zero();
    }
    let log_f = (a + b) * (T::one() + rho).ln() + (a - b) * (T::one() - rho).ln();
    log_f.exp()
  }

  /// Effective parameters κ*, μ*, σ* (van Emmerich form).
  pub fn effective_params(&self) -> (T, T, T) {
    (self.kappa_star(), self.mu_star(), self.sigma)
  }
}

impl<T: FloatExt, S: SeedExt> ProcessExt<T> for TengSCP<T, S> {
  type Output = Array1<T>;

  fn sample(&self) -> Self::Output {
    let mut seed = self.seed;
    let n_steps = self.n.saturating_sub(1);
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

    let mut rho = Array1::<T>::zeros(self.n);
    if self.n == 0 {
      return rho;
    }

    let x0 = self
      .rho0
      .clamp(T::from_f64_fast(-0.999), T::from_f64_fast(0.999))
      .atanh();
    let mut x = x0;
    rho[0] = x.tanh();

    for i in 1..self.n {
      // Modified Ou: dX = κ(μ - tanh(X))dt + σ dW
      let drift = self.kappa * (self.mu - x.tanh());
      x = x + drift * dt + self.sigma * gn[i - 1];
      rho[i] = x.tanh();
    }

    rho
  }
}

py_process_1d!(PyTengSCP, TengSCP,
  sig: (kappa, mu, sigma, rho0, n, t=None, seed=None, dtype=None),
  params: (kappa: f64, mu: f64, sigma: f64, rho0: f64, n: usize, t: Option<f64>)
);

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn stays_bounded() {
    let scp = TengSCP::seeded(8.0_f64, 0.0, 1.2, 0.3, 2000, Some(2.0), 123);
    let path = scp.sample();
    assert!(path.iter().all(|&r| r > -1.0 && r < 1.0));
  }

  #[test]
  fn mean_reverts() {
    let mu = -0.4_f64;
    let scp = TengSCP::seeded(12.0, mu, 0.5, 0.5, 5000, Some(10.0), 99);
    let path = scp.sample();
    let tail = &path.as_slice().unwrap()[4000..];
    let avg: f64 = tail.iter().sum::<f64>() / tail.len() as f64;
    assert!(
      (avg - mu).abs() < 0.15,
      "Expected mean near {mu}, got {avg}"
    );
  }

  #[test]
  fn stationary_density_peaks_near_mu() {
    let scp = TengSCP::new(8.0_f64, 0.3, 0.5, 0.0, 100, None);
    let d_at_mu = scp.stationary_density_unnorm(0.3);
    let d_at_0 = scp.stationary_density_unnorm(0.0);
    assert!(d_at_mu > d_at_0);
  }

  #[test]
  fn seeded_reproducibility() {
    let p1 = TengSCP::seeded(5.0_f64, 0.0, 0.8, 0.0, 200, Some(1.0), 42).sample();
    let p2 = TengSCP::seeded(5.0_f64, 0.0, 0.8, 0.0, 200, Some(1.0), 42).sample();
    for i in 0..200 {
      assert!((p1[i] - p2[i]).abs() < 1e-14, "diverged at i={i}");
    }
  }
}
