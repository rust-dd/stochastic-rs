//! # Hawkes Jump Diffusion
//!
//! $$
//! \frac{dS_t}{S_{t^-}} = \mu\,dt + \sigma\,dW_t + (Y-1)\,dN_t,\quad
//! d\lambda_t = \beta(\mu_\lambda - \lambda_t)\,dt + \alpha\,dN_t
//! $$
//!
//! Jump diffusion where jump arrivals follow a self-exciting Hawkes process
//! instead of a constant-rate Poisson process. Produces volatility clustering
//! and serially correlated jumps.
//!
//! Reference:
//! - Hawkes (1971), "Spectra of some self-exciting and mutually exciting point processes"
//! - Merton (1976), "Option pricing when underlying stock returns are discontinuous"

use ndarray::Array1;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::normal::SimdNormal;
use stochastic_rs_distributions::uniform::SimdUniform;

use crate::traits::FloatExt;
use crate::traits::ProcessExt;

/// Hawkes-driven jump diffusion process (log-price).
///
/// Combines Gbm diffusion with self-exciting Hawkes jump arrivals
/// and log-normal jump sizes.
pub struct HawkesJD<T: FloatExt, S: SeedExt = Unseeded> {
  /// Drift $\mu$.
  pub mu: T,
  /// Diffusion volatility $\sigma$.
  pub sigma: T,
  /// Hawkes baseline intensity $\mu_\lambda$.
  pub mu_lambda: T,
  /// Hawkes excitation magnitude $\alpha \ge 0$.
  pub alpha: T,
  /// Hawkes mean-reversion rate $\beta > 0$.
  pub beta: T,
  /// Mean of log-jump size $\mu_J$.
  pub mu_j: T,
  /// Std of log-jump size $\sigma_J$.
  pub sigma_j: T,
  /// Number of time steps.
  pub n: usize,
  /// Initial log-price $X_0$.
  pub x0: Option<T>,
  /// Time horizon $T$.
  pub t: Option<T>,
  pub seed: S,
}

impl<T: FloatExt, S: SeedExt> HawkesJD<T, S> {
  pub fn new(
    mu: T,
    sigma: T,
    mu_lambda: T,
    alpha: T,
    beta: T,
    mu_j: T,
    sigma_j: T,
    n: usize,
    x0: Option<T>,
    t: Option<T>,
    seed: S,
  ) -> Self {
    assert!(beta > T::zero(), "beta must be positive");
    assert!(alpha >= T::zero(), "alpha must be non-negative");
    assert!(
      alpha / beta < T::one(),
      "stationarity requires alpha/beta < 1"
    );
    Self {
      mu,
      sigma,
      mu_lambda,
      alpha,
      beta,
      mu_j,
      sigma_j,
      n,
      x0,
      t,
      seed,
    }
  }
}

impl<T: FloatExt, S: SeedExt> ProcessExt<T> for HawkesJD<T, S> {
  type Output = Array1<T>;

  fn sample(&self) -> Self::Output {
    let t_max = self.t.unwrap_or(T::one());
    let dt = t_max / T::from_usize_(self.n - 1);
    let sqrt_dt = dt.sqrt();
    let two = T::from_usize_(2);

    let normal = SimdNormal::<T, 64>::from_seed_source(T::zero(), T::one(), &self.seed);
    let uniform = SimdUniform::from_seed_source(T::zero(), T::one(), &self.seed);
    let jump_normal = SimdNormal::<T, 64>::from_seed_source(self.mu_j, self.sigma_j, &self.seed);

    let mut x = Array1::<T>::zeros(self.n);
    x[0] = self.x0.unwrap_or(T::zero());

    let mut lambda = self.mu_lambda;

    for i in 1..self.n {
      // Diffusion
      let dw = normal.sample_fast() * sqrt_dt;
      let drift = (self.mu - self.sigma * self.sigma / two) * dt;

      // Hawkes intensity: check for jump in [t_{i-1}, t_i]
      let jump_prob = lambda * dt;
      let u = uniform.sample_fast();
      let jump = if u < jump_prob {
        // Jump occurs — excite intensity
        lambda += self.alpha;
        jump_normal.sample_fast()
      } else {
        T::zero()
      };

      // Mean-revert intensity
      lambda = lambda + self.beta * (self.mu_lambda - lambda) * dt;
      lambda = lambda.max(T::zero());

      x[i] = x[i - 1] + drift + self.sigma * dw + jump;
    }

    x
  }
}

py_process_1d!(PyHawkesJD, HawkesJD,
  sig: (mu, sigma, mu_lambda, alpha, beta, mu_j, sigma_j, n, x0=None, t=None, seed=None, dtype=None),
  params: (mu: f64, sigma: f64, mu_lambda: f64, alpha: f64, beta: f64, mu_j: f64, sigma_j: f64, n: usize, x0: Option<f64>, t: Option<f64>)
);

#[cfg(test)]
mod tests {
  use stochastic_rs_core::simd_rng::Deterministic;

  use super::*;

  #[test]
  fn hawkes_jd_runs() {
    let hjd = HawkesJD::new(
      0.05,
      0.2,
      3.0,
      1.5,
      5.0,
      -0.01,
      0.05,
      252,
      Some(0.0),
      Some(1.0),
      Unseeded,
    );
    let path = hjd.sample();
    assert_eq!(path.len(), 252);
    assert!(path[0] == 0.0);
  }

  #[test]
  fn hawkes_jd_clustering() {
    // Hawkes with high excitation should produce clustered jumps
    let hjd = HawkesJD::new(
      0.05,
      0.2,
      5.0,
      3.0,
      8.0,
      -0.02,
      0.05,
      1000,
      Some(0.0),
      Some(1.0),
      Unseeded,
    );
    let path = hjd.sample();
    assert_eq!(path.len(), 1000);
    // Path should have non-trivial movement
    // Path should differ from initial value
    assert!(path[path.len() - 1] != path[0], "path should have movement");
  }

  #[test]
  fn hawkes_jd_seeded_deterministic() {
    // Two separately built instances with the same seed reproduce each other's
    // first path. (Same instance, repeated `.sample()` calls advance the seed
    // state and produce different paths — desired for Monte Carlo reuse.)
    let mk = || {
      HawkesJD::new(
        0.05,
        0.2,
        3.0,
        1.5,
        5.0,
        -0.01,
        0.05,
        100,
        Some(0.0),
        Some(1.0),
        Deterministic::new(42),
      )
    };
    let a = mk();
    let b = mk();
    assert_eq!(a.sample(), b.sample());
  }
}
