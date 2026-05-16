//! # Fractional Ou driven by Riemann–Liouville fBM (RFSV log-volatility)
//!
//! $$
//! dX_t = \kappa(\mu - X_t)\,dt + \nu\,dW^H_t,\qquad H \in (0, 1/2)
//! $$
//!
//! The RFSV log-volatility dynamics of Gatheral, Jaisson & Rosenbaum (2018).
//! The noise is RL-fBM generated non-cumulatively via the Bilokon–Wong
//! modified fast algorithm; the Ou drift is integrated with the usual Euler
//! rule on the resulting increments $\delta W^H_n = W^H_{t_{n+1}} - W^H_{t_n}$.
//!
//! Reference: Bayer C., Friz P., Gatheral J. *Pricing under rough volatility*.
//! Quantitative Finance 16 (2016), 887–904.
use ndarray::Array1;
use ndarray::Array2;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;

use super::markov_lift::RoughSimd;
use super::rl_fbm::RlFBm;
use crate::traits::FloatExt;
use crate::traits::ProcessExt;

/// Fractional Ornstein–Uhlenbeck driven by RL-fBM noise.
#[derive(Clone)]
pub struct RlFOU<T: FloatExt, S: SeedExt = Unseeded> {
  /// Hurst exponent of the driving fBM.
  pub hurst: T,
  /// Mean-reversion speed $\kappa$.
  pub kappa: T,
  /// Long-run mean $\mu$.
  pub mu: T,
  /// Diffusion scale $\nu$.
  pub sigma: T,
  /// Number of simulation points.
  pub n: usize,
  /// Initial value $X_0$.
  pub x0: Option<T>,
  /// Simulation horizon $T$.
  pub t: Option<T>,
  /// Quadrature degree passed through to the underlying [`RlFBm`].
  pub degree: Option<usize>,
  /// Seed strategy.
  pub seed: S,
  fbm: RlFBm<T>,
}

impl<T: FloatExt, S: SeedExt> RlFOU<T, S> {
  #[must_use]
  pub fn new(
    hurst: T,
    kappa: T,
    mu: T,
    sigma: T,
    n: usize,
    x0: Option<T>,
    t: Option<T>,
    degree: Option<usize>,
    seed: S,
  ) -> Self {
    assert!(n >= 2, "n must be at least 2");
    Self {
      hurst,
      kappa,
      mu,
      sigma,
      n,
      x0,
      t,
      degree,
      seed,
      fbm: RlFBm::new(hurst, n, t, degree, Unseeded),
    }
  }
}

impl<T: FloatExt + RoughSimd, S: SeedExt> RlFOU<T, S> {
  /// Generate $m$ independent RFSV log-volatility paths as an $(m, n)$ array.
  /// The RL-fBM noise is generated in a single batch via
  /// [`RlFBm::sample_batch`], then each path is Euler-integrated independently.
  pub fn sample_batch(&self, m: usize) -> Array2<T> {
    let fbm = self.fbm.sample_batch_impl(&self.seed.derive(), m);
    let dt = self.t.unwrap_or(T::one()) / T::from_usize_(self.n - 1);
    let x0 = self.x0.unwrap_or(T::zero());
    let mut out = Array2::<T>::zeros((m, self.n));

    for p in 0..m {
      out[[p, 0]] = x0;
      for i in 1..self.n {
        let dfbm = fbm[[p, i]] - fbm[[p, i - 1]];
        out[[p, i]] =
          out[[p, i - 1]] + self.kappa * (self.mu - out[[p, i - 1]]) * dt + self.sigma * dfbm;
      }
    }
    out
  }
}

impl<T: FloatExt + RoughSimd, S: SeedExt> ProcessExt<T> for RlFOU<T, S> {
  type Output = Array1<T>;

  fn sample(&self) -> Self::Output {
    let dt = self.t.unwrap_or(T::one()) / T::from_usize_(self.n - 1);
    let fbm = self.fbm.sample_impl(&self.seed.derive());

    let mut x = Array1::<T>::zeros(self.n);
    x[0] = self.x0.unwrap_or(T::zero());
    for i in 1..self.n {
      let dfbm = fbm[i] - fbm[i - 1];
      x[i] = x[i - 1] + self.kappa * (self.mu - x[i - 1]) * dt + self.sigma * dfbm;
    }
    x
  }
}

#[cfg(test)]
mod tests {
  use super::RlFOU;
  use crate::traits::ProcessExt;
  use stochastic_rs_core::simd_rng::Deterministic;
  use stochastic_rs_core::simd_rng::Unseeded;

  #[test]
  fn fou_sigma_zero_matches_deterministic_euler() {
    let kappa = 1.3_f64;
    let mu = 0.8_f64;
    let n = 129;
    let x0 = 0.2_f64;
    let t = 1.0_f64;

    let p = RlFOU::<f64>::new(0.3, kappa, mu, 0.0, n, Some(x0), Some(t), None, Unseeded);
    let x = p.sample();

    let dt = t / (n as f64 - 1.0);
    let mut expected = x0;
    for i in 1..n {
      expected = expected + kappa * (mu - expected) * dt;
      assert!((x[i] - expected).abs() < 1e-12, "mismatch at {i}");
    }
  }

  #[test]
  fn finite_output_at_typical_rfsv_parameters() {
    let p = RlFOU::new(
      0.1_f64,
      2.0,
      0.15_f64.ln(),
      0.25,
      512,
      Some(0.15_f64.ln()),
      Some(1.0),
      None,
      Deterministic::new(7),
    );
    let x = p.sample();
    assert_eq!(x.len(), 512);
    assert!(x.iter().all(|v| v.is_finite()));
  }
}
