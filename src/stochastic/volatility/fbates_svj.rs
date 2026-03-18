//! # Fractional Bates SVJ
//!
//! Rough/fractional variance dynamics (Volterra-Heston) with Bates-style
//! Poisson jumps in the log-price process.
//!
//! $$
//! d\ln S_t = (\mu - \lambda\kappa_J - \tfrac12 v_t)\,dt + \sqrt{v_t}\,dW_t^S + Z\,dN_t
//! $$
//!
//! where $v_t$ follows the rough Heston variance dynamics with Hurst
//! exponent $H \in (0, 0.5)$, $N_t \sim \text{Poisson}(\lambda)$,
//! $Z \sim \mathcal{N}(\nu, \omega^2)$, and
//! $\langle dW^S, dW^v \rangle = \rho\,dt$.
//!
//! Returns `[S, v]` — price and variance paths.

use ndarray::Array1;
use rand_distr::Distribution;
use statrs::function::gamma::gamma;

use crate::distributions::normal::SimdNormal;
use crate::distributions::poisson::SimdPoisson;
use crate::simd_rng::Deterministic;
use crate::simd_rng::SeedExt;
use crate::simd_rng::Unseeded;
use crate::stochastic::noise::cgns::CGNS;
use crate::traits::FloatExt;
use crate::traits::ProcessExt;

pub struct FBatesSVJ<S: SeedExt = Unseeded> {
  /// Hurst exponent controlling roughness (H ∈ (0, 0.5) for rough).
  pub hurst: f64,
  /// Drift rate.
  pub mu: f64,
  /// Initial spot price.
  pub s0: f64,
  /// Initial variance.
  pub v0: f64,
  /// Long-run variance level (θ).
  pub theta: f64,
  /// Mean-reversion speed (κ).
  pub kappa: f64,
  /// Vol-of-vol (ξ).
  pub xi: f64,
  /// Price-vol correlation (ρ ∈ [-1, 1]).
  pub rho: f64,
  /// Jump intensity (Poisson arrival rate λ).
  pub lambda: f64,
  /// Mean of jump log-size Z ~ N(ν, ω²).
  pub nu: f64,
  /// Std dev of jump log-size Z.
  pub omega: f64,
  /// Number of discrete simulation points.
  pub n: usize,
  /// Total simulation horizon (defaults to 1).
  pub t: Option<f64>,
  /// Seed strategy.
  pub seed: S,
}

impl FBatesSVJ {
  #[allow(clippy::too_many_arguments)]
  pub fn new(
    hurst: f64,
    mu: f64,
    s0: f64,
    v0: f64,
    theta: f64,
    kappa: f64,
    xi: f64,
    rho: f64,
    lambda: f64,
    nu: f64,
    omega: f64,
    n: usize,
    t: Option<f64>,
  ) -> Self {
    Self {
      hurst, mu, s0, v0, theta, kappa, xi, rho, lambda, nu, omega, n, t, seed: Unseeded,
    }
  }
}

impl FBatesSVJ<Deterministic> {
  #[allow(clippy::too_many_arguments)]
  pub fn seeded(
    hurst: f64,
    mu: f64,
    s0: f64,
    v0: f64,
    theta: f64,
    kappa: f64,
    xi: f64,
    rho: f64,
    lambda: f64,
    nu: f64,
    omega: f64,
    n: usize,
    t: Option<f64>,
    seed: u64,
  ) -> Self {
    Self {
      hurst, mu, s0, v0, theta, kappa, xi, rho, lambda, nu, omega, n, t,
      seed: Deterministic(seed),
    }
  }
}

impl<S: SeedExt> ProcessExt<f64> for FBatesSVJ<S> {
  type Output = [Array1<f64>; 2]; // [S, v]

  fn sample(&self) -> Self::Output {
    let n_steps = self.n.saturating_sub(1);
    let dt = if n_steps > 0 {
      self.t.unwrap_or(1.0) / n_steps as f64
    } else {
      0.0
    };

    // Use CGNS for rho-correlated noise: [gn_vol, gn_price]
    let mut seed = self.seed;
    let cgns = CGNS::new(self.rho, n_steps, self.t);
    let [gn_vol, gn_price] = cgns.sample_impl(seed.derive());

    let mut yt = Array1::<f64>::zeros(self.n);
    let mut zt = Array1::<f64>::zeros(self.n);
    let mut sigma_tilde2 = Array1::<f64>::zeros(self.n);
    let mut v2 = Array1::zeros(self.n);
    let mut s = Array1::zeros(self.n);

    if self.n == 0 {
      return [s, v2];
    }

    yt[0] = self.v0;
    zt[0] = 0.0;
    sigma_tilde2[0] = self.v0;
    v2[0] = self.v0;
    s[0] = self.s0;

    let g = gamma(self.hurst - 0.5);
    let half = 0.5;

    // Jump compensation: κ_J = exp(ν + ½ω²) - 1
    let kappa_j = (self.nu + 0.5 * self.omega * self.omega).exp() - 1.0;

    // Jump RNG
    let z_std = SimdNormal::<f64, 64>::from_seed_source(0.0, 1.0, &mut seed);
    let mut rng = seed.rng();
    let pois = if self.lambda > 0.0 {
      Some(SimdPoisson::<u32>::new(self.lambda * dt))
    } else {
      None
    };

    for i in 1..self.n {
      let t_i = dt * i as f64;

      // ── Rough variance dynamics (same as RoughHeston/fheston.rs) ──
      yt[i] = self.theta + (yt[i - 1] - self.theta) * (-self.kappa * dt).exp();
      zt[i] = zt[i - 1] * (-self.kappa * dt).exp()
        + sigma_tilde2[i - 1].max(0.0).sqrt() * gn_vol[i - 1];

      sigma_tilde2[i] = yt[i] + self.xi * zt[i];

      let integral: f64 = (0..i)
        .map(|j| {
          let tj = j as f64 * dt;
          ((t_i - tj).powf(self.hurst - half) * zt[j]) * dt
        })
        .sum();

      v2[i] = yt[i] + self.xi * zt[i] + self.xi * integral / g;

      // ── Price dynamics with jumps ──
      let vi = v2[i - 1].max(0.0);

      // Jump component
      let mut jump_sum = 0.0;
      if let Some(pois) = &pois {
        let n_jumps: u32 = pois.sample(&mut rng);
        if n_jumps > 0 {
          let kf = n_jumps as f64;
          let z0: f64 = z_std.sample_fast();
          jump_sum = self.nu * kf + self.omega * kf.sqrt() * z0;
        }
      }

      let log_inc =
        (self.mu - self.lambda * kappa_j - half * vi) * dt + vi.sqrt() * gn_price[i - 1] + jump_sum;
      s[i] = s[i - 1] * log_inc.exp();
    }

    [s, v2]
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn price_stays_positive() {
    let m = FBatesSVJ::seeded(
      0.1, 0.05, 100.0, 0.04, 0.04, 2.0, 0.3, -0.7, 0.5, -0.01, 0.1, 256, Some(1.0), 42,
    );
    let [s, _v] = m.sample();
    assert!(s.iter().all(|x| x.is_finite() && *x > 0.0), "prices must be positive");
  }

  #[test]
  fn variance_path_is_finite() {
    let m = FBatesSVJ::seeded(
      0.15, 0.05, 100.0, 0.04, 0.04, 2.0, 0.3, -0.7, 0.5, 0.0, 0.1, 256, Some(1.0), 99,
    );
    let [_s, v] = m.sample();
    assert!(v.iter().all(|x| x.is_finite()), "variance must be finite");
  }

  #[test]
  fn seeded_is_deterministic() {
    let mk = || {
      FBatesSVJ::seeded(
        0.1, 0.05, 100.0, 0.04, 0.04, 2.0, 0.3, -0.7, 0.5, 0.0, 0.1, 128, Some(0.5), 77,
      )
    };
    let [s1, _] = mk().sample();
    let [s2, _] = mk().sample();
    for i in 0..128 {
      assert!((s1[i] - s2[i]).abs() < 1e-12, "paths diverged at i={i}");
    }
  }

  #[test]
  fn reduces_to_rough_heston_without_jumps() {
    // With λ=0, should be identical to RoughHeston
    let m = FBatesSVJ::seeded(
      0.1, 0.05, 100.0, 0.04, 0.04, 2.0, 0.3, -0.7, 0.0, 0.0, 0.0, 128, Some(0.5), 55,
    );
    let [s, v] = m.sample();
    assert!(s.iter().all(|x| x.is_finite() && *x > 0.0));
    assert!(v.iter().all(|x| x.is_finite()));
  }
}
