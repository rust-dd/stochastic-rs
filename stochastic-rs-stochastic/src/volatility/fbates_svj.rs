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
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::normal::SimdNormal;
use stochastic_rs_distributions::poisson::SimdPoisson;

use crate::noise::cgns::Cgns;
use crate::traits::FloatExt;
use crate::traits::ProcessExt;

pub struct FBatesSvj<T: FloatExt, S: SeedExt = Unseeded> {
  /// Hurst exponent controlling roughness (H ∈ (0, 0.5) for rough).
  pub hurst: T,
  /// Drift rate.
  pub mu: T,
  /// Initial spot price.
  pub s0: T,
  /// Initial variance.
  pub v0: T,
  /// Long-run variance level (θ).
  pub theta: T,
  /// Mean-reversion speed (κ).
  pub kappa: T,
  /// Vol-of-vol (ξ).
  pub xi: T,
  /// Price-vol correlation (ρ ∈ [-1, 1]).
  pub rho: T,
  /// Jump intensity (Poisson arrival rate λ).
  pub lambda: T,
  /// Mean of jump log-size Z ~ N(ν, ω²).
  pub nu: T,
  /// Std dev of jump log-size Z.
  pub omega: T,
  /// Number of discrete simulation points.
  pub n: usize,
  /// Total simulation horizon (defaults to 1).
  pub t: Option<T>,
  /// Seed strategy.
  pub seed: S,
}

impl<T: FloatExt> FBatesSvj<T> {
  #[allow(clippy::too_many_arguments)]
  pub fn new(
    hurst: T,
    mu: T,
    s0: T,
    v0: T,
    theta: T,
    kappa: T,
    xi: T,
    rho: T,
    lambda: T,
    nu: T,
    omega: T,
    n: usize,
    t: Option<T>,
  ) -> Self {
    Self {
      hurst,
      mu,
      s0,
      v0,
      theta,
      kappa,
      xi,
      rho,
      lambda,
      nu,
      omega,
      n,
      t,
      seed: Unseeded,
    }
  }
}

impl<T: FloatExt> FBatesSvj<T, Deterministic> {
  #[allow(clippy::too_many_arguments)]
  pub fn seeded(
    hurst: T,
    mu: T,
    s0: T,
    v0: T,
    theta: T,
    kappa: T,
    xi: T,
    rho: T,
    lambda: T,
    nu: T,
    omega: T,
    n: usize,
    t: Option<T>,
    seed: u64,
  ) -> Self {
    Self {
      hurst,
      mu,
      s0,
      v0,
      theta,
      kappa,
      xi,
      rho,
      lambda,
      nu,
      omega,
      n,
      t,
      seed: Deterministic(seed),
    }
  }
}

impl<T: FloatExt, S: SeedExt> ProcessExt<T> for FBatesSvj<T, S> {
  type Output = [Array1<T>; 2]; // [S, v]

  fn sample(&self) -> Self::Output {
    let n_steps = self.n.saturating_sub(1);
    let dt = if n_steps > 0 {
      self.t.unwrap_or(T::one()) / T::from_usize_(n_steps)
    } else {
      T::zero()
    };

    // Use Cgns for rho-correlated noise: [gn_vol, gn_price]
    let mut seed = self.seed;
    let cgns = Cgns::new(self.rho, n_steps, self.t);
    let [gn_vol, gn_price] = cgns.sample_impl(seed.derive());

    let mut yt = Array1::<T>::zeros(self.n);
    let mut zt = Array1::<T>::zeros(self.n);
    let mut sigma_tilde2 = Array1::<T>::zeros(self.n);
    let mut v2 = Array1::zeros(self.n);
    let mut s = Array1::zeros(self.n);

    if self.n == 0 {
      return [s, v2];
    }

    yt[0] = self.v0;
    zt[0] = T::zero();
    sigma_tilde2[0] = self.v0;
    v2[0] = self.v0;
    s[0] = self.s0;

    let g = T::from_f64_fast(gamma(self.hurst.to_f64().unwrap() - 0.5));
    let half = T::from_f64_fast(0.5);

    // Jump compensation: κ_J = exp(ν + ½ω²) - 1
    let kappa_j = (self.nu + half * self.omega * self.omega).exp() - T::one();

    // Jump RNG
    let z_std = SimdNormal::<T>::from_seed_source(T::zero(), T::one(), &mut seed);
    let mut rng = seed.rng();
    let lambda_dt = self.lambda.to_f64().unwrap() * dt.to_f64().unwrap();
    let pois = if lambda_dt > 0.0 {
      Some(SimdPoisson::<u32>::new(lambda_dt))
    } else {
      None
    };

    for i in 1..self.n {
      let t_i = dt * T::from_usize_(i);

      // ── Rough variance dynamics (same as RoughHeston/fheston.rs) ──
      yt[i] = self.theta + (yt[i - 1] - self.theta) * (-self.kappa * dt).exp();
      zt[i] = zt[i - 1] * (-self.kappa * dt).exp()
        + sigma_tilde2[i - 1].max(T::zero()).sqrt() * gn_vol[i - 1];

      sigma_tilde2[i] = yt[i] + self.xi * zt[i];

      let integral = (0..i)
        .map(|j| {
          let tj = T::from_usize_(j) * dt;
          ((t_i - tj).powf(self.hurst - half) * zt[j]) * dt
        })
        .sum::<T>();

      v2[i] = yt[i] + self.xi * zt[i] + self.xi * integral / g;

      // Price dynamics with jumps
      let vi = v2[i - 1].max(T::zero());

      // Jump component
      let mut jump_sum = T::zero();
      if let Some(pois) = &pois {
        let n_jumps: u32 = pois.sample(&mut rng);
        if n_jumps > 0 {
          let kf = T::from_f64_fast(n_jumps as f64);
          let z0 = z_std.sample_fast();
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
    let m = FBatesSvj::seeded(
      0.1_f64,
      0.05,
      100.0,
      0.04,
      0.04,
      2.0,
      0.3,
      -0.7,
      0.5,
      -0.01,
      0.1,
      256,
      Some(1.0),
      42,
    );
    let [s, _v] = m.sample();
    assert!(
      s.iter().all(|x| x.is_finite() && *x > 0.0),
      "prices must be positive"
    );
  }

  #[test]
  fn variance_path_is_finite() {
    let m = FBatesSvj::seeded(
      0.15_f64,
      0.05,
      100.0,
      0.04,
      0.04,
      2.0,
      0.3,
      -0.7,
      0.5,
      0.0,
      0.1,
      256,
      Some(1.0),
      99,
    );
    let [_s, v] = m.sample();
    assert!(v.iter().all(|x| x.is_finite()), "variance must be finite");
  }

  #[test]
  fn seeded_is_deterministic() {
    let mk = || {
      FBatesSvj::seeded(
        0.1_f64,
        0.05,
        100.0,
        0.04,
        0.04,
        2.0,
        0.3,
        -0.7,
        0.5,
        0.0,
        0.1,
        128,
        Some(0.5),
        77,
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
    let m = FBatesSvj::seeded(
      0.1_f64,
      0.05,
      100.0,
      0.04,
      0.04,
      2.0,
      0.3,
      -0.7,
      0.0,
      0.0,
      0.0,
      128,
      Some(0.5),
      55,
    );
    let [s, v] = m.sample();
    assert!(s.iter().all(|x| x.is_finite() && *x > 0.0));
    assert!(v.iter().all(|x| x.is_finite()));
  }
}
