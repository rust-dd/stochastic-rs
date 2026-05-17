//! # Heston Model with Stochastic Correlation (HSCM)
//!
//! Teng, Ehrhardt & Günther (2016), "On the Heston model with stochastic
//! correlation", Int. J. Theor. Appl. Finance 19(6).
//!
//! $$
//! dS = rS\,dt + \sqrt{v}\,S\,dW^S \\
//! dv = \kappa_v(\mu_v - v)\,dt + \sigma_v\sqrt{v}\,dW^v \\
//! d\rho = \kappa_r(\mu_r - \tanh(X))\,dt + \sigma_r\,dW^\rho,\quad \rho=\tanh(X)
//! $$
//!
//! With correlations (Eq. 2.4, ρ₁ = 0 assumption):
//!   dW^S·dW^v = ρ_t dt,    dW^v·dW^ρ = ρ₂ dt
//!
//! Returns `[S, v, ρ]` — three paths.

use ndarray::Array1;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::normal::SimdNormal;

use crate::traits::FloatExt;
use crate::traits::ProcessExt;

/// Single-asset Heston model with stochastic price-vol correlation.
///
/// The correlation ρ_t between the asset log-return and variance
/// innovations follows a modified Ou process (Eq. 2.10):
///
/// dρ_t = (1−ρ²)\[κ_r(μ_r−ρ_t)−σ_r²ρ_t\]dt + (1−ρ²)σ_r dW^ρ
///
/// The log-price Cholesky decomposition (Eq. 2.7 with ρ₁=0):
///
/// dx = (r−½v)dt + ρ_t√v dW̃^ν + ρ₂√v dW̃^ρ + √(1−ρ_t²−ρ₂²)√v dW̃^x
#[derive(Debug, Clone)]
pub struct HestonStochCorr<T: FloatExt, S: SeedExt = Unseeded> {
  // Market
  /// Risk-free rate.
  pub r: T,
  /// Initial spot price.
  pub s0: T,

  // Cir variance: dv = κ_v(μ_v − v)dt + σ_v√v dW^v
  /// Initial variance.
  pub v0: T,
  /// Mean-reversion speed of variance.
  pub kappa_v: T,
  /// Long-run variance level.
  pub mu_v: T,
  /// Vol-of-vol.
  pub sigma_v: T,

  // Stochastic correlation (modified Ou, tanh): ρ = tanh(X)
  /// Initial correlation (ρ₀ ∈ (−1, 1)).
  pub rho0: T,
  /// Mean-reversion speed of correlation.
  pub kappa_r: T,
  /// Long-run correlation level (μ_r ∈ (−1, 1)).
  pub mu_r: T,
  /// Volatility of correlation process.
  pub sigma_r: T,

  /// Constant correlation between dW^v and dW^ρ.
  pub rho2: T,

  /// Number of discrete simulation points.
  pub n: usize,
  /// Total simulation horizon (defaults to 1).
  pub t: Option<T>,
  /// Seed strategy.
  pub seed: S,
}

impl<T: FloatExt, S: SeedExt> HestonStochCorr<T, S> {
  #[allow(clippy::too_many_arguments)]
  pub fn new(
    r: T,
    s0: T,
    v0: T,
    kappa_v: T,
    mu_v: T,
    sigma_v: T,
    rho0: T,
    kappa_r: T,
    mu_r: T,
    sigma_r: T,
    rho2: T,
    n: usize,
    t: Option<T>,
    seed: S,
  ) -> Self {
    Self {
      r,
      s0,
      v0,
      kappa_v,
      mu_v,
      sigma_v,
      rho0,
      kappa_r,
      mu_r,
      sigma_r,
      rho2,
      n,
      t,
      seed,
    }
  }
}

impl<T: FloatExt, S: SeedExt> ProcessExt<T> for HestonStochCorr<T, S> {
  type Output = [Array1<T>; 3]; // [S, v, rho]

  fn sample(&self) -> Self::Output {
    let n_steps = self.n.saturating_sub(1);
    let dt = if n_steps > 0 {
      self.t.unwrap_or(T::one()) / T::from_usize_(n_steps)
    } else {
      T::zero()
    };
    let sqrt_dt = dt.sqrt();
    let half = T::from_f64_fast(0.5);
    let zero = T::zero();

    // 3 independent noise streams: dW̃^ν, dW̃^ρ, dW̃^x
    let gen_noise = |seed: &S| -> Array1<T> {
      let mut gn = Array1::<T>::zeros(n_steps);
      if let Some(slice) = gn.as_slice_mut() {
        let normal = SimdNormal::<T>::new(zero, sqrt_dt, seed);
        normal.fill_slice_fast(slice);
      }
      gn
    };

    let dw_v = gen_noise(&self.seed); // variance Bm
    let dw_rho = gen_noise(&self.seed); // correlation Bm
    let dw_x = gen_noise(&self.seed); // independent price Bm

    let mut s_path = Array1::<T>::zeros(self.n);
    let mut v_path = Array1::<T>::zeros(self.n);
    let mut rho_path = Array1::<T>::zeros(self.n);

    if self.n == 0 {
      return [s_path, v_path, rho_path];
    }

    s_path[0] = self.s0;
    v_path[0] = self.v0;

    // Correlation: work in X-space, X₀ = atanh(ρ₀)
    let mut x_corr = self
      .rho0
      .clamp(T::from_f64_fast(-0.999), T::from_f64_fast(0.999))
      .atanh();
    rho_path[0] = x_corr.tanh();

    for i in 1..self.n {
      // ── Correlation dynamics (modified Ou in X-space) ──
      let corr_drift = self.kappa_r * (self.mu_r - x_corr.tanh());
      x_corr = x_corr + corr_drift * dt + self.sigma_r * dw_rho[i - 1];
      let rho_t = x_corr.tanh();
      rho_path[i] = rho_t;

      // ── Variance dynamics (Cir) ──
      let v_prev = v_path[i - 1].max(zero);
      v_path[i] = (v_prev
        + self.kappa_v * (self.mu_v - v_prev) * dt
        + self.sigma_v * v_prev.sqrt() * dw_v[i - 1])
        .max(zero);

      // ── Log-price dynamics (Cholesky, Eq. 2.7 with ρ₁=0) ──
      // dx = (r − ½v)dt + ρ_t√v dW̃^ν + ρ₂√v dW̃^ρ + √(1−ρ_t²−ρ₂²)√v dW̃^x
      let rho2_sq = self.rho2 * self.rho2;
      let indep_coeff = (T::one() - rho_t * rho_t - rho2_sq).max(zero).sqrt();
      let sqrt_v = v_prev.sqrt();

      let log_inc = (self.r - half * v_prev) * dt
        + rho_t * sqrt_v * dw_v[i - 1]
        + self.rho2 * sqrt_v * dw_rho[i - 1]
        + indep_coeff * sqrt_v * dw_x[i - 1];

      s_path[i] = s_path[i - 1] * log_inc.exp();
    }

    [s_path, v_path, rho_path]
  }
}

#[cfg(test)]
mod tests {
  use stochastic_rs_core::simd_rng::Deterministic;

  use super::*;

  #[test]
  fn produces_valid_paths() {
    // Parameters from Table 2 in the paper
    let model = HestonStochCorr::new(
      0.0_f64,                // r
      100.0,                  // s0
      0.02,                   // v0
      2.1,                    // kappa_v
      0.03,                   // mu_v
      0.2,                    // sigma_v
      -0.4,                   // rho0
      3.4,                    // kappa_r
      -0.6,                   // mu_r
      0.1,                    // sigma_r
      0.4,                    // rho2
      500,                    // n
      Some(1.0),              // t
      Deterministic::new(42), // seed
    );
    let [s, v, rho] = model.sample();
    assert_eq!(s.len(), 500);
    assert!(
      s.iter().all(|x| x.is_finite() && *x > 0.0),
      "prices must be positive"
    );
    assert!(
      v.iter().all(|x| x.is_finite() && *x >= 0.0),
      "variance must be non-negative"
    );
    assert!(
      rho.iter().all(|&r| r > -1.0 && r < 1.0),
      "correlation must be in (-1,1)"
    );
  }

  #[test]
  fn seeded_is_deterministic() {
    let mk = || {
      HestonStochCorr::new(
        0.03_f64,
        100.0,
        0.04,
        2.0,
        0.04,
        0.3,
        -0.7,
        5.0,
        -0.5,
        0.2,
        0.3,
        200,
        Some(0.5),
        Deterministic::new(99),
      )
    };
    let [s1, _, _] = mk().sample();
    let [s2, _, _] = mk().sample();
    for i in 0..200 {
      assert!((s1[i] - s2[i]).abs() < 1e-14, "paths diverged at i={i}");
    }
  }

  #[test]
  fn constant_corr_when_sigma_r_zero() {
    let model = HestonStochCorr::new(
      0.0_f64,
      100.0,
      0.04,
      2.0,
      0.04,
      0.3,
      -0.7,
      5.0,
      -0.7,
      1e-15,
      0.0,
      300,
      Some(1.0),
      Deterministic::new(55),
    );
    let [_, _, rho] = model.sample();
    // With σ_r ≈ 0, correlation should stay near ρ₀ = -0.7
    for &r in rho.iter() {
      assert!(
        (r - (-0.7)).abs() < 0.05,
        "expected constant correlation near -0.7, got {r}"
      );
    }
  }
}
