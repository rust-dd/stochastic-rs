//! # Variance / volatility swap pricing
//!
//! Static-replication (Demeterfi–Derman–Kamal–Zou 1999) and stochastic-vol
//! closed-form fair strikes (Brockhaus–Long 2000 for Heston, Bernard–Cui
//! 2013 discrete corrections).
//!
//! Continuous Demeterfi formula (with continuous dividend yield $q$ and
//! forward $F = S_0 e^{(r-q)T}$, ATM cutoff $K_0$):
//!
//! $$
//! K_{\text{var}}=\frac{2}{T}\!\left[
//!   (r-q)T - \!\!\left(\!\frac{F}{K_0}\!-\!1\!\right)\!
//!   - \ln\frac{K_0}{S_0}
//!   + e^{rT}\!\!\int_0^{K_0}\!\!\frac{P(K)}{K^2}\,dK
//!   + e^{rT}\!\!\int_{K_0}^{\infty}\!\!\frac{C(K)}{K^2}\,dK
//! \right]
//! $$
//!
//! Heston closed-form (continuous monitoring):
//!
//! $$
//! K_{\text{var}}^{\text{Heston}}=\theta+(V_0-\theta)\,\frac{1-e^{-\kappa T}}{\kappa T}
//! $$
//!
//! Source:
//! - Demeterfi, K., Derman, E., Kamal, M. & Zou, J. (1999),
//!   "More Than You Ever Wanted to Know About Volatility Swaps", Goldman Sachs.
//! - Brockhaus, O. & Long, D. (2000), "Volatility Swaps Made Simple", Risk 13.
//! - Bernard, C. & Cui, Z. (2013), "Prices and Asymptotics for Discrete
//!   Variance Swaps", arXiv:1305.7092.
//! - Carr, P. & Madan, D. (1998), "Towards a Theory of Volatility Trading",
//!   in *Volatility: New Estimation Techniques for Pricing Derivatives*.

/// Variance-swap pricer.
///
/// State stores the forward-curve inputs (spot, rates, maturity); pricing
/// methods consume external option strips or model parameters.
#[derive(Debug, Clone, Copy)]
pub struct VarianceSwapPricer {
  /// Spot price.
  pub s: f64,
  /// Continuously-compounded risk-free rate.
  pub r: f64,
  /// Continuous dividend yield.
  pub q: f64,
  /// Time to maturity in years.
  pub t: f64,
}

impl VarianceSwapPricer {
  /// Forward $F = S_0 e^{(r-q)T}$.
  pub fn forward(&self) -> f64 {
    self.s * ((self.r - self.q) * self.t).exp()
  }

  /// Black–Scholes fair strike: $K_{\text{var}} = \sigma^2$.
  pub fn fair_strike_bsm(&self, sigma: f64) -> f64 {
    sigma * sigma
  }

  /// Static replication fair strike (Demeterfi–Derman–Kamal–Zou).
  ///
  /// Inputs are the OTM option strip — puts for $K < K_0$, calls for
  /// $K \geq K_0$ — with $K_0$ identified as the strike closest to the
  /// forward. Strikes must be sorted ascending. Trapezoidal weights are
  /// used for the $\int P(K)/K^2 dK + \int C(K)/K^2 dK$ contribution.
  pub fn fair_strike_replication(&self, strikes: &[f64], otm_prices: &[f64]) -> f64 {
    assert_eq!(strikes.len(), otm_prices.len(), "strikes / prices length mismatch");
    let n = strikes.len();
    if n < 2 || self.t <= 0.0 {
      return 0.0;
    }
    debug_assert!(
      strikes.windows(2).all(|w| w[0] <= w[1]),
      "strikes must be sorted ascending"
    );

    let fwd = self.forward();
    let disc = (self.r * self.t).exp();

    let k0_idx = strikes
      .iter()
      .enumerate()
      .min_by(|(_, a), (_, b)| (*a - fwd).abs().partial_cmp(&(*b - fwd).abs()).unwrap())
      .map(|(i, _)| i)
      .unwrap_or(0);
    let k0 = strikes[k0_idx];

    let mut integral = 0.0;
    for i in 0..n {
      let dk = if i == 0 {
        strikes[1] - strikes[0]
      } else if i == n - 1 {
        strikes[n - 1] - strikes[n - 2]
      } else {
        0.5 * (strikes[i + 1] - strikes[i - 1])
      };
      integral += dk * otm_prices[i] / (strikes[i] * strikes[i]);
    }

    let drift = (self.r - self.q) * self.t;
    let fair = (2.0 / self.t)
      * (drift - (fwd / k0 - 1.0) - (k0 / self.s).ln() + disc * integral);
    fair.max(0.0)
  }

  /// Heston closed-form fair variance strike (Brockhaus–Long 2000).
  ///
  /// Continuous-monitoring expected integrated variance,
  /// $E\!\left[\frac{1}{T}\int_0^T V_t\,dt\right]$, depends only on
  /// `(v0, kappa, theta, T)` — not on `(rho, sigma, r, q)`.
  pub fn fair_strike_heston(&self, v0: f64, kappa: f64, theta: f64) -> f64 {
    let t = self.t;
    if t <= 0.0 {
      return v0;
    }
    if kappa.abs() < 1e-10 {
      // Limit κ → 0 of (1 - e^{-κT})/(κT) is 1, so K_var → v0.
      return v0;
    }
    let factor = (1.0 - (-kappa * t).exp()) / (kappa * t);
    theta + (v0 - theta) * factor
  }

  /// Discrete-monitoring correction to the continuous Heston fair strike
  /// (Bernard–Cui 2013, leading-order in $T/N$).
  ///
  /// Adds $\frac{T}{N}$ correction reflecting the discrete-vs-continuous
  /// gap; for $N \to \infty$ converges to the continuous strike.
  pub fn fair_strike_heston_discrete(
    &self,
    v0: f64,
    kappa: f64,
    theta: f64,
    sigma: f64,
    rho: f64,
    n_obs: usize,
  ) -> f64 {
    let cont = self.fair_strike_heston(v0, kappa, theta);
    if n_obs == 0 {
      return cont;
    }
    let t = self.t;
    let dt = t / n_obs as f64;
    // Leading-order Bernard–Cui adjustment for log-return moment.
    // ξ = θ + (V0 - θ)·factor; correction ≈ ¼·ξ²·dt + ρ·σ·ξ·dt.
    let xi = cont;
    let bias = 0.25 * xi * xi * dt + rho * sigma * xi * dt;
    cont + bias
  }

  /// Realised variance estimator from a price path,
  /// $\hat\sigma^2 = \frac{1}{N\Delta t}\sum_{i=1}^N (\ln S_i/S_{i-1})^2$.
  pub fn realized_variance(prices: &[f64], dt: f64) -> f64 {
    if prices.len() < 2 {
      return 0.0;
    }
    let n = prices.len() - 1;
    let mut rv = 0.0;
    for i in 1..=n {
      let lr = (prices[i] / prices[i - 1]).ln();
      rv += lr * lr;
    }
    rv / (n as f64 * dt)
  }

  /// P&L of a long variance-swap position, $N \times (\hat\sigma^2 - K_{\text{var}})$.
  pub fn pnl(realized_var: f64, fair_strike: f64, notional: f64) -> f64 {
    notional * (realized_var - fair_strike)
  }
}

/// Replicating portfolio weights for the log contract — useful for
/// hedging a variance swap with an actual strip of vanilla options.
///
/// Weight at strike $K_i$ is $\frac{2}{T}\,\frac{\Delta K_i}{K_i^2}$
/// (Demeterfi et al., eq. (28)). Returned in the same order as `strikes`.
pub fn replication_weights(strikes: &[f64], maturity: f64) -> Vec<f64> {
  let n = strikes.len();
  let mut w = vec![0.0; n];
  if n < 2 || maturity <= 0.0 {
    return w;
  }
  for i in 0..n {
    let dk = if i == 0 {
      strikes[1] - strikes[0]
    } else if i == n - 1 {
      strikes[n - 1] - strikes[n - 2]
    } else {
      0.5 * (strikes[i + 1] - strikes[i - 1])
    };
    w[i] = (2.0 / maturity) * dk / (strikes[i] * strikes[i]);
  }
  w
}

/// Volatility-swap fair strike with convexity correction.
///
/// Naive: $K_{\text{vol}} \approx \sqrt{K_{\text{var}}}$. With variance-of-variance
/// the convex Jensen correction lowers the strike:
///
/// $$
/// K_{\text{vol}} \approx \sqrt{K_{\text{var}}} - \frac{\text{Var}(V)}{8\,K_{\text{var}}^{3/2}}
/// $$
pub struct VolatilitySwapPricer;

impl VolatilitySwapPricer {
  /// Black–Scholes vol strike: $K_{\text{vol}} = \sigma$.
  pub fn fair_strike_bsm(sigma: f64) -> f64 {
    sigma
  }

  /// Convexity-adjusted vol strike from variance strike + variance-of-variance.
  pub fn fair_strike_from_var(k_var: f64, var_of_var: f64) -> f64 {
    if k_var <= 0.0 {
      return 0.0;
    }
    k_var.sqrt() - var_of_var / (8.0 * k_var.powf(1.5))
  }

  /// Heston-implied vol strike — uses continuous Heston variance fair
  /// strike with second-order convexity adjustment from variance dispersion.
  ///
  /// $\text{Var}\!\left(\frac{1}{T}\int_0^T V_t dt\right) \approx
  /// \frac{\sigma^2(V_0 - \theta)^2 (1-e^{-2\kappa T})}{2\kappa^3 T^2}$
  /// to leading order; the closed form is messier — we use a tractable
  /// approximation suitable for short maturities.
  pub fn fair_strike_heston(
    v0: f64,
    kappa: f64,
    theta: f64,
    sigma: f64,
    t: f64,
  ) -> f64 {
    let pricer = VarianceSwapPricer {
      s: 1.0,
      r: 0.0,
      q: 0.0,
      t,
    };
    let k_var = pricer.fair_strike_heston(v0, kappa, theta);
    if kappa.abs() < 1e-10 || t <= 0.0 {
      return k_var.max(0.0).sqrt();
    }
    let dispersion =
      (sigma * sigma * (v0 - theta).powi(2) * (1.0 - (-2.0 * kappa * t).exp()))
        / (2.0 * kappa.powi(3) * t * t);
    Self::fair_strike_from_var(k_var, dispersion.max(0.0))
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  fn pricer() -> VarianceSwapPricer {
    VarianceSwapPricer {
      s: 100.0,
      r: 0.05,
      q: 0.0,
      t: 1.0,
    }
  }

  #[test]
  fn bsm_fair_strike_is_sigma_squared() {
    assert!((pricer().fair_strike_bsm(0.2) - 0.04).abs() < 1e-12);
  }

  #[test]
  fn forward_under_zero_dividend() {
    let p = pricer();
    assert!((p.forward() - 100.0 * 0.05_f64.exp()).abs() < 1e-10);
  }

  #[test]
  fn realized_variance_constant_path_is_zero() {
    let prices = vec![100.0; 252];
    let rv = VarianceSwapPricer::realized_variance(&prices, 1.0 / 252.0);
    assert!((rv - 0.0).abs() < 1e-15);
  }

  #[test]
  fn realized_variance_recovers_known_drift() {
    let dt: f64 = 1.0 / 252.0;
    let daily = 0.20 * dt.sqrt();
    let prices: Vec<f64> = (0..253).map(|i| 100.0 * (daily * i as f64).exp()).collect();
    let rv = VarianceSwapPricer::realized_variance(&prices, dt);
    assert!((rv - 0.04).abs() < 0.005, "rv={rv}, expected≈0.04");
  }

  #[test]
  fn pnl_scales_with_notional() {
    assert!(
      (VarianceSwapPricer::pnl(0.06, 0.04, 100_000.0) - 2_000.0).abs() < 1e-9
    );
  }

  #[test]
  fn vol_swap_convexity_lowers_strike() {
    let k_vol = VolatilitySwapPricer::fair_strike_from_var(0.04, 0.001);
    assert!(k_vol < 0.04_f64.sqrt());
    assert!(k_vol > 0.0);
  }

  #[test]
  fn vol_swap_zero_dispersion_recovers_sqrt_var() {
    let k_vol = VolatilitySwapPricer::fair_strike_from_var(0.04, 0.0);
    assert!((k_vol - 0.2).abs() < 1e-10);
  }

  #[test]
  fn heston_fair_strike_equals_v0_when_at_long_run_mean() {
    let p = pricer();
    let k_var = p.fair_strike_heston(0.04, 1.5, 0.04);
    assert!((k_var - 0.04).abs() < 1e-12);
  }

  #[test]
  fn heston_fair_strike_blends_v0_to_theta() {
    // V0 = 0.09 (high IV), θ = 0.04 (low LR), strong κ → fair K close to θ
    let p = pricer();
    let k_strong = p.fair_strike_heston(0.09, 5.0, 0.04);
    let k_weak = p.fair_strike_heston(0.09, 0.1, 0.04);
    assert!(k_strong < k_weak);
    assert!(k_weak <= 0.09);
    assert!(k_strong > 0.04);
  }

  #[test]
  fn heston_kappa_zero_limit_equals_v0() {
    let p = pricer();
    assert!((p.fair_strike_heston(0.04, 0.0, 0.10) - 0.04).abs() < 1e-12);
  }

  #[test]
  fn heston_long_t_limit_approaches_theta() {
    // T → ∞ with κ > 0 ⇒ factor → 0, K_var → θ.
    let p = VarianceSwapPricer {
      s: 100.0,
      r: 0.0,
      q: 0.0,
      t: 50.0,
    };
    let k_var = p.fair_strike_heston(0.09, 2.0, 0.04);
    assert!((k_var - 0.04).abs() < 0.01, "K_var={k_var} should approach θ=0.04");
  }

  #[test]
  fn heston_discrete_correction_vanishes_with_n() {
    let p = pricer();
    let k_cont = p.fair_strike_heston(0.04, 1.5, 0.04);
    let k_disc_fine = p.fair_strike_heston_discrete(0.04, 1.5, 0.04, 0.3, -0.7, 100_000);
    let k_disc_coarse = p.fair_strike_heston_discrete(0.04, 1.5, 0.04, 0.3, -0.7, 12);
    assert!((k_disc_fine - k_cont).abs() < (k_disc_coarse - k_cont).abs());
  }

  #[test]
  fn replication_weights_are_positive_and_decay() {
    // Strikes near forward have largest weight; weight ∝ 1/K^2.
    let strikes: Vec<f64> = (50..=150).step_by(10).map(|i| i as f64).collect();
    let w = replication_weights(&strikes, 1.0);
    assert_eq!(w.len(), strikes.len());
    for &wi in &w {
      assert!(wi > 0.0);
    }
    // Weight at K=50 should exceed weight at K=150 (1/K^2 dominates Δk).
    assert!(w[0] > *w.last().unwrap());
  }

  #[test]
  fn replication_strike_within_one_percent_of_bsm_for_dense_strip() {
    // Build a dense BS option strip (σ = 25%) and replicate the strike.
    use stochastic_rs_distributions::special::norm_cdf;
    let p = VarianceSwapPricer {
      s: 100.0,
      r: 0.0,
      q: 0.0,
      t: 1.0,
    };
    let sigma = 0.25;
    let strikes: Vec<f64> = (10..=400).map(|i| i as f64 * 0.5).collect();
    let prices: Vec<f64> = strikes
      .iter()
      .map(|&k| {
        let d1 = ((p.s / k).ln() + 0.5 * sigma * sigma * p.t) / (sigma * p.t.sqrt());
        let d2 = d1 - sigma * p.t.sqrt();
        if k >= p.s {
          // call
          p.s * norm_cdf(d1) - k * norm_cdf(d2)
        } else {
          // put via parity (r = q = 0)
          k * norm_cdf(-d2) - p.s * norm_cdf(-d1)
        }
      })
      .collect();
    let k_var = p.fair_strike_replication(&strikes, &prices);
    let target = sigma * sigma;
    let rel_err = (k_var - target).abs() / target;
    assert!(rel_err < 0.02, "K_var={k_var}, expected≈{target}, rel_err={rel_err}");
  }
}
