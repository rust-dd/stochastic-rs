//! # VarianceSwap
//!
//! $$
//! K_{\text{var}}=\frac{2}{T}\left[rT-\frac{S_0 e^{rT}}{K_0}-1-\ln\frac{K_0}{S_0}
//! +e^{rT}\!\int_0^{K_0}\!\frac{P(K)}{K^2}dK+e^{rT}\!\int_{K_0}^\infty\!\frac{C(K)}{K^2}dK\right]
//! $$
//!
//! Source:
//! - Demeterfi, K., Derman, E., Kamal, M. & Zou, J. (1999),
//!   "More Than You Ever Wanted to Know About Volatility Swaps"
//!

/// Variance swap pricing utilities.
pub struct VarianceSwapPricer {
  /// Spot price.
  pub s: f64,
  /// Risk-free rate.
  pub r: f64,
  /// Dividend yield.
  pub q: f64,
  /// Time to maturity in years.
  pub t: f64,
}

impl VarianceSwapPricer {
  /// Fair variance strike under BSM ($K_{\text{var}}=\sigma^2$).
  pub fn fair_strike_bsm(&self, sigma: f64) -> f64 {
    sigma * sigma
  }

  /// Fair variance strike from a discrete strip of OTM option prices
  /// (static replication method).
  ///
  /// Inputs must be sorted by strike. Puts for `K < s`, calls for `K >= s`.
  pub fn fair_strike_replication(&self, strikes: &[f64], prices: &[f64]) -> f64 {
    assert_eq!(strikes.len(), prices.len());
    let n = strikes.len();
    if n < 2 {
      return 0.0;
    }

    let fwd = self.s * ((self.r - self.q) * self.t).exp();
    let disc = (self.r * self.t).exp();

    // Find ATM index (strike closest to forward)
    let k0_idx = strikes
      .iter()
      .enumerate()
      .min_by(|(_, a), (_, b)| (*a - fwd).abs().partial_cmp(&(*b - fwd).abs()).unwrap())
      .map(|(i, _)| i)
      .unwrap_or(0);
    let k0 = strikes[k0_idx];

    // Trapezoidal integration of OTM option prices / K^2
    let mut integral = 0.0;
    for i in 0..n {
      let dk = if i == 0 {
        strikes[1] - strikes[0]
      } else if i == n - 1 {
        strikes[n - 1] - strikes[n - 2]
      } else {
        0.5 * (strikes[i + 1] - strikes[i - 1])
      };
      integral += dk * prices[i] / (strikes[i] * strikes[i]);
    }

    let fair = (2.0 / self.t) * (disc * integral - ((fwd / k0) - 1.0) - (fwd / k0).ln());

    fair.max(0.0)
  }

  /// Realised variance from a price path.
  ///
  /// $$
  /// \hat\sigma^2=\frac{1}{N\Delta t}\sum_{i=1}^N\!\left(\ln\frac{S_i}{S_{i-1}}\right)^2
  /// $$
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

  /// P&L of a variance swap.
  pub fn pnl(realized_var: f64, fair_strike: f64, notional: f64) -> f64 {
    notional * (realized_var - fair_strike)
  }
}

/// Volatility swap pricer (approximate).
///
/// $$
/// K_{\text{vol}}\approx\sqrt{K_{\text{var}}}-\frac{\text{Var}(V)}{8\,K_{\text{var}}^{3/2}}
/// $$
pub struct VolatilitySwapPricer;

impl VolatilitySwapPricer {
  /// Approximate fair volatility strike under BSM.
  pub fn fair_strike_bsm(sigma: f64) -> f64 {
    sigma
  }

  /// Approximate fair vol strike from fair variance strike with convexity
  /// adjustment.
  pub fn fair_strike_from_var(k_var: f64, var_of_var: f64) -> f64 {
    if k_var <= 0.0 {
      return 0.0;
    }
    k_var.sqrt() - var_of_var / (8.0 * k_var.powf(1.5))
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn bsm_fair_strike() {
    let p = VarianceSwapPricer {
      s: 100.0,
      r: 0.05,
      q: 0.0,
      t: 1.0,
    };
    let k = p.fair_strike_bsm(0.2);
    assert!((k - 0.04).abs() < 1e-10);
  }

  #[test]
  fn realized_variance_constant() {
    let prices = vec![100.0; 10];
    let rv = VarianceSwapPricer::realized_variance(&prices, 1.0 / 252.0);
    assert!((rv - 0.0).abs() < 1e-15);
  }

  #[test]
  fn realized_variance_known_path() {
    // Geometric growth at 20% annualized over 252 days
    let dt: f64 = 1.0 / 252.0;
    let daily_return = 0.20 * dt.sqrt();
    let prices: Vec<f64> = (0..253)
      .map(|i| 100.0 * (daily_return * i as f64).exp())
      .collect();
    let rv = VarianceSwapPricer::realized_variance(&prices, dt);
    // Should be close to 0.04 (=0.2^2), but log-return squared sum approach
    // gives approximately sigma^2
    assert!((rv - 0.04).abs() < 0.005, "rv={rv}, expected≈0.04");
  }

  #[test]
  fn pnl_positive_when_realized_exceeds_strike() {
    let pnl = VarianceSwapPricer::pnl(0.06, 0.04, 100_000.0);
    assert!(pnl > 0.0);
    assert!((pnl - 2000.0).abs() < 1e-10);
  }

  #[test]
  fn vol_swap_convexity() {
    let k_var = 0.04;
    let k_vol = VolatilitySwapPricer::fair_strike_from_var(k_var, 0.001);
    assert!(k_vol < k_var.sqrt());
    assert!(k_vol > 0.0);
  }

  #[test]
  fn vol_swap_bsm_equals_sigma() {
    let k_vol = VolatilitySwapPricer::fair_strike_bsm(0.25);
    assert!((k_vol - 0.25).abs() < 1e-15);
  }

  #[test]
  fn vol_swap_zero_var_of_var() {
    // With zero var-of-var, vol strike = sqrt(var strike)
    let k_var = 0.04;
    let k_vol = VolatilitySwapPricer::fair_strike_from_var(k_var, 0.0);
    assert!((k_vol - 0.2).abs() < 1e-10);
  }
}
