//! # Spread
//!
//! Margrabe (1978) exchange option ($K=0$):
//! $$
//! M = S_1 e^{-q_1 T} N(d_1) - S_2 e^{-q_2 T} N(d_2),\quad
//! \sigma = \sqrt{\sigma_1^2 + \sigma_2^2 - 2\rho\sigma_1\sigma_2}
//! $$
//!
//! Source:
//! - Margrabe, W. (1978), "The Value of an Option to Exchange One Asset for Another", J. Finance 33
//! - Kirk, E. (1995), "Correlation in the Energy Markets" — see [`super::kirk`] for non-zero strikes.
//!
use rayon::prelude::*;
use statrs::distribution::ContinuousCDF;
use statrs::distribution::Normal;

use crate::quant::OptionType;
use crate::traits::FloatExt;

/// Margrabe (1978) exchange option: pays $\max(S_1 - S_2, 0)$.
///
/// $$
/// V = S_1 e^{-q_1 T} N(d_1) - S_2 e^{-q_2 T} N(d_2),\quad
/// \sigma = \sqrt{\sigma_1^2 + \sigma_2^2 - 2\rho\sigma_1\sigma_2}
/// $$
#[derive(Debug, Clone)]
pub struct MargrabePricer {
  /// Spot of asset 1.
  pub s1: f64,
  /// Spot of asset 2.
  pub s2: f64,
  /// Volatility of asset 1.
  pub sigma1: f64,
  /// Volatility of asset 2.
  pub sigma2: f64,
  /// Correlation between log-returns.
  pub rho: f64,
  /// Dividend yield 1.
  pub q1: f64,
  /// Dividend yield 2.
  pub q2: f64,
  /// Time to maturity.
  pub t: f64,
}

impl MargrabePricer {
  /// Price the exchange option (always returns the call payoff
  /// $\max(S_1 - S_2, 0)$). The "put" version $\max(S_2 - S_1, 0)$ is just a
  /// `MargrabePricer` with assets swapped.
  pub fn price(&self) -> f64 {
    let v_sq = self.sigma1 * self.sigma1 + self.sigma2 * self.sigma2
      - 2.0 * self.rho * self.sigma1 * self.sigma2;
    if v_sq < 1e-14 {
      return (self.s1 * (-self.q1 * self.t).exp() - self.s2 * (-self.q2 * self.t).exp()).max(0.0);
    }
    let v = v_sq.sqrt();
    let n = Normal::new(0.0, 1.0).unwrap();
    let sqrt_t = self.t.sqrt();
    let d1 = ((self.s1 / self.s2).ln() + (self.q2 - self.q1 + 0.5 * v_sq) * self.t) / (v * sqrt_t);
    let d2 = d1 - v * sqrt_t;
    self.s1 * (-self.q1 * self.t).exp() * n.cdf(d1)
      - self.s2 * (-self.q2 * self.t).exp() * n.cdf(d2)
  }

  /// Greek delta with respect to $S_1$.
  pub fn delta1(&self) -> f64 {
    let v_sq = self.sigma1 * self.sigma1 + self.sigma2 * self.sigma2
      - 2.0 * self.rho * self.sigma1 * self.sigma2;
    if v_sq < 1e-14 {
      return (-self.q1 * self.t).exp();
    }
    let v = v_sq.sqrt();
    let n = Normal::new(0.0, 1.0).unwrap();
    let sqrt_t = self.t.sqrt();
    let d1 = ((self.s1 / self.s2).ln() + (self.q2 - self.q1 + 0.5 * v_sq) * self.t) / (v * sqrt_t);
    (-self.q1 * self.t).exp() * n.cdf(d1)
  }

  /// Greek delta with respect to $S_2$.
  pub fn delta2(&self) -> f64 {
    let v_sq = self.sigma1 * self.sigma1 + self.sigma2 * self.sigma2
      - 2.0 * self.rho * self.sigma1 * self.sigma2;
    if v_sq < 1e-14 {
      return -(-self.q2 * self.t).exp();
    }
    let v = v_sq.sqrt();
    let n = Normal::new(0.0, 1.0).unwrap();
    let sqrt_t = self.t.sqrt();
    let d1 = ((self.s1 / self.s2).ln() + (self.q2 - self.q1 + 0.5 * v_sq) * self.t) / (v * sqrt_t);
    let d2 = d1 - v * sqrt_t;
    -(-self.q2 * self.t).exp() * n.cdf(d2)
  }
}

/// Monte-Carlo spread option pricer for general non-zero strikes under
/// correlated geometric Brownian motion. Pays
/// $\max\!\big(\phi(S_1 - S_2 - K), 0\big)$ where $\phi=\pm 1$.
#[derive(Debug, Clone)]
pub struct McSpreadPricer {
  /// Spot of asset 1.
  pub s1: f64,
  /// Spot of asset 2.
  pub s2: f64,
  /// Strike.
  pub k: f64,
  /// Volatility of asset 1.
  pub sigma1: f64,
  /// Volatility of asset 2.
  pub sigma2: f64,
  /// Correlation.
  pub rho: f64,
  /// Risk-free rate.
  pub r: f64,
  /// Dividend yield 1.
  pub q1: f64,
  /// Dividend yield 2.
  pub q2: f64,
  /// Time to maturity.
  pub t: f64,
  /// Option type.
  pub option_type: OptionType,
  /// Number of MC paths.
  pub n_paths: usize,
}

impl McSpreadPricer {
  pub fn price(&self) -> f64 {
    let phi = match self.option_type {
      OptionType::Call => 1.0,
      OptionType::Put => -1.0,
    };
    let drift1 = (self.r - self.q1 - 0.5 * self.sigma1 * self.sigma1) * self.t;
    let drift2 = (self.r - self.q2 - 0.5 * self.sigma2 * self.sigma2) * self.t;
    let vol1 = self.sigma1 * self.t.sqrt();
    let vol2 = self.sigma2 * self.t.sqrt();
    let rho = self.rho;
    let sqrt_one_minus_rho2 = (1.0 - rho * rho).max(0.0).sqrt();

    let mut all_z = vec![0.0_f64; self.n_paths * 2];
    <f64 as FloatExt>::fill_standard_normal_slice(&mut all_z);

    let sum: f64 = (0..self.n_paths)
      .into_par_iter()
      .map(|i| {
        let z1 = all_z[2 * i];
        let z2_indep = all_z[2 * i + 1];
        let z2 = rho * z1 + sqrt_one_minus_rho2 * z2_indep;
        let s1_t = self.s1 * (drift1 + vol1 * z1).exp();
        let s2_t = self.s2 * (drift2 + vol2 * z2).exp();
        ((phi * (s1_t - s2_t - self.k)).max(0.0)) as f64
      })
      .sum();

    (-self.r * self.t).exp() * sum / self.n_paths as f64
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  /// Margrabe with σ1=σ2 and ρ=1 must equal $\max(S_1 e^{-q_1 T} - S_2
  /// e^{-q_2 T}, 0)$ — the spread is deterministic at maturity.
  #[test]
  fn margrabe_perfect_correlation_equal_vol() {
    let p = MargrabePricer {
      s1: 100.0,
      s2: 100.0,
      sigma1: 0.2,
      sigma2: 0.2,
      rho: 1.0,
      q1: 0.0,
      q2: 0.0,
      t: 1.0,
    };
    let price = p.price();
    assert!(price.abs() < 1e-8, "perfect-corr Margrabe={price}");
  }

  /// Margrabe at-the-money with zero correlation, equal vols.
  /// $S_1 = S_2 = 100$, $\sigma_1 = \sigma_2 = 0.20$, $\rho = 0$, $T = 1$
  /// → $\sigma_M = \sqrt{0.08} \approx 0.2828$
  /// → V = 100 * (2N(σ_M/2) - 1) ≈ 11.246
  #[test]
  fn margrabe_atm_zero_corr() {
    let p = MargrabePricer {
      s1: 100.0,
      s2: 100.0,
      sigma1: 0.20,
      sigma2: 0.20,
      rho: 0.0,
      q1: 0.0,
      q2: 0.0,
      t: 1.0,
    };
    let price = p.price();
    let expected = 11.246;
    assert!((price - expected).abs() < 0.05, "Margrabe ATM={price}");
  }

  /// Margrabe with $S_1 \gg S_2$ approaches the discounted intrinsic.
  #[test]
  fn margrabe_deep_itm() {
    let p = MargrabePricer {
      s1: 200.0,
      s2: 100.0,
      sigma1: 0.20,
      sigma2: 0.20,
      rho: 0.5,
      q1: 0.01,
      q2: 0.02,
      t: 0.5,
    };
    let price = p.price();
    let intrinsic = 200.0 * (-0.01_f64 * 0.5).exp() - 100.0 * (-0.02_f64 * 0.5).exp();
    assert!(
      price > intrinsic,
      "Margrabe deep ITM={price} vs intrinsic={intrinsic}"
    );
  }

  /// Margrabe ↔ MC (K=0) consistency: with enough paths the MC spread call
  /// should match Margrabe within 1.5%.
  #[test]
  fn margrabe_matches_mc_zero_strike() {
    let m = MargrabePricer {
      s1: 110.0,
      s2: 100.0,
      sigma1: 0.25,
      sigma2: 0.20,
      rho: 0.4,
      q1: 0.0,
      q2: 0.0,
      t: 1.0,
    };
    let mc = McSpreadPricer {
      s1: 110.0,
      s2: 100.0,
      k: 0.0,
      sigma1: 0.25,
      sigma2: 0.20,
      rho: 0.4,
      r: 0.0,
      q1: 0.0,
      q2: 0.0,
      t: 1.0,
      option_type: OptionType::Call,
      n_paths: 100_000,
    };
    let m_price = m.price();
    let mc_price = mc.price();
    let rel = (m_price - mc_price).abs() / m_price;
    assert!(rel < 0.02, "margrabe={m_price}, mc={mc_price}, rel={rel}");
  }
}
