//! # Lookback
//!
//! $$
//! c_{\text{float}}=Se^{-qT}N(a_1)-S_{\min}e^{-rT}N(a_1-\sigma\sqrt T)
//! +\frac{S\sigma^2}{2b}e^{-rT}\!\left[\left(\frac{S}{S_{\min}}\right)^{-2b/\sigma^2}
//! N\!\left(-a_1+\frac{2b\sqrt T}{\sigma}\right)-e^{bT}N(-a_1)\right]
//! $$
//!
//! where $b=r-q$ is the cost of carry.
//!
//! Source:
//! - Goldman, M.B., Sosin, H.B. & Gatto, M.A. (1979), "Path Dependent Options: Buy at the Low, Sell at the High"
//! - Conze, A. & Viswanathan, R. (1991), "Path Dependent Options: The Case of Lookback Options"
//! - Haug, E.G. (2007), "The Complete Guide to Option Pricing Formulas", 2nd ed.
//!
use statrs::distribution::ContinuousCDF;
use statrs::distribution::Normal;

use crate::OptionType;

/// Floating-strike lookback option pricer (Goldman–Sosin–Gatto 1979).
///
/// A floating-strike lookback call gives the right to buy at the observed
/// minimum; a put gives the right to sell at the observed maximum.
#[derive(Debug, Clone)]
pub struct FloatingLookbackPricer {
  /// Current underlying price.
  pub s: f64,
  /// Observed minimum price (for calls; defaults to `s`).
  pub s_min: Option<f64>,
  /// Observed maximum price (for puts; defaults to `s`).
  pub s_max: Option<f64>,
  /// Risk-free rate.
  pub r: f64,
  /// Dividend yield.
  pub q: f64,
  /// Volatility.
  pub sigma: f64,
  /// Time to maturity in years.
  pub t: f64,
  /// Option type.
  pub option_type: OptionType,
}

impl FloatingLookbackPricer {
  pub fn price(&self) -> f64 {
    let n = Normal::new(0.0, 1.0).unwrap();
    let s = self.s;
    let r = self.r;
    let q = self.q;
    let b = r - q;
    let sigma = self.sigma;
    let t = self.t;
    let sqrt_t = t.sqrt();
    let sigma2 = sigma * sigma;

    match self.option_type {
      OptionType::Call => {
        let s_min = self.s_min.unwrap_or(s);
        let a1 = ((s / s_min).ln() + (b + 0.5 * sigma2) * t) / (sigma * sqrt_t);

        let term1 = s * (-q * t).exp() * n.cdf(a1);
        let term2 = s_min * (-r * t).exp() * n.cdf(a1 - sigma * sqrt_t);

        let premium = if b.abs() < 1e-10 {
          s * (-r * t).exp() * sigma * sqrt_t * n.cdf(a1).max(0.0) * sigma
        } else {
          let coeff = s * (-r * t).exp() * sigma2 / (2.0 * b);
          let bracket = (s / s_min).powf(-2.0 * b / sigma2) * n.cdf(-a1 + 2.0 * b * sqrt_t / sigma)
            - (b * t).exp() * n.cdf(-a1);
          coeff * bracket
        };

        term1 - term2 + premium
      }
      OptionType::Put => {
        let s_max = self.s_max.unwrap_or(s);
        let b1 = ((s / s_max).ln() + (b + 0.5 * sigma2) * t) / (sigma * sqrt_t);

        let term1 = s_max * (-r * t).exp() * n.cdf(-b1 + sigma * sqrt_t);
        let term2 = s * (-q * t).exp() * n.cdf(-b1);

        let premium = if b.abs() < 1e-10 {
          s * (-r * t).exp() * sigma * sqrt_t * n.cdf(-b1).max(0.0) * sigma
        } else {
          let coeff = s * (-r * t).exp() * sigma2 / (2.0 * b);
          let bracket = (b * t).exp() * n.cdf(b1)
            - (s / s_max).powf(-2.0 * b / sigma2) * n.cdf(b1 - 2.0 * b * sqrt_t / sigma);
          coeff * bracket
        };

        term1 - term2 + premium
      }
    }
  }
}

/// Fixed-strike lookback option pricer (Conze–Viswanathan 1991).
///
/// A fixed-strike lookback call pays $\max(S_{\max}-K,0)$;
/// a put pays $\max(K-S_{\min},0)$.
#[derive(Debug, Clone)]
pub struct FixedLookbackPricer {
  /// Current underlying price.
  pub s: f64,
  /// Strike price.
  pub k: f64,
  /// Observed minimum price (for puts).
  pub s_min: Option<f64>,
  /// Observed maximum price (for calls).
  pub s_max: Option<f64>,
  /// Risk-free rate.
  pub r: f64,
  /// Dividend yield.
  pub q: f64,
  /// Volatility.
  pub sigma: f64,
  /// Time to maturity in years.
  pub t: f64,
  /// Option type.
  pub option_type: OptionType,
}

impl FixedLookbackPricer {
  /// Price using the identity:
  /// - Call: `floating_put(S, m) + S·e^{-qT} - K·e^{-rT}` where `m = max(S_max, K)`
  /// - Put:  `floating_call(S, m) + K·e^{-rT} - S·e^{-qT}` where `m = min(S_min, K)`
  pub fn price(&self) -> f64 {
    match self.option_type {
      OptionType::Call => {
        let s_max = self.s_max.unwrap_or(self.s);
        let m = s_max.max(self.k);
        let fp = FloatingLookbackPricer {
          s: self.s,
          s_min: None,
          s_max: Some(m),
          r: self.r,
          q: self.q,
          sigma: self.sigma,
          t: self.t,
          option_type: OptionType::Put,
        };
        fp.price() + self.s * (-self.q * self.t).exp() - self.k * (-self.r * self.t).exp()
      }
      OptionType::Put => {
        let s_min = self.s_min.unwrap_or(self.s);
        let m = s_min.min(self.k);
        let fc = FloatingLookbackPricer {
          s: self.s,
          s_min: Some(m),
          s_max: None,
          r: self.r,
          q: self.q,
          sigma: self.sigma,
          t: self.t,
          option_type: OptionType::Call,
        };
        fc.price() + self.k * (-self.r * self.t).exp() - self.s * (-self.q * self.t).exp()
      }
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn floating_call_intrinsic_bound() {
    // Floating lookback call ≥ discounted intrinsic value (S - S_min)
    let p = FloatingLookbackPricer {
      s: 100.0,
      s_min: Some(90.0),
      s_max: None,
      r: 0.05,
      q: 0.0,
      sigma: 0.2,
      t: 1.0,
      option_type: OptionType::Call,
    };
    let price = p.price();
    let intrinsic = (100.0 - 90.0) * (-0.05_f64).exp();
    assert!(price > 0.0);
    assert!(price >= intrinsic, "price={price} < intrinsic={intrinsic}");
  }

  #[test]
  fn floating_call_goldman_sosin_gatto() {
    // S=100, S_min=100, r=0.10, σ=0.10, T=0.5 → 8.2687 (Haug formula)
    let p = FloatingLookbackPricer {
      s: 100.0,
      s_min: Some(100.0),
      s_max: None,
      r: 0.10,
      q: 0.0,
      sigma: 0.10,
      t: 0.5,
      option_type: OptionType::Call,
    };
    let price = p.price();
    assert!(
      (price - 8.2687).abs() < 0.01,
      "floating call={price}, expected≈8.2687"
    );
  }

  #[test]
  fn floating_put_positive() {
    let p = FloatingLookbackPricer {
      s: 100.0,
      s_min: None,
      s_max: Some(110.0),
      r: 0.05,
      q: 0.0,
      sigma: 0.2,
      t: 1.0,
      option_type: OptionType::Put,
    };
    let price = p.price();
    assert!(price > 0.0, "floating lookback put={price}");
    // Put must be ≥ discounted intrinsic (S_max - S)
    let intrinsic = (110.0 - 100.0) * (-0.05_f64).exp();
    assert!(price >= intrinsic, "price={price} < intrinsic={intrinsic}");
  }

  #[test]
  fn floating_call_more_volatile_is_more_expensive() {
    let base = FloatingLookbackPricer {
      s: 100.0,
      s_min: Some(100.0),
      s_max: None,
      r: 0.05,
      q: 0.0,
      sigma: 0.2,
      t: 1.0,
      option_type: OptionType::Call,
    };
    let high_vol = FloatingLookbackPricer { sigma: 0.4, ..base };
    assert!(high_vol.price() > base.price());
  }

  #[test]
  fn fixed_call_geq_vanilla() {
    // Fixed-strike lookback call ≥ vanilla call (since payoff = max(S_max - K, 0) ≥ max(S_T - K, 0))
    let p = FixedLookbackPricer {
      s: 100.0,
      k: 100.0,
      s_min: None,
      s_max: Some(100.0),
      r: 0.05,
      q: 0.0,
      sigma: 0.2,
      t: 1.0,
      option_type: OptionType::Call,
    };
    let price = p.price();
    // BSM vanilla call ≈ 10.45
    assert!(
      price > 10.0,
      "fixed lookback call={price}, should be > vanilla≈10.45"
    );
  }

  #[test]
  fn fixed_put_geq_vanilla() {
    let p = FixedLookbackPricer {
      s: 100.0,
      k: 100.0,
      s_min: Some(100.0),
      s_max: None,
      r: 0.05,
      q: 0.0,
      sigma: 0.2,
      t: 1.0,
      option_type: OptionType::Put,
    };
    let price = p.price();
    // BSM vanilla put ≈ 5.57
    assert!(
      price > 5.0,
      "fixed lookback put={price}, should be > vanilla≈5.57"
    );
  }
}
