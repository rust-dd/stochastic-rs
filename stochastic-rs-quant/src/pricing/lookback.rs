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
use stochastic_rs_distributions::RealExt;
use stochastic_rs_distributions::special::norm_cdf;

use crate::OptionType;
use crate::pricing::exotic::LookbackOption;
use crate::pricing::exotic::assert_sigma;

/// Goldman–Sosin–Gatto (1979) for a floating strike, Conze–Viswanathan (1991) for a fixed one: a
/// floating call buys at the running low, a fixed call pays the running high over the strike.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LookbackPricer {
  /// Annualised Black–Scholes volatility.
  pub sigma: f64,
}

impl LookbackPricer {
  /// Panics unless `sigma` is finite and positive.
  pub fn new(sigma: f64) -> Self {
    assert_sigma(sigma);
    Self { sigma }
  }

  /// Price of `option` at the query `(s, r, q, tau)`; at `tau = 0` its payoff on the running
  /// extremes, and NaN for a negative or NaN `tau`.
  pub fn price(&self, option: &LookbackOption, s: f64, r: f64, q: f64, tau: f64) -> f64 {
    if !(s > 0.0 && tau >= 0.0) || r.is_nan() || q.is_nan() {
      return f64::NAN;
    }
    if tau == 0.0 {
      return Self::payoff(option, s);
    }
    match option.strike {
      None => self.floating(option, s, r, q, tau),
      Some(k) => self.fixed(option, k, s, r, q, tau),
    }
  }

  /// The running minimum and maximum each include the spot.
  fn payoff(option: &LookbackOption, s: f64) -> f64 {
    let low = option.running_min.unwrap_or(s).min_or_nan(s);
    let high = option.running_max.unwrap_or(s).max_or_nan(s);
    match (option.strike, option.option_type) {
      (None, OptionType::Call) => s - low,
      (None, OptionType::Put) => high - s,
      (Some(k), OptionType::Call) => (high - k).max_or_nan(0.0),
      (Some(k), OptionType::Put) => (k - low).max_or_nan(0.0),
    }
  }

  fn floating(&self, option: &LookbackOption, s: f64, r: f64, q: f64, tau: f64) -> f64 {
    let b = r - q;
    let sigma = self.sigma;
    let sqrt_t = tau.sqrt();
    let sigma2 = sigma * sigma;

    match option.option_type {
      OptionType::Call => {
        let s_min = option.running_min.unwrap_or(s);
        let a1 = ((s / s_min).ln() + (b + 0.5 * sigma2) * tau) / (sigma * sqrt_t);

        let term1 = s * (-q * tau).exp() * norm_cdf(a1);
        let term2 = s_min * (-r * tau).exp() * norm_cdf(a1 - sigma * sqrt_t);

        let premium = if b.abs() < 1e-10 {
          s * (-r * tau).exp() * sigma * sqrt_t * norm_cdf(a1).max(0.0) * sigma
        } else {
          let coeff = s * (-r * tau).exp() * sigma2 / (2.0 * b);
          let bracket = (s / s_min).powf(-2.0 * b / sigma2)
            * norm_cdf(-a1 + 2.0 * b * sqrt_t / sigma)
            - (b * tau).exp() * norm_cdf(-a1);
          coeff * bracket
        };

        term1 - term2 + premium
      }
      OptionType::Put => {
        let s_max = option.running_max.unwrap_or(s);
        let b1 = ((s / s_max).ln() + (b + 0.5 * sigma2) * tau) / (sigma * sqrt_t);

        let term1 = s_max * (-r * tau).exp() * norm_cdf(-b1 + sigma * sqrt_t);
        let term2 = s * (-q * tau).exp() * norm_cdf(-b1);

        let premium = if b.abs() < 1e-10 {
          s * (-r * tau).exp() * sigma * sqrt_t * norm_cdf(-b1).max(0.0) * sigma
        } else {
          let coeff = s * (-r * tau).exp() * sigma2 / (2.0 * b);
          let bracket = (b * tau).exp() * norm_cdf(b1)
            - (s / s_max).powf(-2.0 * b / sigma2) * norm_cdf(b1 - 2.0 * b * sqrt_t / sigma);
          coeff * bracket
        };

        term1 - term2 + premium
      }
    }
  }

  /// A fixed call is the floating put at `m = max(S_max, K)` plus `S e^{-qτ} − K e^{-rτ}`, a fixed
  /// put the floating call at `m = min(S_min, K)` plus `K e^{-rτ} − S e^{-qτ}`.
  fn fixed(&self, option: &LookbackOption, k: f64, s: f64, r: f64, q: f64, tau: f64) -> f64 {
    match option.option_type {
      OptionType::Call => {
        let s_max = option.running_max.unwrap_or(s);
        let m = s_max.max(k);
        let floating_put = LookbackOption {
          strike: None,
          running_min: None,
          running_max: Some(m),
          option_type: OptionType::Put,
        };
        self.floating(&floating_put, s, r, q, tau) + s * (-q * tau).exp() - k * (-r * tau).exp()
      }
      OptionType::Put => {
        let s_min = option.running_min.unwrap_or(s);
        let m = s_min.min(k);
        let floating_call = LookbackOption {
          strike: None,
          running_min: Some(m),
          running_max: None,
          option_type: OptionType::Call,
        };
        self.floating(&floating_call, s, r, q, tau) + k * (-r * tau).exp() - s * (-q * tau).exp()
      }
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  fn lookback(
    strike: Option<f64>,
    running_min: Option<f64>,
    running_max: Option<f64>,
    option_type: OptionType,
  ) -> LookbackOption {
    LookbackOption {
      strike,
      running_min,
      running_max,
      option_type,
    }
  }

  #[test]
  fn floating_call_intrinsic_bound() {
    // Floating lookback call ≥ discounted intrinsic value (S - S_min)
    let option = lookback(None, Some(90.0), None, OptionType::Call);
    let price = LookbackPricer::new(0.2).price(&option, 100.0, 0.05, 0.0, 1.0);
    let intrinsic = (100.0 - 90.0) * (-0.05_f64).exp();
    assert!(price > 0.0);
    assert!(price >= intrinsic, "price={price} < intrinsic={intrinsic}");
  }

  #[test]
  fn floating_call_goldman_sosin_gatto() {
    // S=100, S_min=100, r=0.10, σ=0.10, T=0.5 → 8.2687 (Haug formula)
    let option = lookback(None, Some(100.0), None, OptionType::Call);
    let price = LookbackPricer::new(0.10).price(&option, 100.0, 0.10, 0.0, 0.5);
    assert!(
      (price - 8.2687).abs() < 0.01,
      "floating call={price}, expected≈8.2687"
    );
  }

  #[test]
  fn floating_put_positive() {
    let option = lookback(None, None, Some(110.0), OptionType::Put);
    let price = LookbackPricer::new(0.2).price(&option, 100.0, 0.05, 0.0, 1.0);
    assert!(price > 0.0, "floating lookback put={price}");
    // Put must be ≥ discounted intrinsic (S_max - S)
    let intrinsic = (110.0 - 100.0) * (-0.05_f64).exp();
    assert!(price >= intrinsic, "price={price} < intrinsic={intrinsic}");
  }

  #[test]
  fn floating_call_more_volatile_is_more_expensive() {
    let option = lookback(None, Some(100.0), None, OptionType::Call);
    let base = LookbackPricer::new(0.2).price(&option, 100.0, 0.05, 0.0, 1.0);
    let high_vol = LookbackPricer::new(0.4).price(&option, 100.0, 0.05, 0.0, 1.0);
    assert!(high_vol > base);
  }

  #[test]
  fn fixed_call_geq_vanilla() {
    // Fixed-strike lookback call ≥ vanilla call (since payoff = max(S_max - K, 0) ≥ max(S_T - K, 0))
    let option = lookback(Some(100.0), None, Some(100.0), OptionType::Call);
    let price = LookbackPricer::new(0.2).price(&option, 100.0, 0.05, 0.0, 1.0);
    // BSM vanilla call ≈ 10.45
    assert!(
      price > 10.0,
      "fixed lookback call={price}, should be > vanilla≈10.45"
    );
  }

  #[test]
  fn fixed_put_geq_vanilla() {
    let option = lookback(Some(100.0), Some(100.0), None, OptionType::Put);
    let price = LookbackPricer::new(0.2).price(&option, 100.0, 0.05, 0.0, 1.0);
    // BSM vanilla put ≈ 5.57
    assert!(
      price > 5.0,
      "fixed lookback put={price}, should be > vanilla≈5.57"
    );
  }
}
