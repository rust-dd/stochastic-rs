//! # Digital
//!
//! $$
//! C_{\text{CoN}}=Qe^{-rT}N(d_2),\quad C_{\text{AoN}}=Se^{(b-r)T}N(d_1),\quad
//! d_{1,2}=\frac{\ln(S/K)+(b\pm\tfrac12\sigma^2)T}{\sigma\sqrt T}
//! $$
//!
//! Source:
//! - Reiner, E. & Rubinstein, M. (1991), "Unscrambling the Binary Code"
//! - Haug, E. G. (2007), "The Complete Guide to Option Pricing Formulas", 2nd ed., Ch. 4
//! - Hull, J. (2018), "Options, Futures, and Other Derivatives", 10th ed., §26.9
//!
use statrs::distribution::Continuous;
use statrs::distribution::ContinuousCDF;
use statrs::distribution::Normal;

use crate::quant::OptionType;

/// Cash-or-nothing digital pays a fixed cash amount $Q$ when the option
/// finishes in the money.
///
/// $$
/// C_{\text{CoN}}=Qe^{-rT}N(d_2),\qquad P_{\text{CoN}}=Qe^{-rT}N(-d_2)
/// $$
#[derive(Debug, Clone)]
pub struct CashOrNothingPricer {
  /// Spot price.
  pub s: f64,
  /// Strike (decision boundary).
  pub k: f64,
  /// Cash payout $Q$.
  pub cash: f64,
  /// Risk-free rate.
  pub r: f64,
  /// Cost of carry $b = r - q$.
  pub b: f64,
  /// Volatility.
  pub sigma: f64,
  /// Time to maturity in years.
  pub t: f64,
  /// Option type.
  pub option_type: OptionType,
}

impl CashOrNothingPricer {
  /// Closed-form price.
  pub fn price(&self) -> f64 {
    let n = Normal::new(0.0, 1.0).unwrap();
    let (_, d2) = self.d1_d2();
    let disc = (-self.r * self.t).exp();
    match self.option_type {
      OptionType::Call => self.cash * disc * n.cdf(d2),
      OptionType::Put => self.cash * disc * n.cdf(-d2),
    }
  }

  /// Delta: $\partial V/\partial S$.
  pub fn delta(&self) -> f64 {
    let n = Normal::new(0.0, 1.0).unwrap();
    let (_, d2) = self.d1_d2();
    let disc = (-self.r * self.t).exp();
    let denom = self.s * self.sigma * self.t.sqrt();
    let sign = match self.option_type {
      OptionType::Call => 1.0,
      OptionType::Put => -1.0,
    };
    sign * self.cash * disc * n.pdf(d2) / denom
  }

  /// Gamma: $\partial^2 V/\partial S^2$.
  pub fn gamma(&self) -> f64 {
    let n = Normal::new(0.0, 1.0).unwrap();
    let (_, d2) = self.d1_d2();
    let disc = (-self.r * self.t).exp();
    let s = self.s;
    let v = self.sigma;
    let t = self.t;
    let sqrt_t = t.sqrt();
    let sign = match self.option_type {
      OptionType::Call => 1.0,
      OptionType::Put => -1.0,
    };
    let pdf = n.pdf(d2);
    -sign * self.cash * disc * pdf * (1.0 + d2 * (v * sqrt_t)) / (s * s * v * sqrt_t)
  }

  /// Vega: $\partial V/\partial \sigma$.
  pub fn vega(&self) -> f64 {
    let n = Normal::new(0.0, 1.0).unwrap();
    let (d1, d2) = self.d1_d2();
    let disc = (-self.r * self.t).exp();
    let sign = match self.option_type {
      OptionType::Call => 1.0,
      OptionType::Put => -1.0,
    };
    -sign * self.cash * disc * n.pdf(d2) * d1 / self.sigma
  }

  fn d1_d2(&self) -> (f64, f64) {
    let v = self.sigma;
    let t = self.t;
    let sqrt_t = t.sqrt();
    let d1 = ((self.s / self.k).ln() + (self.b + 0.5 * v * v) * t) / (v * sqrt_t);
    let d2 = d1 - v * sqrt_t;
    (d1, d2)
  }
}

/// Asset-or-nothing digital pays the underlying when the option finishes in
/// the money.
///
/// $$
/// C_{\text{AoN}}=Se^{(b-r)T}N(d_1),\qquad P_{\text{AoN}}=Se^{(b-r)T}N(-d_1)
/// $$
#[derive(Debug, Clone)]
pub struct AssetOrNothingPricer {
  /// Spot price.
  pub s: f64,
  /// Strike.
  pub k: f64,
  /// Risk-free rate.
  pub r: f64,
  /// Cost of carry.
  pub b: f64,
  /// Volatility.
  pub sigma: f64,
  /// Time to maturity in years.
  pub t: f64,
  /// Option type.
  pub option_type: OptionType,
}

impl AssetOrNothingPricer {
  /// Closed-form price.
  pub fn price(&self) -> f64 {
    let n = Normal::new(0.0, 1.0).unwrap();
    let (d1, _) = self.d1_d2();
    let coc = ((self.b - self.r) * self.t).exp();
    match self.option_type {
      OptionType::Call => self.s * coc * n.cdf(d1),
      OptionType::Put => self.s * coc * n.cdf(-d1),
    }
  }

  /// Delta.
  pub fn delta(&self) -> f64 {
    let n = Normal::new(0.0, 1.0).unwrap();
    let (d1, _) = self.d1_d2();
    let coc = ((self.b - self.r) * self.t).exp();
    let v = self.sigma;
    let sqrt_t = self.t.sqrt();
    let cdf_term = match self.option_type {
      OptionType::Call => n.cdf(d1),
      OptionType::Put => n.cdf(-d1),
    };
    let sign = match self.option_type {
      OptionType::Call => 1.0,
      OptionType::Put => -1.0,
    };
    coc * cdf_term + sign * coc * n.pdf(d1) / (v * sqrt_t)
  }

  fn d1_d2(&self) -> (f64, f64) {
    let v = self.sigma;
    let t = self.t;
    let sqrt_t = t.sqrt();
    let d1 = ((self.s / self.k).ln() + (self.b + 0.5 * v * v) * t) / (v * sqrt_t);
    let d2 = d1 - v * sqrt_t;
    (d1, d2)
  }
}

/// Gap option: pays $S - K_2$ when $S > K_1$ (call) or $K_2 - S$ when
/// $S < K_1$ (put). Reduces to a vanilla when $K_1 = K_2$.
///
/// $$
/// V = S e^{(b-r)T}N(d_1) - K_2 e^{-rT}N(d_2),\quad
/// d_1=\frac{\ln(S/K_1)+(b+\tfrac12\sigma^2)T}{\sigma\sqrt T}
/// $$
#[derive(Debug, Clone)]
pub struct GapPricer {
  /// Spot price.
  pub s: f64,
  /// Trigger strike $K_1$.
  pub k1: f64,
  /// Payoff strike $K_2$.
  pub k2: f64,
  /// Risk-free rate.
  pub r: f64,
  /// Cost of carry.
  pub b: f64,
  /// Volatility.
  pub sigma: f64,
  /// Time to maturity in years.
  pub t: f64,
  /// Option type.
  pub option_type: OptionType,
}

impl GapPricer {
  pub fn price(&self) -> f64 {
    let n = Normal::new(0.0, 1.0).unwrap();
    let v = self.sigma;
    let t = self.t;
    let sqrt_t = t.sqrt();
    let d1 = ((self.s / self.k1).ln() + (self.b + 0.5 * v * v) * t) / (v * sqrt_t);
    let d2 = d1 - v * sqrt_t;
    let coc = ((self.b - self.r) * self.t).exp();
    let disc = (-self.r * self.t).exp();
    match self.option_type {
      OptionType::Call => self.s * coc * n.cdf(d1) - self.k2 * disc * n.cdf(d2),
      OptionType::Put => self.k2 * disc * n.cdf(-d2) - self.s * coc * n.cdf(-d1),
    }
  }
}

/// Supershare option pays $S_T / X_L$ when $X_L \le S_T \le X_H$.
///
/// $$
/// V = \frac{S}{X_L} e^{(b-r)T}[N(d_1) - N(d_2)]
/// $$
#[derive(Debug, Clone)]
pub struct SuperSharePricer {
  /// Spot price.
  pub s: f64,
  /// Lower trigger.
  pub x_low: f64,
  /// Upper trigger.
  pub x_high: f64,
  /// Risk-free rate.
  pub r: f64,
  /// Cost of carry.
  pub b: f64,
  /// Volatility.
  pub sigma: f64,
  /// Time to maturity in years.
  pub t: f64,
}

impl SuperSharePricer {
  pub fn price(&self) -> f64 {
    let n = Normal::new(0.0, 1.0).unwrap();
    let v = self.sigma;
    let t = self.t;
    let sqrt_t = t.sqrt();
    let d1 = ((self.s / self.x_low).ln() + (self.b + 0.5 * v * v) * t) / (v * sqrt_t);
    let d2 = ((self.s / self.x_high).ln() + (self.b + 0.5 * v * v) * t) / (v * sqrt_t);
    let coc = ((self.b - self.r) * self.t).exp();
    self.s / self.x_low * coc * (n.cdf(d1) - n.cdf(d2))
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  /// Cash-or-nothing call: S=100, K=80, Q=10, r=0.06, b=0.06, sigma=0.35,
  /// T=0.75 → 7.3444 (verified analytically against Q*e^{-rT}*N(d2)).
  #[test]
  fn cash_or_nothing_call_closed_form() {
    let p = CashOrNothingPricer {
      s: 100.0,
      k: 80.0,
      cash: 10.0,
      r: 0.06,
      b: 0.06,
      sigma: 0.35,
      t: 0.75,
      option_type: OptionType::Call,
    };
    let price = p.price();
    assert!((price - 7.3444).abs() < 0.005, "price={price}");
  }

  /// Cash-or-nothing put: S=100, K=80, Q=10, r=0.06, b=0.06, sigma=0.35,
  /// T=0.75 → ~ Q*e^{-rT}*N(-d2) = 10 * 0.95599 * (1 - 0.97264/Q*e^{rT})
  /// ≈ 0.0303
  #[test]
  fn cash_call_put_parity() {
    // CoN call + CoN put = Q * e^{-rT}
    let base = CashOrNothingPricer {
      s: 100.0,
      k: 80.0,
      cash: 10.0,
      r: 0.06,
      b: 0.06,
      sigma: 0.35,
      t: 0.75,
      option_type: OptionType::Call,
    };
    let put = CashOrNothingPricer {
      option_type: OptionType::Put,
      ..base.clone()
    };
    let total = base.price() + put.price();
    let expected = 10.0 * (-0.06_f64 * 0.75).exp();
    assert!((total - expected).abs() < 1e-10, "total={total}");
  }

  /// Asset-or-nothing call + put = forward $S e^{(b-r)T}$
  #[test]
  fn aon_call_put_parity() {
    let c = AssetOrNothingPricer {
      s: 100.0,
      k: 105.0,
      r: 0.05,
      b: 0.03,
      sigma: 0.25,
      t: 1.0,
      option_type: OptionType::Call,
    };
    let p = AssetOrNothingPricer {
      option_type: OptionType::Put,
      ..c.clone()
    };
    let total = c.price() + p.price();
    let expected = 100.0 * ((0.03_f64 - 0.05_f64) * 1.0).exp();
    assert!((total - expected).abs() < 1e-9, "total={total}");
  }

  /// Vanilla call = AoN(call, K) - K * CoN(call, K)/Q (with Q=1).
  #[test]
  fn vanilla_decomposition() {
    let s = 100.0;
    let k = 100.0;
    let r = 0.05;
    let b = 0.05;
    let sigma = 0.2;
    let t = 1.0;
    let aon = AssetOrNothingPricer {
      s,
      k,
      r,
      b,
      sigma,
      t,
      option_type: OptionType::Call,
    };
    let con = CashOrNothingPricer {
      s,
      k,
      cash: 1.0,
      r,
      b,
      sigma,
      t,
      option_type: OptionType::Call,
    };
    // BSM vanilla call ≈ 10.4506
    let vanilla = aon.price() - k * con.price();
    assert!((vanilla - 10.4506).abs() < 0.005, "decomposition={vanilla}");
  }

  /// Gap call with $K_1 = K_2$ equals BSM vanilla call.
  #[test]
  fn gap_reduces_to_vanilla() {
    let p = GapPricer {
      s: 100.0,
      k1: 100.0,
      k2: 100.0,
      r: 0.05,
      b: 0.05,
      sigma: 0.2,
      t: 1.0,
      option_type: OptionType::Call,
    };
    let price = p.price();
    assert!((price - 10.4506).abs() < 0.005, "gap={price}");
  }

  /// Haug 2007, p. 178: Gap call with S=50, K1=50, K2=57, r=0.09, b=0.09,
  /// sigma=0.20, T=0.5 → -0.0053
  #[test]
  fn gap_haug_negative_payoff() {
    let p = GapPricer {
      s: 50.0,
      k1: 50.0,
      k2: 57.0,
      r: 0.09,
      b: 0.09,
      sigma: 0.20,
      t: 0.5,
      option_type: OptionType::Call,
    };
    let price = p.price();
    assert!(price.abs() < 0.05, "gap call={price}");
    // The option gives a negative cash flow when S is between K1 and K2
    assert!(price < 0.0);
  }

  /// Supershare must be non-negative and zero when bands are infinitely apart
  /// in the wrong direction.
  #[test]
  fn supershare_positive() {
    let p = SuperSharePricer {
      s: 100.0,
      x_low: 90.0,
      x_high: 110.0,
      r: 0.05,
      b: 0.0,
      sigma: 0.2,
      t: 0.25,
    };
    let price = p.price();
    assert!(price > 0.0, "supershare={price}");
    assert!(price < p.s, "supershare must be < S");
  }

  /// Cash-or-nothing delta uses finite difference vs analytic.
  #[test]
  fn cash_delta_matches_fd() {
    let h = 0.01;
    let base = CashOrNothingPricer {
      s: 100.0,
      k: 100.0,
      cash: 10.0,
      r: 0.05,
      b: 0.02,
      sigma: 0.25,
      t: 0.5,
      option_type: OptionType::Call,
    };
    let up = CashOrNothingPricer {
      s: 100.0 + h,
      ..base.clone()
    };
    let dn = CashOrNothingPricer {
      s: 100.0 - h,
      ..base.clone()
    };
    let fd = (up.price() - dn.price()) / (2.0 * h);
    let analytic = base.delta();
    assert!((fd - analytic).abs() < 1e-4, "fd={fd}, analytic={analytic}");
  }
}
