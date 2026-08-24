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
use stochastic_rs_distributions::special::norm_cdf;
use stochastic_rs_distributions::special::norm_pdf;

use crate::OptionType;
use crate::traits::ModelPricer;

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
  pub tau: f64,
  /// Option type.
  pub option_type: OptionType,
}

impl CashOrNothingPricer {
  /// Closed-form price.
  pub fn price(&self) -> f64 {
    let (_, d2) = self.d1_d2();
    let disc = (-self.r * self.tau).exp();
    match self.option_type {
      OptionType::Call => self.cash * disc * norm_cdf(d2),
      OptionType::Put => self.cash * disc * norm_cdf(-d2),
    }
  }

  /// Delta: $\partial V/\partial S$.
  pub fn delta(&self) -> f64 {
    let (_, d2) = self.d1_d2();
    let disc = (-self.r * self.tau).exp();
    let denom = self.s * self.sigma * self.tau.sqrt();
    let sign = match self.option_type {
      OptionType::Call => 1.0,
      OptionType::Put => -1.0,
    };
    sign * self.cash * disc * norm_pdf(d2) / denom
  }

  /// Gamma: $\partial^2 V/\partial S^2$.
  pub fn gamma(&self) -> f64 {
    let (_, d2) = self.d1_d2();
    let disc = (-self.r * self.tau).exp();
    let s = self.s;
    let v = self.sigma;
    let t = self.tau;
    let sqrt_t = t.sqrt();
    let sign = match self.option_type {
      OptionType::Call => 1.0,
      OptionType::Put => -1.0,
    };
    let pdf = norm_pdf(d2);
    -sign * self.cash * disc * pdf * (1.0 + d2 * (v * sqrt_t)) / (s * s * v * sqrt_t)
  }

  /// Vega: $\partial V/\partial \sigma$.
  pub fn vega(&self) -> f64 {
    let (d1, d2) = self.d1_d2();
    let disc = (-self.r * self.tau).exp();
    let sign = match self.option_type {
      OptionType::Call => 1.0,
      OptionType::Put => -1.0,
    };
    -sign * self.cash * disc * norm_pdf(d2) * d1 / self.sigma
  }

  fn d1_d2(&self) -> (f64, f64) {
    let v = self.sigma;
    let t = self.tau;
    let sqrt_t = t.sqrt();
    let d1 = ((self.s / self.k).ln() + (self.b + 0.5 * v * v) * t) / (v * sqrt_t);
    let d2 = d1 - v * sqrt_t;
    (d1, d2)
  }
}

impl crate::traits::GreeksExt for CashOrNothingPricer {
  fn delta(&self) -> f64 {
    CashOrNothingPricer::delta(self)
  }
  fn gamma(&self) -> f64 {
    CashOrNothingPricer::gamma(self)
  }
  fn vega(&self) -> f64 {
    CashOrNothingPricer::vega(self)
  }
}

impl ModelPricer for CashOrNothingPricer {
  /// `cash`/`sigma` are model state; `(s, k, r, q, tau)` is the query, with
  /// $b = r - q$ fed into $d_2$.
  fn price_call(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    let (_, d2) = bsm_d1_d2(s, k, r - q, self.sigma, tau);
    let disc = (-r * tau).exp();
    self.cash * disc * norm_cdf(d2)
  }

  /// Overrides the trait's vanilla-parity default: here
  /// $C_{\text{CoN}} + P_{\text{CoN}} = Qe^{-rT}$, not
  /// $C - Se^{-qT} + Ke^{-rT}$ — see `cash_call_put_parity`.
  fn price_put(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    let (_, d2) = bsm_d1_d2(s, k, r - q, self.sigma, tau);
    let disc = (-r * tau).exp();
    self.cash * disc * norm_cdf(-d2)
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
  pub tau: f64,
  /// Option type.
  pub option_type: OptionType,
}

impl AssetOrNothingPricer {
  /// Closed-form price.
  pub fn price(&self) -> f64 {
    let (d1, _) = self.d1_d2();
    let coc = ((self.b - self.r) * self.tau).exp();
    match self.option_type {
      OptionType::Call => self.s * coc * norm_cdf(d1),
      OptionType::Put => self.s * coc * norm_cdf(-d1),
    }
  }

  /// Delta.
  pub fn delta(&self) -> f64 {
    let (d1, _) = self.d1_d2();
    let coc = ((self.b - self.r) * self.tau).exp();
    let v = self.sigma;
    let sqrt_t = self.tau.sqrt();
    let cdf_term = match self.option_type {
      OptionType::Call => norm_cdf(d1),
      OptionType::Put => norm_cdf(-d1),
    };
    let sign = match self.option_type {
      OptionType::Call => 1.0,
      OptionType::Put => -1.0,
    };
    coc * cdf_term + sign * coc * norm_pdf(d1) / (v * sqrt_t)
  }

  fn d1_d2(&self) -> (f64, f64) {
    let v = self.sigma;
    let t = self.tau;
    let sqrt_t = t.sqrt();
    let d1 = ((self.s / self.k).ln() + (self.b + 0.5 * v * v) * t) / (v * sqrt_t);
    let d2 = d1 - v * sqrt_t;
    (d1, d2)
  }
}

impl crate::traits::GreeksExt for AssetOrNothingPricer {
  fn delta(&self) -> f64 {
    AssetOrNothingPricer::delta(self)
  }
}

impl ModelPricer for AssetOrNothingPricer {
  /// `sigma` is model state; `(s, k, r, q, tau)` is the query, $b = r - q$.
  fn price_call(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    let b = r - q;
    let (d1, _) = bsm_d1_d2(s, k, b, self.sigma, tau);
    let coc = ((b - r) * tau).exp();
    s * coc * norm_cdf(d1)
  }

  /// Overrides the trait's vanilla-parity default: here
  /// $C_{\text{AoN}} + P_{\text{AoN}} = Se^{(b-r)T}$ — see
  /// `aon_call_put_parity` — not vanilla put-call parity.
  fn price_put(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    let b = r - q;
    let (d1, _) = bsm_d1_d2(s, k, b, self.sigma, tau);
    let coc = ((b - r) * tau).exp();
    s * coc * norm_cdf(-d1)
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
  pub tau: f64,
  /// Option type.
  pub option_type: OptionType,
}

impl GapPricer {
  pub fn price(&self) -> f64 {
    let v = self.sigma;
    let t = self.tau;
    let sqrt_t = t.sqrt();
    let d1 = ((self.s / self.k1).ln() + (self.b + 0.5 * v * v) * t) / (v * sqrt_t);
    let d2 = d1 - v * sqrt_t;
    let coc = ((self.b - self.r) * self.tau).exp();
    let disc = (-self.r * self.tau).exp();
    match self.option_type {
      OptionType::Call => self.s * coc * norm_cdf(d1) - self.k2 * disc * norm_cdf(d2),
      OptionType::Put => self.k2 * disc * norm_cdf(-d2) - self.s * coc * norm_cdf(-d1),
    }
  }
}

impl ModelPricer for GapPricer {
  /// $K_1$ (the moneyness trigger) is the query strike `k`; $K_2$ (the
  /// payoff strike) and `sigma` stay model state on `self.k2`/`self.sigma`
  /// — reduces to a vanilla call when `self.k2 == k`, see
  /// `gap_reduces_to_vanilla`.
  fn price_call(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    let b = r - q;
    let (d1, d2) = bsm_d1_d2(s, k, b, self.sigma, tau);
    let coc = ((b - r) * tau).exp();
    let disc = (-r * tau).exp();
    s * coc * norm_cdf(d1) - self.k2 * disc * norm_cdf(d2)
  }

  /// Overrides the trait's default with the gap put's own closed form
  /// rather than trusting vanilla parity against a single strike.
  fn price_put(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    let b = r - q;
    let (d1, d2) = bsm_d1_d2(s, k, b, self.sigma, tau);
    let coc = ((b - r) * tau).exp();
    let disc = (-r * tau).exp();
    self.k2 * disc * norm_cdf(-d2) - s * coc * norm_cdf(-d1)
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
  pub tau: f64,
}

impl SuperSharePricer {
  pub fn price(&self) -> f64 {
    let v = self.sigma;
    let t = self.tau;
    let sqrt_t = t.sqrt();
    let d1 = ((self.s / self.x_low).ln() + (self.b + 0.5 * v * v) * t) / (v * sqrt_t);
    let d2 = ((self.s / self.x_high).ln() + (self.b + 0.5 * v * v) * t) / (v * sqrt_t);
    let coc = ((self.b - self.r) * self.tau).exp();
    self.s / self.x_low * coc * (norm_cdf(d1) - norm_cdf(d2))
  }
}

impl ModelPricer for SuperSharePricer {
  /// $X_L$ (the lower trigger, also the payoff divisor) is the query
  /// strike `k`; $X_H$ and `sigma` stay model state on `self.x_high`/
  /// `self.sigma`.
  fn price_call(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    let b = r - q;
    let d1 = bsm_d1(s, k, b, self.sigma, tau);
    let d2 = bsm_d1(s, self.x_high, b, self.sigma, tau);
    let coc = ((b - r) * tau).exp();
    s / k * coc * (norm_cdf(d1) - norm_cdf(d2))
  }

  /// A supershare has no put analogue — Haug (2007) defines only the
  /// payoff above — so this returns `NaN` rather than the trait's
  /// vanilla-parity default, which would silently return a number with no
  /// corresponding instrument.
  fn price_put(&self, _s: f64, _k: f64, _r: f64, _q: f64, _tau: f64) -> f64 {
    f64::NAN
  }
}

/// $d_1=\frac{\ln(S/K)+(b+\sigma^2/2)T}{\sigma\sqrt T}$ — the standardized
/// moneyness term shared by every [`ModelPricer`] impl above.
fn bsm_d1(s: f64, k: f64, b: f64, sigma: f64, t: f64) -> f64 {
  ((s / k).ln() + (b + 0.5 * sigma * sigma) * t) / (sigma * t.sqrt())
}

/// $(d_1,\ d_1-\sigma\sqrt T)$.
fn bsm_d1_d2(s: f64, k: f64, b: f64, sigma: f64, t: f64) -> (f64, f64) {
  let d1 = bsm_d1(s, k, b, sigma, t);
  (d1, d1 - sigma * t.sqrt())
}

#[cfg(test)]
#[path = "digital_tests.rs"]
mod tests;
