//! # Kirk's Spread Option Approximation
//!
//! Approximate closed-form valuation for European spread options on two
//! commodity (futures-style) underlyings:
//!
//! $$
//! C=(F_2+X)\bigl[F\,e^{-rT}N(d_1)-e^{-rT}N(d_2)\bigr],\quad
//! F=\frac{F_1}{F_2+X}
//! $$
//!
//! with combined volatility
//!
//! $$
//! V=\sqrt{\sigma_1^2+\Bigl(\sigma_2\frac{F_2}{F_2+X}\Bigr)^2
//!          -2\rho\,\sigma_1\sigma_2\frac{F_2}{F_2+X}}
//! $$
//!
//! Reference: Kirk, E. (1995). "Correlation in the Energy Markets."
//! In *Managing Energy Price Risk*, Risk Publications, pp. 71-78.

use statrs::distribution::ContinuousCDF;
use statrs::distribution::Normal;

use crate::traits::PricerExt;
use crate::traits::TimeExt;

/// Kirk's approximation pricer for European spread options.
///
/// The payoff is `max(F1 - F2 - X, 0)` for a call and
/// `max(X - (F1 - F2), 0)` for a put, where `F1` and `F2`
/// are two commodity forward prices and `X` is the strike (conversion cost).
#[derive(Debug, Clone)]
pub struct KirkSpreadPricer {
  /// Forward price of asset 1
  pub f1: f64,
  /// Forward price of asset 2
  pub f2: f64,
  /// Strike (spread strike / conversion cost)
  pub x: f64,
  /// Risk-free rate
  pub r: f64,
  /// Volatility of asset 1
  pub v1: f64,
  /// Volatility of asset 2
  pub v2: f64,
  /// Correlation between asset 1 and asset 2
  pub corr: f64,
  /// Time to maturity in years
  pub tau: Option<f64>,
  /// Evaluation date
  pub eval: Option<chrono::NaiveDate>,
  /// Expiration date
  pub expiration: Option<chrono::NaiveDate>,
}

impl KirkSpreadPricer {
  pub fn new(
    f1: f64,
    f2: f64,
    x: f64,
    r: f64,
    v1: f64,
    v2: f64,
    corr: f64,
    tau: Option<f64>,
    eval: Option<chrono::NaiveDate>,
    expiration: Option<chrono::NaiveDate>,
  ) -> Self {
    Self {
      f1,
      f2,
      x,
      r,
      v1,
      v2,
      corr,
      tau,
      eval,
      expiration,
    }
  }
}

impl PricerExt for KirkSpreadPricer {
  fn calculate_call_put(&self) -> (f64, f64) {
    let tau = self.tau_or_from_dates();
    let n = Normal::new(0.0, 1.0).unwrap();

    // Ratio transformation: F = F1 / (F2 + X)
    let denom = self.f2 + self.x;
    let f = self.f1 / denom;
    let f_temp = self.f2 / denom;

    // Combined volatility (Kirk's approximation)
    let v = (self.v1.powi(2) + (self.v2 * f_temp).powi(2)
      - 2.0 * self.corr * self.v1 * self.v2 * f_temp)
      .sqrt();

    // Black-76 style pricing (b = 0 for futures)
    let d1 = (f.ln() + 0.5 * v.powi(2) * tau) / (v * tau.sqrt());
    let d2 = d1 - v * tau.sqrt();

    let df = (-self.r * tau).exp();

    let call = denom * (f * df * n.cdf(d1) - df * n.cdf(d2));
    let put = denom * (df * n.cdf(-d2) - f * df * n.cdf(-d1));

    (call, put)
  }

  fn calculate_price(&self) -> f64 {
    self.calculate_call_put().0
  }
}

impl TimeExt for KirkSpreadPricer {
  fn tau(&self) -> Option<f64> {
    self.tau
  }

  fn eval(&self) -> Option<chrono::NaiveDate> {
    self.eval
  }

  fn expiration(&self) -> Option<chrono::NaiveDate> {
    self.expiration
  }
}
