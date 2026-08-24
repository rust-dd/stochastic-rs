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

use stochastic_rs_distributions::special::norm_cdf;

/// Kirk's approximation pricer for European spread options.
///
/// The payoff is `max(F1 - F2 - X, 0)` for a call and
/// `max(X - (F1 - F2), 0)` for a put, where `F1` and `F2`
/// are two commodity forward prices and `X` is the strike (conversion cost).
///
/// The struct holds **model state only** — the two volatilities and their
/// correlation. The two forwards, the spread strike, the rate and the
/// maturity are the pricing *query* and travel as arguments to
/// [`call_put`](Self::call_put), so one instance prices a whole
/// forward/strike/maturity grid.
///
/// This is a **two-forward** payoff, so it deliberately carries no
/// [`ModelPricer`](crate::traits::ModelPricer): that trait's
/// `price_call(s, k, r, q, tau)` has one underlying, and widening it to fit
/// a spread would mean an optional second leg. Kirk therefore belongs to the
/// multi-asset "convention, no trait" family alongside `MargrabePricer` and
/// `McSpreadPricer`, and exposes the same model/query split through inherent
/// methods.
///
/// ```
/// use stochastic_rs_quant::pricing::kirk::KirkSpreadPricer;
///
/// let model = KirkSpreadPricer::new(0.35, 0.35, 0.9);
/// let (call, put) = model.call_put(35.0, 34.0, 3.0, 0.05, 1.0);
/// assert!(put > call, "the spread 35 - 34 is far below the strike 3");
/// ```
#[derive(Debug, Clone, Copy)]
pub struct KirkSpreadPricer {
  /// Volatility of asset 1
  pub v1: f64,
  /// Volatility of asset 2
  pub v2: f64,
  /// Correlation between asset 1 and asset 2
  pub corr: f64,
}

impl KirkSpreadPricer {
  pub const fn new(v1: f64, v2: f64, corr: f64) -> Self {
    Self { v1, v2, corr }
  }

  /// Call and put price at one query point.
  ///
  /// `f1` and `f2` are the two forwards, `x` the spread strike (conversion
  /// cost), `r` the risk-free rate and `tau` the maturity in years.
  ///
  /// The combined volatility is query-dependent, not model state: Kirk
  /// weights $\sigma_2$ by $F_2/(F_2+X)$, so it is recomputed per call
  /// rather than cached on the struct.
  pub fn call_put(&self, f1: f64, f2: f64, x: f64, r: f64, tau: f64) -> (f64, f64) {
    // Ratio transformation: F = F1 / (F2 + X)
    let denom = f2 + x;
    let f = f1 / denom;
    let f_temp = f2 / denom;

    // Combined volatility (Kirk's approximation)
    let v = (self.v1.powi(2) + (self.v2 * f_temp).powi(2)
      - 2.0 * self.corr * self.v1 * self.v2 * f_temp)
      .sqrt();

    // Black-76 style pricing (b = 0 for futures)
    let d1 = (f.ln() + 0.5 * v.powi(2) * tau) / (v * tau.sqrt());
    let d2 = d1 - v * tau.sqrt();

    let df = (-r * tau).exp();

    let call = denom * (f * df * norm_cdf(d1) - df * norm_cdf(d2));
    let put = denom * (df * norm_cdf(-d2) - f * df * norm_cdf(-d1));

    (call, put)
  }

  /// Price the spread call $\max(F_1-F_2-X,0)$ at one query point.
  pub fn price_call(&self, f1: f64, f2: f64, x: f64, r: f64, tau: f64) -> f64 {
    self.call_put(f1, f2, x, r, tau).0
  }

  /// Price the spread put $\max(X-(F_1-F_2),0)$ at one query point.
  pub fn price_put(&self, f1: f64, f2: f64, x: f64, r: f64, tau: f64) -> f64 {
    self.call_put(f1, f2, x, r, tau).1
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  /// Cross-arch tolerance: these goldens come from `norm_cdf`, whose last
  /// bit is a hostage to FMA contraction and libm differences between the
  /// aarch64-darwin dev machine and CI's ubuntu x86_64.
  const TOL: f64 = 1e-12;

  /// Values captured from `PricerExt::calculate_call_put()` **before** the
  /// model/query reshape, at `KirkSpreadPricer::new(f1, f2, x, r, v1, v2,
  /// corr, Some(tau), None, None)`. The reshape is an API change only, so
  /// these must not move.
  #[test]
  fn kirk_call_put_matches_pre_refactor_goldens() {
    let heat_rate = KirkSpreadPricer::new(0.35, 0.35, 0.90);
    let (call, put) = heat_rate.call_put(35.0, 34.0, 3.0, 0.05, 1.0);
    assert!((call - 1.2691102653060158).abs() < TOL, "call {call}");
    assert!((put - 3.1715691143074434).abs() < TOL, "put {put}");

    let itm = KirkSpreadPricer::new(0.30, 0.25, 0.7);
    let (call, put) = itm.call_put(100.0, 90.0, 5.0, 0.05, 0.5);
    assert!((call - 8.547484304937198).abs() < TOL, "call {call}");
    assert!((put - 3.670934744795539).abs() < TOL, "put {put}");
  }

  /// `price_call` / `price_put` are projections of `call_put`, not
  /// recomputations — they must agree bit for bit.
  #[test]
  fn kirk_price_legs_project_call_put() {
    let model = KirkSpreadPricer::new(0.30, 0.25, 0.7);
    let (call, put) = model.call_put(100.0, 90.0, 5.0, 0.05, 0.5);
    assert_eq!(model.price_call(100.0, 90.0, 5.0, 0.05, 0.5), call);
    assert_eq!(model.price_put(100.0, 90.0, 5.0, 0.05, 0.5), put);
  }

  /// One model instance prices a whole query grid — the point of the split.
  #[test]
  fn kirk_one_model_prices_a_strike_grid() {
    let model = KirkSpreadPricer::new(0.30, 0.25, 0.7);
    let prices = [1.0, 5.0, 10.0].map(|x| model.price_call(100.0, 90.0, x, 0.05, 0.5));
    assert!(
      prices[0] > prices[1] && prices[1] > prices[2],
      "spread calls must decay in the strike: {prices:?}"
    );
  }
}
