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

/// Kirk's approximation for European spread options, payoff `max(F1 − F2 − X, 0)`; a two-forward
/// query, so no [`ModelPricer`](crate::traits::ModelPricer) and `spread_*` method names instead.
///
/// ```
/// use stochastic_rs_quant::pricing::kirk::KirkSpreadPricer;
///
/// let model = KirkSpreadPricer::new(0.35, 0.35, 0.9);
/// let (call, put) = model.spread_call_put(35.0, 34.0, 3.0, 0.05, 1.0);
/// assert!(put > call, "the spread 35 - 34 is far below the strike 3");
/// ```
///
/// ```compile_fail
/// use stochastic_rs_quant::pricing::kirk::KirkSpreadPricer;
/// use stochastic_rs_quant::traits::ModelPricer;
///
/// let model = KirkSpreadPricer::new(0.35, 0.35, 0.9);
/// let _ = model.price_call(100.0, 95.0, 0.05, 0.02, 1.0);
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
  /// Panics unless `v1, v2 ≥ 0` and `corr ∈ [-1, 1]`: at `corr < -1` the radicand stays positive and
  /// the price comes back finite and an order of magnitude wrong.
  pub fn new(v1: f64, v2: f64, corr: f64) -> Self {
    assert!(
      v1 >= 0.0,
      "KirkSpreadPricer::new: v1 must be a non-negative volatility (got {v1})"
    );
    assert!(
      v2 >= 0.0,
      "KirkSpreadPricer::new: v2 must be a non-negative volatility (got {v2})"
    );
    assert!(
      (-1.0..=1.0).contains(&corr),
      "KirkSpreadPricer::new: corr must be in [-1, 1] (got {corr})"
    );
    Self { v1, v2, corr }
  }

  /// Spread call and spread put at one query point.
  ///
  /// `f1` and `f2` are the two forwards, `x` the spread strike (conversion
  /// cost), `r` the risk-free rate and `tau` the maturity in years. **This is
  /// not the `(s, k, r, q, tau)` query** the single-underlying pricers take
  /// at the same arity; [`KirkSpreadPricer`](Self)'s own documentation has
  /// the reason the name carries the `spread_` prefix.
  ///
  /// The combined volatility is query-dependent, not model state: Kirk
  /// weights $\sigma_2$ by $F_2/(F_2+X)$, so it is recomputed per call
  /// rather than cached on the struct.
  pub fn spread_call_put(&self, f1: f64, f2: f64, x: f64, r: f64, tau: f64) -> (f64, f64) {
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
  ///
  /// `f1` and `f2` are the two forwards, `x` the spread strike (conversion
  /// cost), `r` the risk-free rate and `tau` the maturity in years — the
  /// same query [`spread_call_put`](Self::spread_call_put) takes, of which
  /// this is the first projection.
  pub fn spread_call(&self, f1: f64, f2: f64, x: f64, r: f64, tau: f64) -> f64 {
    self.spread_call_put(f1, f2, x, r, tau).0
  }

  /// Price the spread put $\max(X-(F_1-F_2),0)$ at one query point.
  ///
  /// `f1` and `f2` are the two forwards, `x` the spread strike (conversion
  /// cost), `r` the risk-free rate and `tau` the maturity in years — the
  /// same query [`spread_call_put`](Self::spread_call_put) takes, of which
  /// this is the second projection.
  pub fn spread_put(&self, f1: f64, f2: f64, x: f64, r: f64, tau: f64) -> f64 {
    self.spread_call_put(f1, f2, x, r, tau).1
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
  /// corr, Some(tau), None, None)`. The reshape and the later `spread_*`
  /// rename are API changes only, so these must not move.
  #[test]
  fn kirk_call_put_matches_pre_refactor_goldens() {
    let heat_rate = KirkSpreadPricer::new(0.35, 0.35, 0.90);
    let (call, put) = heat_rate.spread_call_put(35.0, 34.0, 3.0, 0.05, 1.0);
    assert!((call - 1.2691091197737148).abs() < TOL, "call {call}");
    assert!((put - 3.1715679687751486).abs() < TOL, "put {put}");

    let itm = KirkSpreadPricer::new(0.30, 0.25, 0.7);
    let (call, put) = itm.spread_call_put(100.0, 90.0, 5.0, 0.05, 0.5);
    assert!((call - 8.547483131757428).abs() < TOL, "call {call}");
    assert!((put - 3.6709335716157754).abs() < TOL, "put {put}");
  }

  /// `spread_call` / `spread_put` are projections of `spread_call_put`, not
  /// recomputations — they must agree bit for bit.
  #[test]
  fn kirk_price_legs_project_call_put() {
    let model = KirkSpreadPricer::new(0.30, 0.25, 0.7);
    let (call, put) = model.spread_call_put(100.0, 90.0, 5.0, 0.05, 0.5);
    assert_eq!(model.spread_call(100.0, 90.0, 5.0, 0.05, 0.5), call);
    assert_eq!(model.spread_put(100.0, 90.0, 5.0, 0.05, 0.5), put);
  }

  /// A vanilla `(s, k, r, q, tau)` query read as a spread query prices finite and plausible, which is
  /// why the methods carry the `spread_` prefix.
  #[test]
  fn the_misread_vanilla_query_still_produces_a_plausible_number() {
    let model = KirkSpreadPricer::new(0.30, 0.25, 0.7);
    let misread = model.spread_call(100.0, 95.0, 0.05, 0.02, 1.0);
    assert!(
      (misread - 10.943413879466974).abs() < TOL,
      "misread spread price {misread}"
    );
    assert!(
      misread.is_finite() && misread > 0.0,
      "the confusion is silent precisely because this is a healthy number"
    );
  }

  /// One model instance prices a whole query grid — the point of the split.
  #[test]
  fn kirk_one_model_prices_a_strike_grid() {
    let model = KirkSpreadPricer::new(0.30, 0.25, 0.7);
    let prices = [1.0, 5.0, 10.0].map(|x| model.spread_call(100.0, 90.0, x, 0.05, 0.5));
    assert!(
      prices[0] > prices[1] && prices[1] > prices[2],
      "spread calls must decay in the strike: {prices:?}"
    );
  }

  /// Kirk's combined volatility is
  /// `√(σ₁² + (σ₂w)² - 2ρσ₁σ₂w)`, so a correlation outside `[-1, 1]` is not
  /// caught by the square root unless it happens to drive the radicand
  /// negative. At `ρ = -5` it does not: the spread call comes back
  /// `14.0959` against the correct `1.2691` — an order of magnitude out and
  /// entirely plausible. (`ρ = +5` does turn the radicand negative and
  /// yields `NaN`, so only one side of the invalid range announced itself.)
  #[test]
  #[should_panic(expected = "KirkSpreadPricer::new: corr must be in [-1, 1] (got -5)")]
  fn new_rejects_correlation_below_minus_one() {
    let _ = KirkSpreadPricer::new(0.35, 0.35, -5.0);
  }

  #[test]
  #[should_panic(expected = "KirkSpreadPricer::new: corr must be in [-1, 1] (got 5)")]
  fn new_rejects_correlation_above_one() {
    let _ = KirkSpreadPricer::new(0.35, 0.35, 5.0);
  }

  /// `v1 = -0.35` would price the call at `7.8656` instead of `1.2691`.
  #[test]
  #[should_panic(
    expected = "KirkSpreadPricer::new: v1 must be a non-negative volatility (got -0.35)"
  )]
  fn new_rejects_negative_first_volatility() {
    let _ = KirkSpreadPricer::new(-0.35, 0.35, 0.9);
  }

  #[test]
  #[should_panic(
    expected = "KirkSpreadPricer::new: v2 must be a non-negative volatility (got -0.35)"
  )]
  fn new_rejects_negative_second_volatility() {
    let _ = KirkSpreadPricer::new(0.35, -0.35, 0.9);
  }

  /// The admissible edges the validation must not swallow: perfect
  /// correlation either way, and a zero volatility leg.
  #[test]
  fn new_accepts_the_admissible_edges() {
    assert_eq!(KirkSpreadPricer::new(0.35, 0.0, -1.0).corr, -1.0);
    assert_eq!(KirkSpreadPricer::new(0.0, 0.35, 1.0).corr, 1.0);
  }
}
