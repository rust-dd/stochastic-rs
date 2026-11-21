//! # Asian
//!
//! $$
//! V_0=e^{-rT}\,\mathbb E\!\left[\left(\frac1T\int_0^T S_tdt-K\right)^+\right]
//! $$
//!
use stochastic_rs_distributions::special::norm_cdf;

use crate::traits::ModelPricer;

/// Geometric-average Asian option under the Kemna-Vorst style closed form.
///
/// The struct holds **model state only** — the volatility of the
/// underlying. Spot, strike, rate, dividend yield and maturity are the
/// pricing *query* and travel as arguments to
/// [`ModelPricer::price_call`], so one instance prices a whole
/// strike/maturity grid.
///
/// ```
/// use stochastic_rs_quant::pricing::asian::AsianPricer;
/// use stochastic_rs_quant::traits::ModelPricer;
///
/// let model = AsianPricer::new(0.25);
/// let atm = model.price_call(100.0, 100.0, 0.05, 0.02, 0.75);
/// let otm = model.price_call(100.0, 120.0, 0.05, 0.02, 0.75);
/// assert!(atm > otm);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct AsianPricer {
  /// Volatility
  pub sigma: f64,
}

impl AsianPricer {
  /// Panics unless `sigma` is finite and `>= 0`, as a negative one prices a negative call; at
  /// `sigma == 0` the average is deterministic and priced at its discounted intrinsic value.
  pub fn new(sigma: f64) -> Self {
    assert!(
      sigma.is_finite() && sigma >= 0.0,
      "sigma must satisfy `sigma.is_finite() && sigma >= 0.0`, got sigma = {sigma:?}"
    );
    Self { sigma }
  }

  /// Average-rate volatility $\sigma/\sqrt3$ and the adjusted cost of carry
  /// used by the closed form at the query's rates.
  fn averaged_vol_and_carry(&self, r: f64, q: f64) -> (f64, f64) {
    let v = self.sigma / 3.0_f64.sqrt();
    let b = 0.5 * (r - q - 0.5 * v.powi(2) / 6.0);
    (v, b)
  }

  /// The adjusted carry `b` with the closed form's `d1`, `d2` at one query point.
  fn carry_d1_d2(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> (f64, f64, f64) {
    let (v, b) = self.averaged_vol_and_carry(r, q);
    let d1 = ((s / k).ln() + (b + 0.5 * v.powi(2) * tau)) / (v * tau.sqrt());
    (b, d1, d1 - v * tau.sqrt())
  }
}

impl ModelPricer for AsianPricer {
  fn price_call(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    let (b, d1, d2) = self.carry_d1_d2(s, k, r, q, tau);
    s * ((b - r) * tau).exp() * norm_cdf(d1) - k * (-r * tau).exp() * norm_cdf(d2)
  }

  /// Not the parity default: the averaged underlying carries at $e^{(b-r)\tau}$ with
  /// $b=\tfrac12(r-q-\tfrac{\sigma_A^2}{12})$, so parity reads $C-P=Se^{(b-r)\tau}-Ke^{-r\tau}$.
  fn price_put(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    let (b, d1, d2) = self.carry_d1_d2(s, k, r, q, tau);
    -s * ((b - r) * tau).exp() * norm_cdf(-d1) + k * (-r * tau).exp() * norm_cdf(-d2)
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  const S: f64 = 100.0;
  const K: f64 = 105.0;
  const R: f64 = 0.05;
  const Q: f64 = 0.02;
  const TAU: f64 = 0.75;
  const V: f64 = 0.25;

  /// Cross-arch tolerance: these goldens come from `norm_cdf`, whose last
  /// bit is a hostage to FMA contraction and libm differences between the
  /// aarch64-darwin dev machine and CI's ubuntu x86_64.
  const TOL: f64 = 1e-12;

  /// Values captured from `PricerExt::calculate_call_put()` **before** the
  /// `ModelPricer` reshape, at `AsianPricer::new(S, V, K, R, Some(Q),
  /// Some(TAU), None, None)`. The reshape is an API change only, so these
  /// must not move.
  #[test]
  fn asian_model_pricer_matches_pre_refactor_goldens() {
    let model = AsianPricer::new(V);
    let (call, put) = (
      model.price_call(S, K, R, Q, TAU),
      model.price_put(S, K, R, Q, TAU),
    );
    assert!((call - 3.2776906533992545).abs() < TOL, "call {call}");
    assert!((put - 7.067347664422634).abs() < TOL, "put {put}");
    assert_eq!(model.price_call(S, K, R, Q, TAU), call);
    assert_eq!(model.price_put(S, K, R, Q, TAU), put);
  }

  /// The trait's vanilla put-call parity is wrong here — the averaged
  /// underlying carries at $e^{(b-r)\tau}$, not $e^{-q\tau}$.
  #[test]
  fn asian_price_put_overrides_vanilla_parity() {
    let model = AsianPricer::new(V);
    let (call, put) = (
      model.price_call(S, K, R, Q, TAU),
      model.price_put(S, K, R, Q, TAU),
    );

    let vanilla = call - S * (-Q * TAU).exp() + K * (-R * TAU).exp();
    assert!(
      (put - vanilla).abs() > 1e-3,
      "the default would be a silent mispricing: put {put}, default {vanilla}"
    );

    let (v, b) = model.averaged_vol_and_carry(R, Q);
    assert!(v > 0.0);
    let generalised = call - S * ((b - R) * TAU).exp() + K * (-R * TAU).exp();
    assert!(
      (put - generalised).abs() < TOL,
      "put {put} vs {generalised}"
    );
  }

  /// The capability the reshape exists for: one model, a whole grid.
  #[test]
  fn asian_one_model_prices_a_grid() {
    let model = AsianPricer::new(V);
    for &tau in &[0.25, 0.5, 1.0] {
      let mut prev = f64::INFINITY;
      for &k in &[90.0, 100.0, 110.0] {
        let c = model.price_call(S, k, R, Q, tau);
        assert!(c.is_finite() && c < prev, "call must fall in strike");
        prev = c;
      }
    }
  }

  /// A negative volatility flips the sign of `1/(σ√τ)` and so of `d₁`,
  /// while `d₂ = d₁ - σ√τ` moves the *other* way — the two cross, and the
  /// call comes back **negative**: `-4.4548…` at `σ = -0.25` against
  /// `+6.7783` at `σ = +0.25`. Not the price of anything, and finite enough
  /// to flow into a residual unnoticed.
  ///
  /// `σ = 0` is admissible and deliberately still accepted: the averaged
  /// underlying becomes deterministic and the closed form returns its
  /// discounted intrinsic forward value.
  #[test]
  #[should_panic(
    expected = "sigma must satisfy `sigma.is_finite() && sigma >= 0.0`, got sigma = -0.25"
  )]
  fn new_rejects_negative_volatility() {
    let _ = AsianPricer::new(-0.25);
  }

  #[test]
  #[should_panic(
    expected = "sigma must satisfy `sigma.is_finite() && sigma >= 0.0`, got sigma = NaN"
  )]
  fn new_rejects_nan_volatility() {
    let _ = AsianPricer::new(f64::NAN);
  }

  #[test]
  #[should_panic(
    expected = "sigma must satisfy `sigma.is_finite() && sigma >= 0.0`, got sigma = inf"
  )]
  fn new_rejects_an_infinite_volatility() {
    let _ = AsianPricer::new(f64::INFINITY);
  }

  #[test]
  fn new_accepts_the_zero_volatility_limit() {
    let deterministic = AsianPricer::new(0.0);
    assert!(deterministic.price_call(S, K, R, Q, TAU).is_finite());
  }
}
