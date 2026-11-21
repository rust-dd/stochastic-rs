//! Continuous geometric-average Asian option under Black–Scholes, by the law of the average.
//!
//! $$
//! \ln G\sim N\!\left(\ln S+\tfrac12\bigl(b-\tfrac12\sigma^2\bigr)\tau,\ \tfrac13\sigma^2\tau\right),
//! \qquad G=\exp\!\Bigl(\tfrac1\tau\int_0^\tau\ln S_t\,dt\Bigr),\qquad b=r-q
//! $$
//!
//! Reference: Kemna, A. & Vorst, A. (1990), "A Pricing Method for Options Based on Average Asset Values", J. Banking & Finance 14(1), 113–129

use stochastic_rs_distributions::special::norm_cdf;

use crate::OptionType;
use crate::pricing::digital::expired;
use crate::pricing::digital::vanilla_payoff;
use crate::traits::ModelPricer;

/// Geometric-average Asian option by Kemna and Vorst's closed form.
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

  /// The average's volatility `σ_A = σ/√3` and carry `b_A = ½(b − σ²/6)`, `b = r − q`.
  fn averaged_vol_and_carry(&self, r: f64, q: f64) -> (f64, f64) {
    let v = self.sigma / 3.0_f64.sqrt();
    let b = 0.5 * (r - q - self.sigma.powi(2) / 6.0);
    (v, b)
  }

  /// At `tau = 0` the payoff on the spot, the average with no averaging left; NaN past expiry.
  ///
  /// $$
  /// C=Se^{(b_A-r)\tau}N(d_1)-Ke^{-r\tau}N(d_2),\qquad
  /// d_1=\frac{\ln(S/K)+(b_A+\tfrac12\sigma_A^2)\tau}{\sigma_A\sqrt\tau},\qquad
  /// d_2=d_1-\sigma_A\sqrt\tau
  /// $$
  fn price(&self, s: f64, k: f64, r: f64, q: f64, tau: f64, option_type: OptionType) -> f64 {
    if let Some(value) = expired(tau, &[r, q], || vanilla_payoff(option_type, s, k)) {
      return value;
    }
    let (v, b) = self.averaged_vol_and_carry(r, q);
    let (carried, strike) = (s * ((b - r) * tau).exp(), k * (-r * tau).exp());
    let spread = v * tau.sqrt();
    if spread == 0.0 {
      return vanilla_payoff(option_type, carried, strike);
    }
    let d1 = ((s / k).ln() + (b + 0.5 * v.powi(2)) * tau) / spread;
    let d2 = d1 - spread;
    match option_type {
      OptionType::Call => carried * norm_cdf(d1) - strike * norm_cdf(d2),
      OptionType::Put => strike * norm_cdf(-d2) - carried * norm_cdf(-d1),
    }
  }
}

impl ModelPricer for AsianPricer {
  fn price_call(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    self.price(s, k, r, q, tau, OptionType::Call)
  }

  /// Not the parity default: the average carries at $e^{(b_A-r)\tau}$, so parity reads
  /// $C-P=Se^{(b_A-r)\tau}-Ke^{-r\tau}$.
  fn price_put(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    self.price(s, k, r, q, tau, OptionType::Put)
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

  /// Reference: the module's law of `ln G` integrated by mpmath at 50 digits, 3.15336604657145 and
  /// 7.25938857930441.
  #[test]
  fn asian_matches_the_law_of_the_average_at_the_golden_point() {
    let model = AsianPricer::new(V);
    let (call, put) = (
      model.price_call(S, K, R, Q, TAU),
      model.price_put(S, K, R, Q, TAU),
    );
    assert!((call - 3.1533660465714504).abs() < TOL, "call {call}");
    assert!((put - 7.259388579304415).abs() < TOL, "put {put}");
  }

  /// The trait's vanilla parity is wrong here: the average carries at $e^{(b_A-r)\tau}$.
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

  /// A negative volatility crosses `d₁` and `d₂` and prices a negative call.
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
