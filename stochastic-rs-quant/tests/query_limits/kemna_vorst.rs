//! The continuous geometric Asian against the law of its average.

use stochastic_rs_quant::pricing::asian::AsianPricer;
use stochastic_rs_quant::traits::ModelPricer;

const K: f64 = 100.0;

/// Reference: the law of `ln G` integrated by mpmath at 50 digits; the last row is the example of
/// Haug (2007) §4.20.1, whose put he prints as 4.6922.
#[test]
fn the_geometric_asian_is_kemna_and_vorsts_closed_form() {
  for (s, k, r, q, tau, sigma, call, put) in [
    (
      100.0,
      105.0,
      0.05,
      0.02,
      0.75,
      0.25,
      3.1533660465714504,
      7.259388579304415,
    ),
    (
      100.0,
      100.0,
      0.05,
      0.02,
      0.75,
      0.25,
      5.181078226380063,
      4.471128670508919,
    ),
    (
      110.0,
      100.0,
      0.03,
      0.0,
      2.0,
      0.4,
      18.307267848961416,
      8.543732688031188,
    ),
    (
      90.0,
      100.0,
      0.01,
      0.06,
      0.25,
      0.15,
      0.00620252926655567,
      10.58238965303835,
    ),
    (
      100.0,
      100.0,
      0.0,
      0.0,
      1.0,
      0.2,
      4.431893807918629,
      4.764672202466298,
    ),
    (
      80.0,
      85.0,
      0.05,
      -0.03,
      0.25,
      0.2,
      0.48188555462728,
      4.692221312245335,
    ),
  ] {
    let model = AsianPricer::new(sigma);
    let (got_call, got_put) = (
      model.price_call(s, k, r, q, tau),
      model.price_put(s, k, r, q, tau),
    );
    assert!(
      (got_call - call).abs() < 1e-12,
      "call at {s}, {k}, {tau}: {got_call} vs {call}"
    );
    assert!(
      (got_put - put).abs() < 1e-12,
      "put at {s}, {k}, {tau}: {got_put} vs {put}"
    );
  }
}

/// At `sigma = 0` the average is `S e^{bτ/2}`, so the price is its discounted intrinsic value.
#[test]
fn the_geometric_asian_at_zero_volatility_is_the_discounted_intrinsic_value() {
  let model = AsianPricer::new(0.0);
  let (r, q, tau) = (0.05_f64, 0.02_f64, 0.75_f64);
  let average = |s: f64| s * (0.5 * (r - q) * tau).exp();
  for s in [90.0, 100.0, 100.0 / (0.5 * (r - q) * tau).exp(), 110.0] {
    let want = (-r * tau).exp() * (average(s) - K);
    assert!(
      (model.price_call(s, K, r, q, tau) - want.max(0.0)).abs() < 1e-12,
      "call at {s}"
    );
    assert!(
      (model.price_put(s, K, r, q, tau) - (-want).max(0.0)).abs() < 1e-12,
      "put at {s}"
    );
  }
}
