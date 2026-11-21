//! At `lambda = 0` the model is Black-Scholes at `sigma`: only the jump variance rate `λz² = σ²γ`
//! enters, `0` here, so the price never forms `0·∞`; a value at the point, not the `λ → 0⁺` limit.

use super::*;

/// A `lambda = 0` price is finite for every `gamma`, including `gamma = 0`.
#[test]
fn merton_price_lambda_zero_equals_bs() {
  let bs = BsmPricer::new(0.2, BsmCoc::Bsm1973);
  let want = CallPut {
    call: bs.price_call(S, K, R, Q, TAU),
    put: bs.price_put(S, K, R, Q, TAU),
  };
  for &gamma in &[0.0, 0.4, 1.0] {
    let m = merton(0.0, gamma, 20);
    assert_eq!(m.price_call_put(S, K, R, Q, TAU), want, "gamma={gamma}");
    assert_eq!(
      m.price_call(S, K, R, Q, TAU),
      want.call,
      "gamma={gamma} call"
    );
    assert_eq!(m.price_put(S, K, R, Q, TAU), want.put, "gamma={gamma} put");
  }
}

/// The price (through the series) and the Greeks (through `greek_series`'s `λ ≤ 0` branch) agree on
/// the Black-Scholes value at `lambda = 0`.
#[test]
fn merton_price_and_greeks_agree_at_lambda_zero() {
  let m = merton(0.0, 0.4, 20);
  let bs = BsmPricer::new(m.sigma, m.b);
  assert_eq!(
    m.price_call(S, K, R, Q, TAU),
    bs.price_call(S, K, R, Q, TAU)
  );
  assert_eq!(
    m.delta(S, K, R, Q, TAU, OT),
    bs.delta(S, K, R, Q, TAU, OT),
    "delta"
  );
  // delta is the price's own derivative, so a price that had gone somewhere
  // else would have to show up here too.
  let h = S * 1e-4;
  let fd = (m.price_call(S + h, K, R, Q, TAU) - m.price_call(S - h, K, R, Q, TAU)) / (2.0 * h);
  assert!(
    (m.delta(S, K, R, Q, TAU, OT) - fd).abs() < 1e-6,
    "delta {} vs finite difference {fd}",
    m.delta(S, K, R, Q, TAU, OT)
  );
}

/// At fixed `gamma` the `λ → 0⁺` limit is Black-Scholes at `σ√(1 − γ)` (`z² = γσ²/λ → ∞`), not the
/// value at `lambda = 0`; the two coincide only at `gamma = 0`.
#[test]
fn the_lambda_zero_limit_is_discontinuous_in_gamma() {
  let at_zero = |gamma: f64| merton(0.0, gamma, 20).price_call(S, K, R, Q, TAU);
  let approaching = |gamma: f64, lambda: f64| merton(lambda, gamma, 40).price_call(S, K, R, Q, TAU);

  let diffusive_only =
    BsmPricer::new(0.2 * (1.0 - 0.4_f64).sqrt(), BsmCoc::Bsm1973).price_call(S, K, R, Q, TAU);
  assert!(
    (approaching(0.4, 1e-8) - diffusive_only).abs() < 1e-5,
    "limit {} should be Black-Scholes at v*sqrt(1-gamma) = {diffusive_only}",
    approaching(0.4, 1e-8)
  );
  assert!(
    (at_zero(0.4) - approaching(0.4, 1e-8)).abs() > 1.0,
    "the state and the limit must not have been quietly reconciled: {} vs {}",
    at_zero(0.4),
    approaching(0.4, 1e-8)
  );

  assert!(
    (at_zero(0.0) - approaching(0.0, 1e-8)).abs() < 1e-12,
    "at gamma = 0 the state and the limit must agree: {} vs {}",
    at_zero(0.0),
    approaching(0.0, 1e-8)
  );
}
