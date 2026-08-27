//! `lambda = 0` is the no-jump state, and the model there **is**
//! Black-Scholes at the total volatility `v`.
//!
//! The Poisson weights are `1, 0, 0, …`, so only the `n = 0` term survives,
//! and with no jumps the whole variance is diffusive: `d² = v² - λz² = v²`.
//! The Greeks have always answered that way. The price did not — it read
//! the per-jump size `z = √(v²γ/λ)` first, which is `∞` at `gamma > 0` and
//! `NaN` at `gamma = 0`, and `λz²` then became `0·∞ = NaN`. Only the jump
//! *variance rate* `λz²` ever enters the model, and that is `0` here
//! whatever the size of a jump that never happens would have been.
//!
//! `lambda = 0` is a value at a point rather than a limit, and the tests
//! below pin that distinction as well as the value.

use super::*;

/// The defect: every `lambda = 0` price was `NaN`, for every `gamma` —
/// including `gamma = 0`, where the model has no jump component to
/// parameterise at all.
#[test]
fn merton_price_lambda_zero_equals_bs() {
  let bs = BSMPricer::new(0.2, BSMCoc::Bsm1973);
  let want = bs.call_put(S, K, R, Q, TAU);
  for &gamma in &[0.0, 0.4, 1.0] {
    let m = merton(0.0, gamma, 20);
    assert_eq!(m.call_put(S, K, R, Q, TAU), want, "gamma={gamma}");
    assert_eq!(m.price_call(S, K, R, Q, TAU), want.0, "gamma={gamma} call");
    assert_eq!(m.price_put(S, K, R, Q, TAU), want.1, "gamma={gamma} put");
  }
}

/// Price and Greeks reach the Black-Scholes limit by different routes —
/// the price through the series, the Greeks through
/// `greek_series`'s `λ ≤ 0` branch — so this asserts they agree, which is
/// the property that was broken rather than either value on its own.
#[test]
fn merton_price_and_greeks_agree_at_lambda_zero() {
  let m = merton(0.0, 0.4, 20);
  let bs = BSMPricer::new(m.v, m.b);
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

/// The limit is **not** the value, and that is a property of the
/// parameterisation rather than of the model.
///
/// `gamma` is the share of the *total* variance carried by jumps, so
/// holding it fixed while `lambda → 0⁺` sets `z² = γv²/λ → ∞`: the jumps
/// get rarer and larger at once, keeping their share of the variance the
/// whole way down. Every one of those models has diffusive volatility
/// `v√(1-γ)`, and the surviving `n ≥ 1` terms carry weight `O(λ)`, so the
/// price converges to Black-Scholes at `v√(1-γ)` — `3.3205` here, against
/// `4.5817` at `lambda = 0` itself.
///
/// At `gamma = 0` there is no jump variance to strand and the two coincide,
/// which is the second half of the pin: the discontinuity is exactly the
/// variance the limit leaves behind.
#[test]
fn the_lambda_zero_limit_is_discontinuous_in_gamma() {
  let at_zero = |gamma: f64| merton(0.0, gamma, 20).price_call(S, K, R, Q, TAU);
  let approaching = |gamma: f64, lambda: f64| merton(lambda, gamma, 40).price_call(S, K, R, Q, TAU);

  let diffusive_only =
    BSMPricer::new(0.2 * (1.0 - 0.4_f64).sqrt(), BSMCoc::Bsm1973).price_call(S, K, R, Q, TAU);
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
