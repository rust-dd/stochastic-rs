//! A `NaN` that is not a degenerate term now reaches the caller.
//!
//! `greek_series`'s floor is justified for one thing only: a term whose own
//! volatility `σ_n` is exactly zero, whose `1/v`-shaped closed forms are
//! `0/0` and whose `σ → 0⁺` limit really is `0` away from the forward. The
//! test that selects it used to be the bare `contribution.is_nan()`, which
//! is the crate's named laundering shape, and it caught far more than that.
//!
//! Every configuration below has a **`NaN` price** and used to have nine
//! finite Greeks, so price and Greeks disagreed about whether the query was
//! answerable — the same disagreement `an_inadmissible_gamma_still_announces_itself`
//! and item 24's `lambda = 0` split already pinned on the price side. The
//! `λ ≤ 0` branch of `greek_series` never laundered any of them, so the
//! disagreement was also internal to this file.

use super::*;

/// A model that is ordinary in every way, poisoned only by the query.
fn ordinary() -> Merton1976Pricer {
  merton(0.5, 0.4, 10)
}

/// `tau` is the one that arrives poisoned in normal operation:
/// `TimeExt::tau_or_from_dates` returns `NaN` for an expiry that never
/// resolved, and the crate has already closed the same route through the
/// Fourier pricers. An option whose expiry did not resolve reported
/// `delta = gamma = vega = rho = 0.0` — no risk at all — while its price
/// reported `NaN`.
#[test]
fn a_poisoned_query_poisons_the_greeks_and_not_only_the_price() {
  let m = ordinary();
  let cases: [(&str, f64, f64, f64, f64, f64); 9] = [
    ("tau", S, K, R, Q, f64::NAN),
    ("r", S, K, f64::NAN, Q, TAU),
    ("s", f64::NAN, K, R, Q, TAU),
    ("k", S, f64::NAN, R, Q, TAU),
    ("negative s", -S, K, R, Q, TAU),
    ("negative k", S, -K, R, Q, TAU),
    ("negative tau", S, K, R, Q, -TAU),
    ("zero tau", S, K, R, Q, 0.0),
    ("infinite tau", S, K, R, Q, f64::INFINITY),
  ];
  for (name, s, k, r, q, tau) in cases {
    let price = m.price_call(s, k, r, q, tau);
    assert!(
      price.is_nan(),
      "{name}: the price must already be NaN, else this pins nothing (got {price})"
    );
    let got = m.greeks(s, k, r, q, tau, OT).as_array();
    for (greek, v) in Greeks::COMPONENT_NAMES.iter().zip(got) {
      assert!(v.is_nan(), "{name}: {greek} must be NaN, got {v}");
    }
  }
}

/// `Merton1976Pricer::new` documents a `gamma` outside `[0, 1]` as
/// announcing itself as `NaN` rather than as a number. It announced itself
/// in the price — `an_inadmissible_gamma_still_announces_itself` pins that
/// — and reported a confident `0.0` in all nine Greeks. `σ_n` is `NaN`
/// there, not `0`, so the narrowed floor lets it through.
///
/// The `λ < 0` case is deliberately absent: `greek_series`'s `λ ≤ 0` branch
/// returns the Black-Scholes value there and never reaches the floor at
/// all, which item 24 recorded as the lesser of two evils and left alone.
#[test]
fn an_inadmissible_gamma_announces_itself_in_the_greeks_too() {
  for gamma in [-0.25, 1.5] {
    let m = merton(0.5, gamma, 20);
    assert!(
      m.price_call(S, K, R, Q, TAU).is_nan(),
      "gamma={gamma} price"
    );
    let got = m.greeks(S, K, R, Q, TAU, OT).as_array();
    for (greek, v) in Greeks::COMPONENT_NAMES.iter().zip(got) {
      assert!(v.is_nan(), "gamma={gamma}: {greek} must be NaN, got {v}");
    }
  }
}

/// The third route, and the one nothing else in the file covers: at
/// `λτ ≈ 5e8` the Poisson weight's running product overflows to `∞` while
/// `e^{-λτ}` underflows to `0`, so the weight itself is `0 · ∞ = NaN` with
/// no degenerate term anywhere in sight. `call_put` has no floor, so the
/// price was already `NaN`; the Greeks reported `0.0`.
///
/// One intensity short of that the weights are merely tiny and both sides
/// come back finite, so this is the boundary and not the whole regime.
#[test]
fn an_overflowing_poisson_weight_is_not_a_degenerate_term() {
  let m = Merton1976Pricer::new(0.2, 1.0e9, 0.4, 50, BSMCoc::Bsm1973);
  assert!(m.price_call(S, K, R, Q, TAU).is_nan(), "price");
  let got = m.greeks(S, K, R, Q, TAU, OT).as_array();
  for (greek, v) in Greeks::COMPONENT_NAMES.iter().zip(got) {
    assert!(v.is_nan(), "{greek} must be NaN, got {v}");
  }
  // The neighbouring finite regime still prices, so the narrowing did not
  // simply poison every large intensity.
  let tame = Merton1976Pricer::new(0.2, 1.0e3, 0.4, 50, BSMCoc::Bsm1973);
  assert!(tame.price_call(S, K, R, Q, TAU).is_finite());
  assert!(tame.delta(S, K, R, Q, TAU, OT).is_finite());
}
