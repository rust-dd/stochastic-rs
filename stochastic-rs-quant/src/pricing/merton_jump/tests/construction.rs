//! `Merton1976Pricer::new` validates the two parameters that admit a value
//! the series cannot answer for.
//!
//! `m` is the Poisson-series length, and `m = 0` runs the loop zero times:
//! `call_put` returns **`(0.0, 0.0)`** — the crate's named anti-pattern, a
//! plausible-looking sentinel indistinguishable from a genuinely worthless
//! option.
//!
//! `v` is squared everywhere it is used in the price, so a negative
//! volatility silently prices as its own absolute value — `10.2113` at both
//! `v = ±0.2` for the one-year at-the-money call, an answer to a question
//! the caller did not ask. The Greeks are worse: `with_v_bump` floors the
//! bumped volatility at `1e-8`, so at `v = -0.2` both legs of the central
//! difference land on the floor and `vega` comes back **0.0**.
//!
//! `lambda` and `gamma` are deliberately **not** guarded. `lambda = 0` is a
//! supported state, not an invalid one — the Greeks collapse to plain
//! Black-Scholes there and `merton_greeks_lambda_zero_equals_bs` pins it —
//! and a `gamma` outside `[0, 1]` drives `v² - λz²` negative, which
//! announces itself as `NaN` rather than as a number.

use super::*;

#[test]
#[should_panic(expected = "Merton1976Pricer::new: m must be at least 1 (got 0)")]
fn new_rejects_an_empty_series() {
  let _ = merton(0.5, 0.4, 0);
}

#[test]
#[should_panic(expected = "Merton1976Pricer::new: v must be a non-negative volatility (got -0.2)")]
fn new_rejects_negative_volatility() {
  let _ = Merton1976Pricer::new(-0.2, 0.5, 0.4, 10, BSMCoc::Bsm1973);
}

#[test]
#[should_panic(expected = "Merton1976Pricer::new: v must be a non-negative volatility (got NaN)")]
fn new_rejects_nan_volatility() {
  let _ = Merton1976Pricer::new(f64::NAN, 0.5, 0.4, 10, BSMCoc::Bsm1973);
}

/// The supported states the guards must not swallow: the `λ = 0`
/// Black-Scholes limit, a single-term series, and a zero total
/// volatility.
#[test]
fn the_supported_degenerate_states_stay_constructible() {
  assert_eq!(merton(0.0, 0.4, 20).lambda, 0.0);
  assert_eq!(merton(0.5, 0.4, 1).m, 1);
  assert_eq!(
    Merton1976Pricer::new(0.0, 0.5, 0.4, 10, BSMCoc::Bsm1973).v,
    0.0
  );
}
