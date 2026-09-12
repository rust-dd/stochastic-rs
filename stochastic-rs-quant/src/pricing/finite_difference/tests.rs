use stochastic_rs_stochastic::K;
use stochastic_rs_stochastic::S0;

use super::*;

fn atm_pricer(style: OptionStyle, r#type: OptionType, method: FiniteDifferenceMethod) -> f64 {
  FiniteDifferencePricer::new(0.1, 10000, 250, style, method).price(S0, K, 0.05, 0.0, 1.0, r#type)
}

#[test]
fn eu_explicit_call() {
  let call = atm_pricer(
    OptionStyle::European,
    OptionType::Call,
    FiniteDifferenceMethod::Explicit,
  );
  assert!(call.is_finite() && call > 0.0);
}

#[test]
fn eu_implicit_call() {
  let call = atm_pricer(
    OptionStyle::European,
    OptionType::Call,
    FiniteDifferenceMethod::Implicit,
  );
  assert!(call.is_finite() && call > 0.0);
}

#[test]
fn eu_crank_nicolson_call() {
  let call = atm_pricer(
    OptionStyle::European,
    OptionType::Call,
    FiniteDifferenceMethod::CrankNicolson,
  );
  assert!(call.is_finite() && call > 0.0);
}

#[test]
fn am_explicit_call() {
  let call = atm_pricer(
    OptionStyle::American,
    OptionType::Call,
    FiniteDifferenceMethod::Explicit,
  );
  assert!(call.is_finite() && call > 0.0);
}

#[test]
fn am_implicit_call() {
  let call = atm_pricer(
    OptionStyle::American,
    OptionType::Call,
    FiniteDifferenceMethod::Implicit,
  );
  assert!(call.is_finite() && call > 0.0);
}

#[test]
fn am_crank_nicolson_call() {
  let call = atm_pricer(
    OptionStyle::American,
    OptionType::Call,
    FiniteDifferenceMethod::CrankNicolson,
  );
  assert!(call.is_finite() && call > 0.0);
}

#[test]
fn eu_explicit_put() {
  let put = atm_pricer(
    OptionStyle::European,
    OptionType::Put,
    FiniteDifferenceMethod::Explicit,
  );
  assert!(put.is_finite() && put > 0.0);
}

#[test]
fn eu_implicit_put() {
  let put = atm_pricer(
    OptionStyle::European,
    OptionType::Put,
    FiniteDifferenceMethod::Implicit,
  );
  assert!(put.is_finite() && put > 0.0);
}

#[test]
fn eu_crank_nicolson_put() {
  let put = atm_pricer(
    OptionStyle::European,
    OptionType::Put,
    FiniteDifferenceMethod::CrankNicolson,
  );
  assert!(put.is_finite() && put > 0.0);
}

#[test]
fn am_explicit_put() {
  let put = atm_pricer(
    OptionStyle::American,
    OptionType::Put,
    FiniteDifferenceMethod::Explicit,
  );
  assert!(put.is_finite() && put > 0.0);
}

#[test]
fn am_implicit_put() {
  let put = atm_pricer(
    OptionStyle::American,
    OptionType::Put,
    FiniteDifferenceMethod::Implicit,
  );
  assert!(put.is_finite() && put > 0.0);
}

#[test]
fn am_crank_nicolson_put() {
  let put = atm_pricer(
    OptionStyle::American,
    OptionType::Put,
    FiniteDifferenceMethod::CrankNicolson,
  );
  assert!(put.is_finite() && put > 0.0);
}

const S: f64 = 100.0;
const KK: f64 = 105.0;
const R: f64 = 0.05;
const TAU: f64 = 0.75;
const V: f64 = 0.25;

#[test]
fn european_schemes_match_black_scholes() {
  use crate::pricing::bsm::BSMCoc;
  use crate::pricing::bsm::BSMPricer;
  let bs = BSMPricer::new(V, BSMCoc::Merton1973);
  for method in [
    FiniteDifferenceMethod::Explicit,
    FiniteDifferenceMethod::Implicit,
    FiniteDifferenceMethod::CrankNicolson,
  ] {
    let model = FiniteDifferencePricer::new(V, 4000, 150, OptionStyle::European, method);
    for ot in [OptionType::Call, OptionType::Put] {
      let actual = model.price(S, KK, R, 0.0, TAU, ot);
      let expected = bs.price_option(S, KK, R, 0.0, TAU, ot);
      assert!(
        (actual - expected).abs() < 0.02,
        "{method:?}/{ot:?}: {actual} vs {expected}"
      );
    }
  }
}

#[test]
fn boundaries_discount_from_the_terminal_payoff() {
  for style in [OptionStyle::European, OptionStyle::American] {
    let model =
      FiniteDifferencePricer::new(V, 500, 100, style, FiniteDifferenceMethod::CrankNicolson);
    for option_type in [OptionType::Call, OptionType::Put] {
      let solve = FdSolve {
        model: &model,
        s: S,
        k: KK,
        r: 0.1,
        q: 0.02,
        tau: 2.0,
        option_type,
      };
      for elapsed in [0.0_f64, 0.5, 2.0] {
        let (spot, european) = match option_type {
          OptionType::Call => (
            300.0,
            300.0 * (-0.02 * elapsed).exp() - KK * (-0.1 * elapsed).exp(),
          ),
          OptionType::Put => (0.0, KK * (-0.1 * elapsed).exp()),
        };
        let expected = match style {
          OptionStyle::European => european,
          OptionStyle::American => european.max(solve.payoff(spot)),
        };
        assert!((solve.boundary_condition(spot, elapsed) - expected).abs() < 1e-12);
      }
    }
  }
}

/// The dividend-yield term this task added to the PDE actually moves the
/// price, in the direction and roughly the magnitude Black-Scholes says
/// it should. Without this, `q` could be threaded through and silently
/// ignored — the `pricing/slv.rs` failure mode.
#[test]
fn fd_dividend_yield_drives_the_price() {
  let model = FiniteDifferencePricer::new(
    V,
    500,
    100,
    OptionStyle::European,
    FiniteDifferenceMethod::CrankNicolson,
  );
  let no_div = model.price_call(S, KK, R, 0.0, TAU);
  let with_div = model.price_call(S, KK, R, 0.08, TAU);
  assert!(
    with_div < no_div - 1.0,
    "a large dividend yield must cut the call materially: {with_div} vs {no_div}"
  );

  use crate::pricing::bsm::BSMCoc;
  use crate::pricing::bsm::BSMPricer;
  let bs = BSMPricer::new(V, BSMCoc::Merton1973).price_call(S, KK, R, 0.08, TAU);
  assert!(
    (with_div - bs).abs() < 0.1,
    "European FD with q must track Black-Scholes with the same q: {with_div} vs {bs}"
  );
}

/// American exercise binds for a call once the dividend yield exceeds the
/// rate — the case that is unreachable without a `q` input, and the
/// reason the pre-query pricer's American and European calls were always
/// equal.
#[test]
fn fd_american_call_beats_european_under_dividends() {
  let eu = FiniteDifferencePricer::new(
    V,
    500,
    100,
    OptionStyle::European,
    FiniteDifferenceMethod::CrankNicolson,
  )
  .price_call(S, KK, 0.03, 0.10, TAU);
  let am = FiniteDifferencePricer::new(
    V,
    500,
    100,
    OptionStyle::American,
    FiniteDifferenceMethod::CrankNicolson,
  )
  .price_call(S, KK, 0.03, 0.10, TAU);
  assert!(am > eu, "american {am} must exceed european {eu}");
}

/// The trait's European put-call parity is the wrong answer for an
/// American solve.
#[test]
fn fd_price_put_overrides_vanilla_parity() {
  let model = FiniteDifferencePricer::new(
    V,
    500,
    100,
    OptionStyle::American,
    FiniteDifferenceMethod::CrankNicolson,
  );
  let call = model.price_call(S, KK, R, 0.0, TAU);
  let put = model.price_put(S, KK, R, 0.0, TAU);
  let vanilla = call - S + KK * (-R * TAU).exp();
  assert!(
    put > vanilla + 1e-3,
    "American put must exceed the European-parity value: {put} vs {vanilla}"
  );
}

/// The capability the reshape exists for: one model, a whole grid.
#[test]
fn fd_one_model_prices_a_grid() {
  let model = FiniteDifferencePricer::new(
    V,
    200,
    80,
    OptionStyle::European,
    FiniteDifferenceMethod::CrankNicolson,
  );
  for &tau in &[0.25, 0.5, 1.0] {
    let mut prev = f64::INFINITY;
    for &k in &[90.0, 100.0, 110.0] {
      let c = model.price_call(S, k, R, 0.02, tau);
      assert!(c.is_finite() && c < prev, "call must fall in strike");
      prev = c;
    }
  }
}

/// `t_n` and `s_n` are grid *counts*, so zero is not a coarse grid — it
/// is no grid. `t_n = 0` skips the time loop and returns the payoff read
/// straight off the initial grid: `0.6667` for an ATM call, a small
/// finite number that looks exactly like a cheap short-dated option.
/// `s_n = 0` is louder — it underflows `s_n - 1` — but neither is a
/// price, and only one of the two announced itself.
#[test]
#[should_panic(expected = "FiniteDifferencePricer::new: t_n must be at least 1 (got 0)")]
fn new_rejects_zero_time_steps() {
  let _ = FiniteDifferencePricer::new(
    0.25,
    0,
    100,
    OptionStyle::European,
    FiniteDifferenceMethod::CrankNicolson,
  );
}

#[test]
#[should_panic(expected = "FiniteDifferencePricer::new: s_n must be at least 2 (got 0)")]
fn new_rejects_zero_price_steps() {
  let _ = FiniteDifferencePricer::new(
    0.25,
    500,
    0,
    OptionStyle::European,
    FiniteDifferenceMethod::CrankNicolson,
  );
}

/// The PDE coefficients use `v²`, so a negative volatility prices as its
/// own absolute value — the solver silently answers a question the caller
/// did not ask rather than the one they did.
#[test]
#[should_panic(
  expected = "FiniteDifferencePricer::new: v must be a non-negative volatility (got -0.25)"
)]
fn new_rejects_negative_volatility() {
  let _ = FiniteDifferencePricer::new(
    -0.25,
    500,
    100,
    OptionStyle::European,
    FiniteDifferenceMethod::CrankNicolson,
  );
}
