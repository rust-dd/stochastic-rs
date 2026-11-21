//! `BarrierPricer` on the deterministic path of a zero volatility.

use stochastic_rs_quant::OptionType;
use stochastic_rs_quant::pricing::barrier::BarrierPricer;
use stochastic_rs_quant::pricing::barrier::BarrierType;
use stochastic_rs_quant::pricing::barrier::McBarrierPricer;
use stochastic_rs_quant::pricing::exotic::BarrierOption;

const R: f64 = 0.05;
const Q: f64 = 0.02;
const NAN: f64 = f64::NAN;

/// At `sigma = 0` the path is `S e^{(r−q)t}`: it meets `H` at `t* = ln(H/S)/(r − q)` or never, so
/// the price is a discounted rebate or payoff, which the Monte Carlo on that one path repeats.
#[test]
fn a_barrier_at_zero_volatility_prices_its_deterministic_path() {
  use BarrierType::*;
  use OptionType::*;
  let (s, tau, rebate) = (100.0_f64, 1.0_f64, 3.0_f64);
  let steps = 4_000;
  for (barrier_type, option_type, k, h, r, q) in [
    (UpAndOut, Call, 95.0, 102.0, 0.05, 0.02),
    (UpAndOut, Call, 95.0, 120.0, 0.05, 0.02),
    (UpAndIn, Call, 95.0, 102.0, 0.05, 0.02),
    (UpAndIn, Put, 105.0, 120.0, 0.05, 0.02),
    (DownAndOut, Put, 105.0, 90.0, 0.05, 0.02),
    (DownAndOut, Call, 90.0, 97.0, 0.01, 0.06),
    (DownAndIn, Put, 100.0, 97.0, 0.01, 0.06),
    (DownAndIn, Call, 90.0, 80.0, 0.01, 0.06),
    (UpAndOut, Put, 105.0, 110.0, 0.03, 0.03),
  ] {
    let option = BarrierOption {
      strike: k,
      barrier: h,
      rebate,
      barrier_type,
      option_type,
    };
    let terminal = s * ((r - q) * tau).exp();
    let hit = if h > s { terminal >= h } else { terminal <= h };
    let knock_in = matches!(barrier_type, UpAndIn | DownAndIn);
    let payoff = match option_type {
      Call => (terminal - k).max(0.0),
      Put => (k - terminal).max(0.0),
    };
    let want = match (knock_in, hit) {
      (false, true) => rebate * (-r * (h / s).ln() / (r - q)).exp(),
      (true, false) => rebate * (-r * tau).exp(),
      _ => payoff * (-r * tau).exp(),
    };
    let closed = BarrierPricer::new(0.0).price(&option, s, r, q, tau);
    assert!(
      (closed - want).abs() < 1e-12,
      "{option:?}: {closed} vs {want}"
    );
    let mc = McBarrierPricer::new(0.0, 2, steps).price(&option, s, r, q, tau);
    // The grid sees the hit up to one step late, which moves the rebate's discount by `r dt`.
    let slack = if hit && !knock_in {
      rebate * r.abs() * tau / steps as f64
    } else {
      1e-10
    };
    assert!(
      (mc.mean - closed).abs() <= slack,
      "{option:?}: {} vs {closed}",
      mc.mean
    );
  }
  let call = BarrierOption {
    strike: NAN,
    barrier: 120.0,
    rebate,
    barrier_type: UpAndOut,
    option_type: Call,
  };
  assert!(BarrierPricer::new(0.0).price(&call, s, R, Q, tau).is_nan());
}

/// A knock-in already breached is its vanilla, whose forward `S e^{(r−q)τ}` sits at the strike
/// here: 72 cases, each worth the payoff `0` and not BSM's `0/0`.
#[test]
fn a_knocked_in_barrier_at_zero_volatility_with_its_forward_at_the_strike_is_worth_nothing() {
  use BarrierType::*;
  let (s, k, r, q) = (100.0_f64, 100.0_f64, 0.03_f64, 0.03_f64);
  let mut cases = 0;
  for (barrier_type, barriers) in [
    (UpAndIn, [80.0, 97.0, 99.0]),
    (DownAndIn, [101.0, 103.0, 120.0]),
  ] {
    for option_type in [OptionType::Call, OptionType::Put] {
      for barrier in barriers {
        for rebate in [0.0, 3.0] {
          for tau in [0.5, 1.0, 2.0] {
            let option = BarrierOption {
              strike: k,
              barrier,
              rebate,
              barrier_type,
              option_type,
            };
            let closed = BarrierPricer::new(0.0).price(&option, s, r, q, tau);
            let mc = McBarrierPricer::new(0.0, 1, 2000).price(&option, s, r, q, tau);
            assert_eq!((closed, mc.mean), (0.0, 0.0), "{option:?} at tau = {tau}");
            cases += 1;
          }
        }
      }
    }
  }
  assert_eq!(cases, 72);
}

/// A breached knock-in off the strike, with a drift that carries the path back across the barrier.
#[test]
fn a_knocked_in_barrier_at_zero_volatility_stays_knocked_in() {
  let (s, tau, r, q) = (100.0_f64, 1.0_f64, 0.01_f64, 0.06_f64);
  let option = BarrierOption {
    strike: 90.0,
    barrier: 97.0,
    rebate: 3.0,
    barrier_type: BarrierType::UpAndIn,
    option_type: OptionType::Call,
  };
  let terminal = s * ((r - q) * tau).exp();
  assert!(terminal < option.barrier);
  let want = (terminal - 90.0) * (-r * tau).exp();
  let closed = BarrierPricer::new(0.0).price(&option, s, r, q, tau);
  assert!((closed - want).abs() < 1e-12, "{closed} vs {want}");
  let mc = McBarrierPricer::new(0.0, 1, 2000).price(&option, s, r, q, tau);
  assert!((mc.mean - closed).abs() < 1e-10, "{} vs {closed}", mc.mean);
  let nan_strike = BarrierOption {
    strike: NAN,
    ..option
  };
  let closed = BarrierPricer::new(0.0).price(&nan_strike, s, r, q, tau);
  assert!(closed.is_nan());
}

/// The strike rule does not depend on the volatility: a breached knock-in is its vanilla, NaN
/// before expiry at a strike that is not positive.
#[test]
fn a_knocked_in_barrier_is_nan_at_a_strike_that_is_not_positive_at_any_volatility() {
  for k in [0.0, -5.0] {
    let option = BarrierOption {
      strike: k,
      barrier: 97.0,
      rebate: 3.0,
      barrier_type: BarrierType::UpAndIn,
      option_type: OptionType::Call,
    };
    for sigma in [0.0, 0.2] {
      let price = BarrierPricer::new(sigma).price(&option, 100.0, R, Q, 1.0);
      assert!(price.is_nan(), "k = {k}, sigma = {sigma}: {price}");
      let expired = BarrierPricer::new(sigma).price(&option, 100.0, R, Q, 0.0);
      assert_eq!(expired, 100.0 - k, "k = {k}, sigma = {sigma}");
    }
  }
}
