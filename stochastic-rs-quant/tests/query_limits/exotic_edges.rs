//! The exotic closed forms at the edges of their domains.

use stochastic_rs_quant::OptionType;
use stochastic_rs_quant::pricing::barrier::BarrierPricer;
use stochastic_rs_quant::pricing::barrier::BarrierType;
use stochastic_rs_quant::pricing::barrier::DoubleBarrierPricer;
use stochastic_rs_quant::pricing::bsm::BsmCoc;
use stochastic_rs_quant::pricing::bsm::BsmPricer;
use stochastic_rs_quant::pricing::chooser::ChooserPricer;
use stochastic_rs_quant::pricing::exotic::BarrierOption;
use stochastic_rs_quant::pricing::exotic::ChooserOption;
use stochastic_rs_quant::pricing::exotic::DoubleBarrierOption;
use stochastic_rs_quant::pricing::exotic::ForwardStartOption;
use stochastic_rs_quant::pricing::exotic::LookbackOption;
use stochastic_rs_quant::pricing::heston::HestonParams;
use stochastic_rs_quant::pricing::lookback::LookbackPricer;
use stochastic_rs_quant::pricing::variance_swap::VarianceSwapPricer;
use stochastic_rs_quant::pricing::variance_swap::VolatilitySwapPricer;
use stochastic_rs_quant::traits::ModelPricer;

const R: f64 = 0.05;
const Q: f64 = 0.02;
const SIGMA: f64 = 0.25;

fn heston(v0: f64, kappa: f64, theta: f64) -> HestonParams {
  HestonParams {
    v0,
    kappa,
    theta,
    sigma: 0.3,
    rho: -0.7,
  }
}

/// At expiry the barrier, double-barrier and lookback pricers pay out at a spot that is not
/// positive, as `BsmPricer`, the choosers and the compound do; before it that spot is NaN.
#[test]
fn an_exotic_at_expiry_pays_its_payoff_at_a_spot_that_is_not_positive() {
  use BarrierType::*;
  use OptionType::*;
  let barrier = BarrierPricer::new(SIGMA);
  let double = DoubleBarrierPricer::new(SIGMA);
  let lookback = LookbackPricer::new(SIGMA);
  for s in [0.0, -5.0] {
    for (barrier_type, h, option_type, at_expiry) in [
      (DownAndOut, 80.0, Call, 3.0),
      (DownAndIn, 80.0, Put, 100.0 - s),
      (UpAndOut, 130.0, Put, 100.0 - s),
      (UpAndIn, 130.0, Call, 3.0),
    ] {
      let option = BarrierOption {
        strike: 100.0,
        barrier: h,
        rebate: 3.0,
        barrier_type,
        option_type,
      };
      assert_eq!(
        barrier.price(&option, s, R, Q, 0.0),
        at_expiry,
        "{option:?} {s}"
      );
      assert!(
        barrier.price(&option, s, R, Q, 1.0).is_nan(),
        "{option:?} {s}"
      );
    }
    let corridor = DoubleBarrierOption {
      strike: 100.0,
      lower: 70.0,
      upper: 130.0,
      option_type: Put,
    };
    assert_eq!(double.price(&corridor, s, R, Q, 0.0), 0.0, "{s}");
    assert!(double.price(&corridor, s, R, Q, 1.0).is_nan(), "{s}");
    let fixed_put = LookbackOption {
      strike: Some(100.0),
      running_min: Some(90.0),
      running_max: None,
      option_type: Put,
    };
    assert_eq!(lookback.price(&fixed_put, s, R, Q, 0.0), 100.0 - s, "{s}");
    assert!(lookback.price(&fixed_put, s, R, Q, 1.0).is_nan(), "{s}");
  }
  let knock_in = BarrierOption {
    strike: 100.0,
    barrier: 130.0,
    rebate: 3.0,
    barrier_type: UpAndIn,
    option_type: Call,
  };
  assert!(barrier.price(&knock_in, f64::NAN, R, Q, 0.0).is_nan());
}

/// Deciding at 0 the holder takes the better leg at once, which the closed form reads as `0/0`
/// where the forward equals the strike.
#[test]
fn a_simple_chooser_deciding_at_0_takes_the_better_leg() {
  let chooser = ChooserPricer::new(SIGMA);
  let vanilla = BsmPricer::new(SIGMA, BsmCoc::Merton1973);
  let option = ChooserOption {
    strike: 100.0,
    choice_time: 0.0,
  };
  for (s, r, q) in [(100.0, 0.03, 0.03), (90.0, R, Q), (110.0, R, Q)] {
    let better = vanilla
      .price_call(s, 100.0, r, q, 1.0)
      .max(vanilla.price_put(s, 100.0, r, q, 1.0));
    assert_eq!(chooser.price_simple(&option, s, r, q, 1.0), better, "{s}");
  }
}

/// The forward start pays `(s − αs)⁺` at expiry and when struck at it, and is NaN for a start
/// outside `[0, tau]`, a NaN rate or a negative spot before expiry.
#[test]
fn the_forward_start_pays_out_at_its_edges_and_is_nan_outside_them() {
  let pricer = ChooserPricer::new(SIGMA);
  let started = |moneyness: f64, start_time: f64, option_type: OptionType| ForwardStartOption {
    moneyness,
    start_time,
    option_type,
  };
  for (moneyness, call, put) in [
    (1.0, 0.0, 0.0),
    (0.9, 10.0, 0.0),
    (1.1, 0.0, 10.000000000000014),
  ] {
    let option = started(moneyness, 0.0, OptionType::Call);
    assert_eq!(pricer.price_forward_start(&option, 100.0, R, Q, 0.0), call);
    let option = started(moneyness, 0.0, OptionType::Put);
    assert_eq!(pricer.price_forward_start(&option, 100.0, R, Q, 0.0), put);
  }
  let at_expiry = started(1.0, 1.0, OptionType::Call);
  assert_eq!(
    pricer.price_forward_start(&at_expiry, 100.0, R, Q, 1.0),
    0.0
  );
  let carried = 100.0 * ((R - Q - R) * 1.0_f64).exp();
  let struck_at_expiry = started(0.9, 1.0, OptionType::Call);
  let price = pricer.price_forward_start(&struck_at_expiry, 100.0, R, Q, 1.0);
  assert_eq!(price, carried - 0.9 * carried);
  for (start_time, r, s) in [
    (1.5, R, 100.0),
    (-0.25, R, 100.0),
    (f64::NAN, R, 100.0),
    (0.25, f64::NAN, 100.0),
    (0.25, R, -1.0),
  ] {
    let option = started(1.0, start_time, OptionType::Call);
    let price = pricer.price_forward_start(&option, s, r, Q, 1.0);
    assert!(price.is_nan(), "{start_time} {r} {s}: {price}");
  }
}

/// A lower bound that is not positive is NaN before expiry, and a corridor narrow against `σ√τ`
/// prices 0 without summing its images.
#[test]
fn the_double_barrier_at_the_edges_of_its_series() {
  let pricer = DoubleBarrierPricer::new(SIGMA);
  let mut option = DoubleBarrierOption {
    strike: 100.0,
    lower: 0.0,
    upper: 130.0,
    option_type: OptionType::Call,
  };
  assert!(pricer.price(&option, 100.0, R, Q, 1.0).is_nan());
  assert_eq!(pricer.price(&option, 110.0, R, Q, 0.0), 10.0);
  option.lower = 99.999;
  option.upper = 100.001;
  assert_eq!(pricer.price(&option, 100.0, R, Q, 1.0), 0.0);
}

/// `kappa * tau` is NaN at `kappa = 0` and `tau = ∞` and `(v0 − θ)·∞` at `v0 = θ`, where the
/// variance strike's limits are `v0` and `θ`.
#[test]
fn the_heston_variance_strike_takes_its_limit_where_the_product_has_none() {
  let strike =
    |v0, kappa, theta, tau| VarianceSwapPricer::fair_strike_heston(&heston(v0, kappa, theta), tau);
  assert_eq!(strike(0.04, 0.0, 0.04, f64::INFINITY), 0.04);
  assert_eq!(strike(0.09, 0.0, 0.04, f64::INFINITY), 0.04 + (0.09 - 0.04));
  assert_eq!(strike(0.04, -0.5, 0.04, 2000.0), 0.04);
  assert_eq!(strike(0.04, -0.5, 0.04, f64::INFINITY), 0.04);
  assert_eq!(strike(0.09, -0.5, 0.04, f64::INFINITY), f64::INFINITY);
  assert_eq!(strike(0.09, 1.5, 0.04, f64::INFINITY), 0.04);
}

/// The vol strike's correction vanishes with `tau` and is NaN once `Var(V)` passes `8K²`, where the
/// second-order expansion turns negative.
#[test]
fn the_heston_vol_strike_vanishes_at_short_maturities_and_is_nan_past_its_expansion() {
  for tau in [1e-8, 1e-6, 1e-4] {
    let strike = VolatilitySwapPricer::fair_strike_heston(&heston(0.09, 1.5, 0.04), tau);
    assert!((strike - 0.3).abs() < tau, "{tau}: {strike}");
  }
  let strike = VolatilitySwapPricer::fair_strike_heston(&heston(0.0, 1e-9, 0.04), 1.0);
  assert!(strike.is_nan(), "{strike}");
}
