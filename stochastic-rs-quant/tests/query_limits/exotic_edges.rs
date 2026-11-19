//! The exotic closed forms at the edges of their domains.

use stochastic_rs_quant::OptionType;
use stochastic_rs_quant::pricing::barrier::DoubleBarrierPricer;
use stochastic_rs_quant::pricing::exotic::DoubleBarrierOption;
use stochastic_rs_quant::pricing::heston::HestonParams;
use stochastic_rs_quant::pricing::variance_swap::VarianceSwapPricer;
use stochastic_rs_quant::pricing::variance_swap::VolatilitySwapPricer;

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
