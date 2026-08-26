use ndarray::Array1;
use ndarray::Array2;

use super::*;

fn params() -> HestonSlvParams {
  HestonSlvParams {
    kappa: 2.0,
    theta: 0.04,
    sigma: 0.3,
    rho: -0.7,
    v0: 0.04,
    eta: 1.0,
  }
}

/// $L \equiv 1$ — a pure Heston model, so the surface has no rate
/// provenance and every query rate is legitimate.
fn unit_leverage() -> LeverageSurface {
  LeverageSurface::new(
    Array1::from_vec(vec![50.0, 100.0, 150.0]),
    Array1::from_vec(vec![0.25, 0.5, 1.0]),
    Array2::ones((3, 3)),
  )
}

fn tune(pricer: HestonSlvPricer) -> HestonSlvPricer {
  pricer
    .with_paths(4_000)
    .with_steps_per_year(48)
    .with_seed(7)
}

#[test]
fn unanchored_price_call_responds_to_rate() {
  let pricer = tune(HestonSlvPricer::unanchored(params(), unit_leverage()));
  let low = pricer.price_call(100.0, 100.0, 0.05, 0.0, 0.5);
  let high = pricer.price_call(100.0, 100.0, 0.10, 0.0, 0.5);
  assert!(
    high > low,
    "call rho is positive: r=0.05 -> {low}, r=0.10 -> {high}"
  );
}

#[test]
fn unanchored_price_call_responds_to_dividend() {
  let pricer = tune(HestonSlvPricer::unanchored(params(), unit_leverage()));
  let no_div = pricer.price_call(100.0, 100.0, 0.05, 0.0, 0.5);
  let with_div = pricer.price_call(100.0, 100.0, 0.05, 0.05, 0.5);
  assert!(
    with_div < no_div,
    "a dividend yield must cheapen a call: q=0 -> {no_div}, q=0.05 -> {with_div}"
  );
}

#[test]
fn calibrated_pricer_accepts_its_own_rates() {
  let pricer = tune(HestonSlvPricer::new(params(), unit_leverage(), 0.05, 0.0));
  let c = pricer.price_call(100.0, 100.0, 0.05, 0.0, 0.5);
  assert!(
    c.is_finite() && c > 0.0,
    "call must be finite-positive: {c}"
  );
}

#[test]
#[should_panic(expected = "calibrated at r=0.05, q=0 but queried at r=0.1, q=0")]
fn calibrated_pricer_rejects_rate_mismatch() {
  let pricer = tune(HestonSlvPricer::new(params(), unit_leverage(), 0.05, 0.0));
  pricer.price_call(100.0, 100.0, 0.1, 0.0, 0.5);
}

#[test]
#[should_panic(expected = "calibrated at r=0.05, q=0 but queried at r=0.05, q=0.02")]
fn calibrated_pricer_rejects_dividend_mismatch() {
  let pricer = tune(HestonSlvPricer::new(params(), unit_leverage(), 0.05, 0.0));
  pricer.price_call(100.0, 100.0, 0.05, 0.02, 0.5);
}

#[test]
#[should_panic(expected = "calibrated at r=0.05, q=0 but queried at r=0.02, q=0")]
fn calibrated_pricer_rejects_mismatch_through_price_put() {
  let pricer = tune(HestonSlvPricer::new(params(), unit_leverage(), 0.05, 0.0));
  pricer.price_put(100.0, 100.0, 0.02, 0.0, 0.5);
}

/// `unit_leverage` spans `50 ..= 150` out to `tau = 1`. Every query below is
/// outside that box, and every one of them returned a finite, plausible,
/// in-band number before the gate: `69.01` at `s = 1000`, `0.345` at
/// `s = 5`, `29.05` at `tau = 5`.
#[test]
fn price_call_above_the_spot_grid_is_nan() {
  let pricer = tune(HestonSlvPricer::unanchored(params(), unit_leverage()));
  let c = pricer.price_call(1000.0, 1000.0, 0.05, 0.0, 0.5);
  assert!(c.is_nan(), "s = 1000 is past the 150 edge, got {c}");
}

#[test]
fn price_call_below_the_spot_grid_is_nan() {
  let pricer = tune(HestonSlvPricer::unanchored(params(), unit_leverage()));
  let c = pricer.price_call(5.0, 5.0, 0.05, 0.0, 0.5);
  assert!(c.is_nan(), "s = 5 is below the 50 edge, got {c}");
}

#[test]
fn price_call_beyond_the_horizon_is_nan() {
  let pricer = tune(HestonSlvPricer::unanchored(params(), unit_leverage()));
  let c = pricer.price_call(100.0, 100.0, 0.05, 0.0, 5.0);
  assert!(c.is_nan(), "tau = 5 is past the 1.0 horizon, got {c}");
}

/// The gate is a bound and not a "spot must equal `s0`" equality: a strike
/// ladder around the calibration spot is what the type is for, and the grid
/// edges themselves are inside.
#[test]
fn price_call_on_the_grid_boundary_still_prices() {
  let pricer = tune(HestonSlvPricer::unanchored(params(), unit_leverage()));
  for (s, tau) in [(50.0, 1.0), (150.0, 1.0), (100.0, 0.0625), (100.0, 1.0)] {
    let c = pricer.price_call(s, 100.0, 0.05, 0.0, tau);
    assert!(c.is_finite(), "({s}, {tau}) is inside the grid, got {c}");
  }
  for k in [60.0, 80.0, 100.0, 120.0, 140.0] {
    let c = pricer.price_call(100.0, k, 0.05, 0.0, 0.5);
    assert!(
      c.is_finite(),
      "the strike ladder never leaves the grid: k = {k} gave {c}"
    );
  }
}

#[test]
fn price_put_propagates_the_out_of_grid_nan() {
  let pricer = tune(HestonSlvPricer::unanchored(params(), unit_leverage()));
  let p = pricer.price_put(1000.0, 1000.0, 0.05, 0.0, 0.5);
  assert!(p.is_nan(), "parity carries the NaN through, got {p}");
}

/// `x = ln(s)` is `NaN` for a negative spot and `(NaN - k).max(0.0)` is
/// `0.0`, so this used to price at a confident zero. The extent gate catches
/// it on the way in, because no leverage grid starts at or below zero.
#[test]
fn a_negative_spot_no_longer_prices_as_zero() {
  let pricer = tune(HestonSlvPricer::unanchored(params(), unit_leverage()));
  let c = pricer.price_call(-100.0, 100.0, 0.05, 0.0, 0.5);
  assert!(c.is_nan(), "a negative spot must not price, got {c}");
  let nan_spot = pricer.price_call(f64::NAN, 100.0, 0.05, 0.0, 0.5);
  assert!(
    nan_spot.is_nan(),
    "a NaN spot must not price, got {nan_spot}"
  );
}

/// Same laundering on the other axis: `sqrt(dt)` is `NaN` for a negative
/// `tau`, and the `.max(0.0)` payoff floor turned that into `0.0` too.
#[test]
fn a_negative_maturity_no_longer_prices_as_zero() {
  let pricer = tune(HestonSlvPricer::unanchored(params(), unit_leverage()));
  let c = pricer.price_call(100.0, 100.0, 0.05, 0.0, -0.5);
  assert!(c.is_nan(), "a negative maturity must not price, got {c}");
}

/// `tau = 0` is the genuine expiry limit, not an edge: it is inside the
/// covered box and still returns intrinsic value.
#[test]
fn zero_maturity_stays_the_intrinsic_limit() {
  let pricer = tune(HestonSlvPricer::unanchored(params(), unit_leverage()));
  let c = pricer.price_call(100.0, 90.0, 0.05, 0.0, 0.0);
  assert!(
    (c - 10.0).abs() < 1e-9,
    "tau = 0 must still price intrinsic 10.0, got {c}"
  );
}

/// A query wrong on both counts panics rather than returning `NaN`: a rate
/// the pricer never agreed to is a wiring error, not a surface edge.
#[test]
#[should_panic(expected = "calibrated at r=0.05, q=0 but queried at r=0.2, q=0")]
fn the_rate_guard_outranks_the_extent_guard() {
  let pricer = tune(HestonSlvPricer::new(params(), unit_leverage(), 0.05, 0.0));
  pricer.price_call(1000.0, 1000.0, 0.2, 0.0, 5.0);
}
