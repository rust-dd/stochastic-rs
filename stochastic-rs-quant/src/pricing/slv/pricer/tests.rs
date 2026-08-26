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
