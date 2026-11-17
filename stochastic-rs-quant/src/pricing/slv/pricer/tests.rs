use ndarray::Array1;
use ndarray::Array2;

use super::*;
use crate::pricing::heston::HestonParams;

fn params() -> HestonSlvParams {
  HestonSlvParams {
    heston: HestonParams {
      v0: 0.04,
      kappa: 2.0,
      theta: 0.04,
      sigma: 0.3,
      rho: -0.7,
    },
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
fn calibrated_pricer_answers_nan_at_another_rate() {
  let pricer = tune(HestonSlvPricer::new(params(), unit_leverage(), 0.05, 0.0));
  assert!(pricer.price_call(100.0, 100.0, 0.1, 0.0, 0.5).is_nan());
  let estimate = pricer.price_call_estimate(100.0, 100.0, 0.1, 0.0, 0.5);
  assert!(estimate.mean.is_nan() && estimate.std_err.is_nan() && estimate.n_samples == 0);
}

#[test]
fn calibrated_pricer_answers_nan_at_another_dividend_yield() {
  let pricer = tune(HestonSlvPricer::new(params(), unit_leverage(), 0.05, 0.0));
  assert!(pricer.price_call(100.0, 100.0, 0.05, 0.02, 0.5).is_nan());
}

#[test]
fn calibrated_pricer_answers_nan_at_another_rate_through_price_put() {
  let pricer = tune(HestonSlvPricer::new(params(), unit_leverage(), 0.05, 0.0));
  assert!(pricer.price_put(100.0, 100.0, 0.02, 0.0, 0.5).is_nan());
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

/// A negative spot fails the extent gate, since `x = ln(s)` is NaN and the payoff floor would turn
/// it into a confident zero.
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

#[test]
fn a_query_wrong_on_both_counts_is_nan() {
  let pricer = tune(HestonSlvPricer::new(params(), unit_leverage(), 0.05, 0.0));
  assert!(pricer.price_call(1000.0, 1000.0, 0.2, 0.0, 5.0).is_nan());
}

/// `L ≡ 1` under the full vol-of-vol is the Heston model, so the Monte Carlo
/// price has to land on the closed form: within three standard errors, plus
/// the weak-order-one bias of the Euler scheme at 250 steps a year (Lord,
/// Koekkoek & van Dijk 2010 put the absorption scheme's bias at this step
/// size in the low cents for these parameters).
#[test]
fn unit_leverage_reprices_the_closed_form_heston_call() {
  use crate::pricing::heston::HestonPricer;
  let p = params();
  let pricer = HestonSlvPricer::unanchored(p, unit_leverage())
    .with_paths(60_000)
    .with_steps_per_year(250)
    .with_seed(99);
  let (s, k, r, q, tau) = (100.0, 100.0, 0.03, 0.01, 0.5);
  let estimate = pricer.price_call_estimate(s, k, r, q, tau);
  let exact = HestonPricer::new(p.heston).price_call(s, k, r, q, tau);
  let band = 3.0 * estimate.std_err + 0.03;
  assert!(
    (estimate.mean - exact).abs() < band,
    "slv {} ± {} vs heston {exact}",
    estimate.mean,
    estimate.std_err
  );
}
