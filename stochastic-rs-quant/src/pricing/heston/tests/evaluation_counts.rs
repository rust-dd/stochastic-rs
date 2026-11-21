use super::*;

/// The bench's strikes at `(s, r, q) = (100, 0.05, 0.02)`, ordinary parameters.
const STRIKES: [f64; 9] = [70.0, 80.0, 90.0, 95.0, 100.0, 105.0, 110.0, 120.0, 130.0];

fn evaluations(run: impl Fn(&HestonPricer, f64) -> f64) -> usize {
  let pricer = HestonPricer::new(P);
  EVALUATIONS.with(|n| n.set(0));
  for k in STRIKES {
    assert!(run(&pricer, k).is_finite());
  }
  EVALUATIONS.with(|n| n.get())
}

/// The head in `ln φ` and the two tails cost 1.95 and 1.34 times a walk from `1e-8` on these
/// strikes (7,098 and 9,744); 3 % allows a panel that another libm splits.
#[test]
fn a_price_keeps_its_evaluation_count() {
  for (tau, pinned) in [(1.0, 13_848), (0.02, 13_048)] {
    let count = evaluations(|pricer, k| pricer.price_call(100.0, k, 0.05, 0.02, tau));
    assert!(
      count.abs_diff(pinned) * 100 <= 3 * pinned,
      "tau = {tau}: {count} evaluations against {pinned}"
    );
  }
}

#[test]
fn a_vega_keeps_its_evaluation_count() {
  for (tau, pinned) in [(1.0, 14_520), (0.02, 15_256)] {
    let count = evaluations(|pricer, k| pricer.initial_variance_vega(100.0, k, 0.05, 0.02, tau));
    assert!(
      count.abs_diff(pinned) * 100 <= 3 * pinned,
      "tau = {tau}: {count} evaluations against {pinned}"
    );
  }
}

/// At `v0 = 1e-8`, `θ = 0`, `τ = 0.01` the control's `v₀`-slope grows to 80 times the integrand, so
/// each panel's rounding passes its target; mpmath's vega, Lewis's integral differentiated in `v₀`.
#[test]
fn a_vega_whose_panels_round_past_their_target_is_not_bisected_to_the_backstop() {
  let params = HestonParams {
    v0: 1e-8,
    theta: 0.0,
    ..P
  };
  EVALUATIONS.with(|n| n.set(0));
  let vega = HestonPricer::new(params).initial_variance_vega(100.0, 100.0, 0.05, 0.0, 0.01);
  let count = EVALUATIONS.with(|n| n.get());
  assert!((vega - 216.202_024_369_282_5).abs() < 2e-6, "{vega}");
  assert!(count < 60_000, "{count} evaluations");
}
