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

/// 7,098 and 9,744 are the counts of the walk before the head in `ln φ` and the tail rule, which
/// took 13,848 and 13,048; 3 % allows a panel that another libm splits.
#[test]
fn an_ordinary_price_costs_what_the_plain_walk_costs() {
  for (tau, plain) in [(1.0, 7_098), (0.02, 9_744)] {
    let count = evaluations(|pricer, k| pricer.price_call(100.0, k, 0.05, 0.02, tau));
    assert!(
      count.abs_diff(plain) * 100 <= 3 * plain,
      "tau = {tau}: {count} evaluations against {plain}"
    );
  }
}

/// 7,242 and 9,936 are the unsplit vega's counts at the same tolerance; the head and tails took
/// 14,520 and 15,256.
#[test]
fn an_ordinary_vega_costs_what_the_plain_walk_costs() {
  for (tau, plain) in [(1.0, 7_242), (0.02, 9_936)] {
    let count = evaluations(|pricer, k| pricer.initial_variance_vega(100.0, k, 0.05, 0.02, tau));
    assert!(
      count.abs_diff(plain) * 100 <= 3 * plain,
      "tau = {tau}: {count} evaluations against {plain}"
    );
  }
}

/// Past the control's cap, `E₁[∫v] = 3.7e4` here, both `P_j` take the sized walk: `f_1(1e-8)` is
/// then `1.8e-4` from 1, a head the plain walk's start would drop.
#[test]
fn a_capped_control_takes_the_head_in_ln_phi() {
  let pricer = HestonPricer::new(HestonParams {
    kappa: 0.5,
    sigma: 1.0,
    rho: 0.9,
    ..P
  });
  assert!(pricer.sized(30.0) && !pricer.sized(1.0));
  assert!(!HestonPricer::new(P).sized(1.0));
}
