// docs: processes#cheyette-quasi-gaussian-short-rate
//! Backs the Cheyette example on the processes page.

use stochastic_rs::simd_rng::Deterministic;
use stochastic_rs::stochastic::interest::cheyette::Cheyette;
use stochastic_rs::traits::ProcessExt;

fn flat_forward(_t: f64) -> f64 {
  0.03
}

fn displaced_vol(_t: f64, x: f64) -> f64 {
  0.01 + 0.3 * x
}

#[test]
fn cheyette_state_and_bond_reconstruction() {
  // Flat 3 % initial curve, κ = 0.5, displaced local volatility; one year on 100 steps.
  let model = Cheyette::<f64, _>::new(
    flat_forward as fn(f64) -> f64,
    0.5,
    displaced_vol as fn(f64, f64) -> f64,
    101,
    Some(1.0),
    Deterministic::new(7),
  );
  let [x, y] = model.sample();
  assert_eq!(x.len(), 101);
  assert_eq!((x[0], y[0]), (0.0, 0.0));
  // y accumulates the variance of x, so it stays positive along the path.
  assert!(y.iter().skip(1).all(|v| *v > 0.0));

  // Rates and bonds rebuild from the terminal state.
  let r_1 = model.short_rate(1.0, x[100]);
  let p_1_3 = model.zero_bond(1.0, 3.0, x[100], y[100]);
  assert!((r_1 - 0.03).abs() < 0.05 && p_1_3 > 0.8 && p_1_3 < 1.0);

  // At the origin the reconstruction is the initial curve.
  assert!((model.zero_bond(0.0, 2.0, 0.0, 0.0) - (-0.06_f64).exp()).abs() < 1e-12);
}
