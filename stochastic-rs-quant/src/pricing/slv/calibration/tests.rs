use ndarray::Array1;
use ndarray::Array2;
use stochastic_rs_distributions::traits::Grid2D;

use super::*;

fn params(eta: f64) -> HestonSlvParams {
  HestonSlvParams {
    kappa: 2.0,
    theta: 0.04,
    sigma: 0.3,
    rho: -0.7,
    v0: 0.04,
    eta,
  }
}

/// A smile that bends in strike and drifts in time, so a leverage that
/// merely copied a constant would be caught.
fn local_vol(t: f64, k: f64) -> f64 {
  0.15 + 0.1 * (k / 100.0 - 1.0).powi(2) + 0.05 * t
}

fn local_vol_grid() -> Grid2D<f64> {
  let ts = Array1::from_vec(vec![0.1, 0.5, 1.0]);
  let ks = Array1::linspace(60.0, 140.0, 17);
  let values = Array2::from_shape_fn((ts.len(), ks.len()), |(j, i)| local_vol(ts[j], ks[i]));
  Grid2D::new(ts, ks, values)
}

fn method(n: usize) -> ParticleMethod {
  ParticleMethod::default()
    .with_particles(n)
    .with_steps_per_year(20)
    .with_seed(11)
}

/// With `eta = 0` the variance has no noise, and started at its long-run
/// level it never moves, so `E[V | S] = v0` exactly and the calibration
/// condition has the closed form `L = sigma_LV / sqrt(v0)` at every node
/// and every step — the whole estimator, kernel, window and bandwidth
/// included, has to return it.
#[test]
fn eta_zero_at_the_long_run_variance_gives_the_local_vol_over_sqrt_v0() {
  let grid = local_vol_grid();
  let run = calibrate_leverage(&params(0.0), 100.0, 0.02, 0.0, &grid, &[0.25, 1.0], &method(2_000)).unwrap();
  let lev = &run.leverage;
  assert_eq!(lev.times()[0], 0.0);
  assert_eq!(lev.horizon(), 1.0);
  for (j, &t) in lev.times().iter().enumerate() {
    for (i, &k) in lev.spots().iter().enumerate() {
      let expected = grid.eval(t, k) / 0.2;
      let got = lev.values()[[j, i]];
      assert!(
        (got - expected).abs() < 1e-9,
        "L({k}, {t}) = {got}, closed form {expected}"
      );
    }
  }
}

/// The `t = 0` row is the degenerate cloud's exact conditional expectation,
/// whatever the mixing.
#[test]
fn the_first_row_is_the_local_vol_over_sqrt_v0() {
  let grid = local_vol_grid();
  let run = calibrate_leverage(&params(1.0), 100.0, 0.02, 0.0, &grid, &[0.5], &method(500)).unwrap();
  for (i, &k) in run.leverage.spots().iter().enumerate() {
    assert_eq!(run.leverage.values()[[0, i]], grid.eval(0.0, k) / 0.2);
  }
}

/// Given the step's start, the log-spot increment is Gaussian with mean
/// `-½L²V dt` and variance `L²V dt`, so the discounted spot is a martingale
/// step by step, whatever leverage the cloud moves under: every snapshot's
/// mean is the forward, within its own standard error.
#[test]
fn snapshots_land_on_the_maturities_and_average_to_the_forward() {
  let grid = local_vol_grid();
  let (s0, r, q) = (100.0, 0.03, 0.01);
  let maturities = [0.1, 0.35, 1.0];
  let run = calibrate_leverage(&params(1.0), s0, r, q, &grid, &maturities, &method(20_000)).unwrap();
  assert_eq!(run.snapshots.len(), maturities.len());
  for (cloud, &tau) in run.snapshots.iter().zip(maturities.iter()) {
    assert_eq!(cloud.len(), 20_000);
    assert!(cloud.iter().all(|s| s.is_finite() && *s > 0.0));
    let mean = cloud.mean().unwrap();
    let std_err = cloud.std(0.0) / (cloud.len() as f64).sqrt();
    let forward = s0 * ((r - q) * tau).exp();
    assert!(
      (mean - forward).abs() < 4.0 * std_err,
      "at tau = {tau}: cloud mean {mean}, forward {forward}, standard error {std_err}"
    );
  }
  assert!(run.leverage.values().iter().all(|l| l.is_finite() && *l > 0.0));
}

#[test]
fn the_seed_pins_the_surface() {
  let grid = local_vol_grid();
  let run = |seed: u64| {
    calibrate_leverage(&params(1.0), 100.0, 0.02, 0.0, &grid, &[0.5], &method(1_000).with_seed(seed))
      .unwrap()
      .leverage
  };
  assert_eq!(run(3), run(3));
  assert_ne!(run(3), run(4));
}

#[test]
fn every_maturity_is_a_grid_point_and_intervals_are_split_evenly() {
  let times = time_grid(&[0.1, 0.25], 20);
  assert_eq!(times[0], 0.0);
  assert!(times.contains(&0.1) && times.contains(&0.25));
  assert_eq!(times.len(), 1 + 2 + 3);
  assert!(times.windows(2).all(|w| w[1] > w[0]));
  assert_eq!(time_grid(&[2.0], 1).len(), 3);
}

#[test]
fn bad_inputs_are_errors_not_panics() {
  let grid = local_vol_grid();
  let ok = |p: &HestonSlvParams, s0: f64, mats: &[f64], m: &ParticleMethod| {
    calibrate_leverage(p, s0, 0.02, 0.0, &grid, mats, m).map(|_| ())
  };
  assert!(ok(&params(1.0), 100.0, &[0.5], &method(2)).is_ok());
  assert!(ok(&params(1.0), 100.0, &[0.5], &method(1)).is_err());
  assert!(ok(&params(1.0), 100.0, &[], &method(2)).is_err());
  assert!(ok(&params(1.0), 100.0, &[0.5, 0.25], &method(2)).is_err());
  assert!(ok(&params(1.0), 100.0, &[0.0], &method(2)).is_err());
  assert!(ok(&params(1.0), -100.0, &[0.5], &method(2)).is_err());
  assert!(ok(&params(-0.5), 100.0, &[0.5], &method(2)).is_err());
  let mut nan = params(1.0);
  nan.rho = 1.5;
  assert!(ok(&nan, 100.0, &[0.5], &method(2)).is_err());
  let holed = Grid2D::new(
    Array1::from_vec(vec![0.5]),
    Array1::from_vec(vec![90.0, 110.0]),
    Array2::from_shape_vec((1, 2), vec![0.2, f64::NAN]).unwrap(),
  );
  assert!(calibrate_leverage(&params(1.0), 100.0, 0.0, 0.0, &holed, &[0.5], &method(2)).is_err());
}

#[test]
fn the_window_holds_the_kernel_mass_above_epsilon() {
  let h = 1.5;
  let half = window_half_width(h);
  let scaled = WINDOW_MASS * (2.0 * std::f64::consts::PI).sqrt() * h;
  let at_edge = (-0.5 * (half / h).powi(2)).exp();
  assert!((at_edge - scaled).abs() < 1e-12);
  assert!(window_half_width(1e6).is_infinite());
}

#[test]
fn gaps_take_the_nearest_estimate_and_an_empty_row_the_fallback() {
  let mut row = vec![f64::NAN, 0.03, f64::NAN, f64::NAN, 0.05, f64::NAN];
  fill_gaps(&mut row, 0.1);
  assert_eq!(row, vec![0.03, 0.03, 0.03, 0.05, 0.05, 0.05]);
  let mut empty = vec![f64::NAN; 3];
  fill_gaps(&mut empty, 0.1);
  assert_eq!(empty, vec![0.1; 3]);
}
