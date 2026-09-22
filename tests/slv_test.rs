//! Tests for the Heston Stochastic Local Volatility (SLV) model.
//!
//! Validates the leverage calibration and the Monte Carlo pricer against
//! known limiting cases: a variance that never moves (`eta = 0` at the
//! long-run level) makes the leverage a closed form, and a flat local
//! volatility then makes the model Black–Scholes.

use ndarray::Array1;
use ndarray::Array2;
use stochastic_rs::quant::pricing::slv::HestonSlvParams;
use stochastic_rs::quant::pricing::slv::HestonSlvPricer;
use stochastic_rs::quant::pricing::slv::LeverageSurface;
use stochastic_rs::quant::pricing::slv::ParticleMethod;
use stochastic_rs::quant::pricing::slv::calibrate_leverage;
use stochastic_rs::traits::Grid2D;
use stochastic_rs::traits::ModelPricer;

fn heston_params(eta: f64) -> HestonSlvParams {
  HestonSlvParams {
    kappa: 2.0,
    theta: 0.04,
    sigma: 0.3,
    rho: -0.7,
    v0: 0.04,
    eta,
  }
}

fn flat_local_vol(vol: f64, spots: &Array1<f64>, times: &Array1<f64>) -> Grid2D<f64> {
  Grid2D::new(
    times.clone(),
    spots.clone(),
    Array2::from_elem((times.len(), spots.len()), vol),
  )
}

fn method(n: usize, seed: u64) -> ParticleMethod {
  ParticleMethod::default()
    .with_particles(n)
    .with_steps_per_year(50)
    .with_seed(seed)
}

#[test]
fn leverage_surface_interpolation() {
  let spots = Array1::from_vec(vec![80.0, 90.0, 100.0, 110.0, 120.0]);
  let times = Array1::from_vec(vec![0.25, 0.5, 1.0]);
  let mut values = Array2::ones((3, 5));
  values[[1, 2]] = 1.5;

  let surf = LeverageSurface::new(spots, times, values);

  assert!((surf.interpolate(100.0, 0.5) - 1.5).abs() < 1e-10);
  assert!((surf.interpolate(70.0, 0.25) - 1.0).abs() < 1e-10);
  assert!((surf.interpolate(130.0, 1.0) - 1.0).abs() < 1e-10);
  let v = surf.interpolate(95.0, 0.5);
  assert!(v > 1.0 && v < 1.5);
}

#[test]
fn sigma_mixed_computation() {
  let p = heston_params(0.5);
  assert!((p.sigma_mixed() - 0.15).abs() < 1e-10);
  let p0 = heston_params(0.0);
  assert!((p0.sigma_mixed()).abs() < 1e-10);
}

#[test]
fn calibrate_leverage_flat_vol_under_full_mixing_stays_near_one_at_the_money() {
  let params = heston_params(1.0);
  let spots = Array1::linspace(70.0, 130.0, 13);
  let times = Array1::from_vec(vec![0.1, 0.25, 0.5]);
  let lv = flat_local_vol(0.2, &spots, &times);

  let run = calibrate_leverage(&params, 100.0, 0.05, 0.0, &lv, &[0.25, 0.5], &method(5_000, 123)).unwrap();
  let leverage = &run.leverage;
  for &v in leverage.values().iter() {
    assert!(v.is_finite() && v > 0.0, "leverage must be finite and positive");
  }
  let l_atm = leverage.interpolate(100.0, 0.25);
  assert!(
    l_atm > 0.7 && l_atm < 1.4,
    "ATM leverage={l_atm} should stay near sigma_LV / sqrt(v0) = 1"
  );
}

/// With `eta = 0` and `v0 = theta` the variance is the constant `v0`, so the
/// calibration condition is the closed form `L = sigma_LV / sqrt(v0)`.
#[test]
fn calibrate_leverage_eta_zero_is_the_closed_form() {
  let params = heston_params(0.0);
  let spots = Array1::linspace(80.0, 120.0, 9);
  let times = Array1::from_vec(vec![0.1, 0.25]);
  let lv = flat_local_vol(0.25, &spots, &times);

  let run = calibrate_leverage(&params, 100.0, 0.05, 0.0, &lv, &[0.25], &method(2_000, 42)).unwrap();
  for &v in run.leverage.values().iter() {
    assert!((v - 1.25).abs() < 1e-9, "eta=0 leverage={v} should be 0.25 / 0.2");
  }
}

#[test]
fn slv_pricer_produces_positive_monotone_prices() {
  let params = heston_params(1.0);
  let spots = Array1::linspace(70.0, 130.0, 11);
  let times = Array1::from_vec(vec![0.1, 0.25, 0.5, 1.0]);
  let lv = flat_local_vol(0.2, &spots, &times);

  let run = calibrate_leverage(&params, 100.0, 0.05, 0.0, &lv, &[0.5, 1.0], &method(3_000, 99)).unwrap();
  let pricer = HestonSlvPricer::new(params, run.leverage, 0.05, 0.0)
    .with_paths(20_000)
    .with_steps_per_year(100)
    .with_seed(77);

  let c_itm = pricer.price_call(100.0, 90.0, 0.05, 0.0, 0.5);
  let c_atm = pricer.price_call(100.0, 100.0, 0.05, 0.0, 0.5);
  let c_otm = pricer.price_call(100.0, 120.0, 0.05, 0.0, 0.5);
  assert!(c_itm > c_atm && c_atm > c_otm && c_otm >= 0.0);
}

/// Flat local vol at `sqrt(v0)` under a variance that never moves is
/// Black–Scholes, so the pricer has to land on the Black price within its
/// own Monte Carlo error.
#[test]
fn slv_pricer_flat_vol_is_black_scholes() {
  let params = heston_params(0.0);
  let (s0, r, q, vol, tau, k) = (100.0, 0.05, 0.0, 0.2, 0.5, 100.0);
  let spots = Array1::linspace(60.0, 150.0, 19);
  let times = Array1::from_vec(vec![0.1, 0.3, 0.5]);
  let lv = flat_local_vol(vol, &spots, &times);

  let run = calibrate_leverage(&params, s0, r, q, &lv, &[0.5], &method(2_000, 42)).unwrap();
  assert!(run.leverage.values().iter().all(|l| (l - 1.0).abs() < 1e-12));
  let pricer = HestonSlvPricer::new(params, run.leverage, r, q)
    .with_paths(50_000)
    .with_steps_per_year(200)
    .with_seed(123);

  let estimate = pricer.price_call_estimate(s0, k, r, q, tau);
  let d1 = ((s0 / k).ln() + (r + 0.5 * vol * vol) * tau) / (vol * tau.sqrt());
  let d2 = d1 - vol * tau.sqrt();
  let bsm = s0 * normal_cdf(d1) - k * (-r * tau).exp() * normal_cdf(d2);
  assert!(
    (estimate.mean - bsm).abs() < 3.0 * estimate.std_err,
    "slv={} ± {}, bsm={bsm}",
    estimate.mean,
    estimate.std_err
  );
}

/// A surface calibrated on 70..130 carries no information about a spot of
/// 1000, but the bilinear clamp answers anyway; the pricer gates the query
/// on the surface's extent instead.
#[test]
fn slv_price_is_nan_outside_the_calibrated_spot_range() {
  let params = heston_params(1.0);
  let spots = Array1::linspace(70.0, 130.0, 11);
  let times = Array1::from_vec(vec![0.1, 0.25, 0.5]);
  let lv = flat_local_vol(0.2, &spots, &times);

  let run = calibrate_leverage(&params, 100.0, 0.05, 0.0, &lv, &[0.5], &method(2_000, 99)).unwrap();
  assert_eq!(run.leverage.spot_range(), (70.0, 130.0));
  let pricer = HestonSlvPricer::new(params, run.leverage, 0.05, 0.0)
    .with_paths(2_000)
    .with_steps_per_year(48)
    .with_seed(77);

  let inside = pricer.price_call(100.0, 100.0, 0.05, 0.0, 0.5);
  assert!(inside.is_finite() && inside > 0.0, "in-grid spot: {inside}");
  for s in [1000.0, 131.0, 10.0] {
    let out = pricer.price_call(s, s, 0.05, 0.0, 0.5);
    assert!(out.is_nan(), "s={s} is outside 70..130 but priced at {out}");
  }
}

/// The same hole on the maturity axis: past the horizon the leverage is the
/// last calibrated row held forward forever.
#[test]
fn slv_price_is_nan_beyond_the_calibrated_horizon() {
  let params = heston_params(1.0);
  let spots = Array1::linspace(70.0, 130.0, 11);
  let times = Array1::from_vec(vec![0.1, 0.25, 0.5]);
  let lv = flat_local_vol(0.2, &spots, &times);

  let run = calibrate_leverage(&params, 100.0, 0.05, 0.0, &lv, &[0.5], &method(2_000, 99)).unwrap();
  assert_eq!(run.leverage.horizon(), 0.5);
  let pricer = HestonSlvPricer::new(params, run.leverage, 0.05, 0.0)
    .with_paths(2_000)
    .with_steps_per_year(48)
    .with_seed(77);

  let at_horizon = pricer.price_call(100.0, 100.0, 0.05, 0.0, 0.5);
  assert!(at_horizon.is_finite() && at_horizon > 0.0);
  for tau in [0.51, 1.0, 5.0] {
    let out = pricer.price_call(100.0, 100.0, 0.05, 0.0, tau);
    assert!(out.is_nan(), "tau={tau} is past the 0.5 horizon but priced at {out}");
  }
}

fn normal_cdf(x: f64) -> f64 {
  stochastic_rs::distributions::special::norm_cdf(x)
}
