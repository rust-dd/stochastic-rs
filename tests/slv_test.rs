//! Tests for the Heston Stochastic Local Volatility (SLV) model.
//!
//! Validates leverage surface calibration and MC pricing against
//! known limiting cases:
//! - eta=0 (pure local vol): leverage should be ~1 everywhere
//! - Flat local vol: SLV call price ≈ BSM call price
//! - Leverage surface interpolation correctness

use ndarray::Array1;
use stochastic_rs::quant::pricing::slv::HestonSlvParams;
use stochastic_rs::quant::pricing::slv::HestonSlvPricer;
use stochastic_rs::quant::pricing::slv::LeverageSurface;
use stochastic_rs::quant::pricing::slv::calibrate_leverage;
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

fn flat_local_vol_grid(vol: f64, spots: &Array1<f64>, times: &Array1<f64>) -> ndarray::Array2<f64> {
  ndarray::Array2::from_elem((times.len(), spots.len()), vol)
}

#[test]
fn leverage_surface_interpolation() {
  let spots = Array1::from_vec(vec![80.0, 90.0, 100.0, 110.0, 120.0]);
  let times = Array1::from_vec(vec![0.25, 0.5, 1.0]);
  let mut values = ndarray::Array2::ones((3, 5));
  values[[1, 2]] = 1.5; // (t=0.5, S=100) = 1.5

  let surf = LeverageSurface::new(spots, times, values);

  // Exact grid point
  assert!((surf.interpolate(100.0, 0.5) - 1.5).abs() < 1e-10);

  // Boundary clamping
  assert!((surf.interpolate(70.0, 0.25) - 1.0).abs() < 1e-10);
  assert!((surf.interpolate(130.0, 1.0) - 1.0).abs() < 1e-10);

  // Interpolation between grid points: (95, 0.5) should be between 1.0 and 1.5
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
fn calibrate_leverage_flat_vol() {
  // With flat local vol = 0.2 and eta=1 (full stochastic vol), the leverage
  // function should adjust for the stochastic variance.
  let params = heston_params(1.0);
  let s0 = 100.0;
  let r = 0.05;
  let q = 0.0;

  let spots = Array1::linspace(70.0, 130.0, 13);
  let times = Array1::from_vec(vec![0.1, 0.25, 0.5]);
  let lv_grid = flat_local_vol_grid(0.2, &spots, &times);

  let leverage = calibrate_leverage(
    &params, s0, r, q, &spots, &times, &lv_grid, &spots, &times, 5000, 123,
  );

  // Leverage should be finite and positive everywhere
  for &v in leverage.values().iter() {
    assert!(v.is_finite(), "leverage must be finite");
    assert!(v > 0.0, "leverage must be positive");
  }

  // At ATM (S≈100) the leverage should be roughly σ_LV / sqrt(v0) = 0.2/0.2 = 1.0
  let l_atm = leverage.interpolate(100.0, 0.25);
  assert!(
    l_atm > 0.5 && l_atm < 2.0,
    "ATM leverage={l_atm} should be near 1.0"
  );
}

#[test]
fn calibrate_leverage_eta_zero_gives_near_unity() {
  // With eta=0 the variance is deterministic (no diffusion), so
  // E[V|S] ≈ v0 (for short T) and L ≈ σ_LV / sqrt(v0).
  // With σ_LV = sqrt(v0) = 0.2, we expect L ≈ 1.0.
  let params = heston_params(0.0);
  let s0 = 100.0;

  let spots = Array1::linspace(80.0, 120.0, 9);
  let times = Array1::from_vec(vec![0.1, 0.25]);
  let vol = params.v0.sqrt(); // 0.2
  let lv_grid = flat_local_vol_grid(vol, &spots, &times);

  let leverage = calibrate_leverage(
    &params, s0, 0.05, 0.0, &spots, &times, &lv_grid, &spots, &times, 5000, 42,
  );

  // All leverage values should be close to 1.0
  for &v in leverage.values().iter() {
    assert!(
      (v - 1.0).abs() < 0.3,
      "eta=0 leverage={v} should be near 1.0"
    );
  }
}

#[test]
fn slv_pricer_produces_positive_prices() {
  let params = heston_params(1.0);
  let s0 = 100.0;

  let spots = Array1::linspace(70.0, 130.0, 11);
  let times = Array1::from_vec(vec![0.1, 0.25, 0.5, 1.0]);
  let lv_grid = flat_local_vol_grid(0.2, &spots, &times);

  let leverage = calibrate_leverage(
    &params, s0, 0.05, 0.0, &spots, &times, &lv_grid, &spots, &times, 3000, 99,
  );

  let pricer = HestonSlvPricer::new(params, leverage, 0.05, 0.0)
    .with_paths(20_000)
    .with_steps_per_year(100)
    .with_seed(77);

  // ITM call
  let c_itm = pricer.price_call(100.0, 90.0, 0.05, 0.0, 0.5);
  assert!(c_itm > 0.0, "ITM call must be positive: {c_itm}");

  // ATM call
  let c_atm = pricer.price_call(100.0, 100.0, 0.05, 0.0, 0.5);
  assert!(c_atm > 0.0, "ATM call must be positive: {c_atm}");

  // OTM call — should be small but non-negative
  let c_otm = pricer.price_call(100.0, 120.0, 0.05, 0.0, 0.5);
  assert!(c_otm >= 0.0, "OTM call must be non-negative: {c_otm}");

  // Monotonicity: C(K=90) > C(K=100) > C(K=120)
  assert!(c_itm > c_atm, "call price must decrease in strike");
  assert!(c_atm > c_otm, "call price must decrease in strike");
}

#[test]
fn slv_pricer_flat_vol_vs_bsm_sanity() {
  // With flat local vol, eta=0 (pure LV), the SLV model should produce
  // prices close to BSM. This is a rough sanity check (MC noise).
  let params = heston_params(0.0);
  let s0 = 100.0;
  let r = 0.05;
  let q = 0.0;
  let vol = 0.2;
  let tau = 0.5;
  let k = 100.0;

  let spots = Array1::linspace(60.0, 150.0, 19);
  let times = Array1::from_vec(vec![0.05, 0.1, 0.2, 0.3, 0.4, 0.5]);
  let lv_grid = flat_local_vol_grid(vol, &spots, &times);

  let leverage = calibrate_leverage(
    &params, s0, r, q, &spots, &times, &lv_grid, &spots, &times, 10_000, 42,
  );

  let pricer = HestonSlvPricer::new(params, leverage, r, q)
    .with_paths(50_000)
    .with_steps_per_year(200)
    .with_seed(123);

  let slv_price = pricer.price_call(s0, k, r, q, tau);

  // BSM reference: C = S N(d1) - K e^{-rT} N(d2)
  let d1 = ((s0 / k).ln() + (r + 0.5 * vol * vol) * tau) / (vol * tau.sqrt());
  let d2 = d1 - vol * tau.sqrt();
  let bsm_price = s0 * normal_cdf(d1) - k * (-r * tau).exp() * normal_cdf(d2);

  // Allow generous tolerance for MC noise
  let rel_err = (slv_price - bsm_price).abs() / bsm_price;
  assert!(
    rel_err < 0.10,
    "SLV (eta=0, flat vol) should be within 10% of BSM: slv={slv_price:.4}, bsm={bsm_price:.4}, err={rel_err:.4}"
  );
}

fn normal_cdf(x: f64) -> f64 {
  stochastic_rs::distributions::special::norm_cdf(x)
}
