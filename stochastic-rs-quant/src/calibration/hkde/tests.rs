use super::*;
use crate::LossMetric;
use crate::OptionType;
use crate::pricing::fourier::HKDEFourier;
use crate::traits::ModelPricer;

/// Analytical reference Heston prices at
/// (v0=0.04, kappa=1.5, theta=0.04, sigma_v=0.3, rho=-0.7),
/// S=100, r=0.05, q=0, T=1.
const HESTON_REF: [f64; 9] = [
  25.095178, 20.976171, 17.106937, 13.548230, 10.361869, 7.604362, 5.317953, 3.519953, 2.193310,
];
const STRIKES: [f64; 9] = [80.0, 85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0, 120.0];

fn ref_hkde_params() -> HKDEParams {
  HKDEParams {
    v0: 0.04,
    kappa: 2.0,
    theta: 0.04,
    sigma_v: 0.3,
    rho: -0.7,
    lambda: 2.0,
    p_up: 0.4,
    eta1: 10.0,
    eta2: 5.0,
  }
}

#[test]
fn hkde_projection_enforces_eta1_above_one() {
  let mut p = ref_hkde_params();
  p.eta1 = 0.5;
  p.project_in_place();
  assert!(p.eta1 > 1.0, "eta1={} should be > 1", p.eta1);
}

#[test]
fn hkde_projection_enforces_feller() {
  let mut p = ref_hkde_params();
  p.kappa = 0.5;
  p.theta = 0.01;
  p.sigma_v = 0.9;
  p.project_in_place();
  assert!(
    2.0 * p.kappa * p.theta >= p.sigma_v * p.sigma_v - 1e-10,
    "Feller violated after projection: 2*k*theta={}, sigma_v^2={}",
    2.0 * p.kappa * p.theta,
    p.sigma_v * p.sigma_v
  );
}

#[test]
fn hkde_calibrate_recovers_heston_prices() {
  let n = STRIKES.len();
  let calibrator = HKDECalibrator::new(
    Some(HKDEParams {
      v0: 0.05,
      kappa: 1.8,
      theta: 0.05,
      sigma_v: 0.4,
      rho: -0.6,
      lambda: 0.05,
      p_up: 0.5,
      eta1: 10.0,
      eta2: 10.0,
    }),
    HESTON_REF.to_vec().into(),
    vec![100.0; n].into(),
    STRIKES.to_vec().into(),
    0.05,
    Some(0.0),
    1.0,
    OptionType::Call,
    false,
  );

  let result = calibrator.calibrate(None);
  assert!(
    result.loss.get(LossMetric::Rmse) < 1.0,
    "Hkde→Heston RMSE={:.6}",
    result.loss.get(LossMetric::Rmse)
  );
}

#[test]
fn hkde_calibrate_self_consistency() {
  let truth = ref_hkde_params();
  let model = HKDEFourier {
    v0: truth.v0,
    kappa: truth.kappa,
    theta: truth.theta,
    sigma_v: truth.sigma_v,
    rho: truth.rho,
    r: 0.03,
    q: 0.0,
    lam: truth.lambda,
    p_up: truth.p_up,
    eta1: truth.eta1,
    eta2: truth.eta2,
  };

  let s_val = 100.0;
  let r = 0.03;
  let q = 0.0;
  let tau = 0.75;
  let market: Vec<f64> = STRIKES
    .iter()
    .map(|&k| model.price_call(s_val, k, r, q, tau).max(0.0))
    .collect();

  let initial = HKDEParams {
    v0: 0.05,
    kappa: 1.5,
    theta: 0.05,
    sigma_v: 0.25,
    rho: -0.5,
    lambda: 1.0,
    p_up: 0.5,
    eta1: 8.0,
    eta2: 6.0,
  };

  let calibrator = HKDECalibrator::new(
    Some(initial),
    market.into(),
    vec![s_val; STRIKES.len()].into(),
    STRIKES.to_vec().into(),
    r,
    Some(q),
    tau,
    OptionType::Call,
    false,
  );
  let result = calibrator.calibrate(None);
  assert!(
    result.loss.get(LossMetric::Rmse) < 1.0,
    "Hkde self-consistency RMSE={:.6}",
    result.loss.get(LossMetric::Rmse)
  );
}

#[test]
fn hkde_calibrate_from_slices_multi_maturity() {
  use crate::calibration::levy::MarketSlice;

  let truth = ref_hkde_params();
  let r = 0.02;
  let q = 0.0;
  let s0 = 100.0;
  let make_model = || HKDEFourier {
    v0: truth.v0,
    kappa: truth.kappa,
    theta: truth.theta,
    sigma_v: truth.sigma_v,
    rho: truth.rho,
    r,
    q,
    lam: truth.lambda,
    p_up: truth.p_up,
    eta1: truth.eta1,
    eta2: truth.eta2,
  };
  let slice_strikes = vec![90.0, 95.0, 100.0, 105.0, 110.0];

  let make_slice = |t: f64| -> MarketSlice {
    let model = make_model();
    let prices: Vec<f64> = slice_strikes
      .iter()
      .map(|&k| model.price_call(s0, k, r, q, t).max(0.0))
      .collect();
    MarketSlice {
      strikes: slice_strikes.clone(),
      prices,
      is_call: vec![true; slice_strikes.len()],
      t,
    }
  };

  let slices = vec![make_slice(0.25), make_slice(0.5), make_slice(1.0)];

  let calibrator = HKDECalibrator::from_slices(
    Some(HKDEParams {
      v0: 0.05,
      kappa: 1.8,
      theta: 0.05,
      sigma_v: 0.25,
      rho: -0.5,
      lambda: 1.5,
      p_up: 0.5,
      eta1: 12.0,
      eta2: 6.0,
    }),
    &slices,
    s0,
    r,
    Some(q),
    OptionType::Call,
    false,
  );
  let result = calibrator.calibrate(None);
  assert!(
    result.loss.get(LossMetric::Rmse) < 1.0,
    "Hkde multi-maturity RMSE={:.6}",
    result.loss.get(LossMetric::Rmse)
  );
}

/// The SHOP parameter set from Agazzotti et al. (2025), Table 1, is fully
/// admissible. Check that the projection is a no-op on it: this guarantees
/// that the published Table 1 numbers can be used verbatim as inputs to
/// our calibrator / pricer without silent clipping.
#[test]
fn paper_table1_shop_is_admissible() {
  let original = paper_table1::SHOP;
  let projected = original.projected();
  let eps = 1e-12;
  assert!((projected.v0 - original.v0).abs() < eps);
  assert!((projected.kappa - original.kappa).abs() < eps);
  assert!((projected.theta - original.theta).abs() < eps);
  assert!((projected.sigma_v - original.sigma_v).abs() < eps);
  assert!((projected.rho - original.rho).abs() < eps);
  assert!((projected.lambda - original.lambda).abs() < eps);
  assert!((projected.p_up - original.p_up).abs() < eps);
  assert!((projected.eta1 - original.eta1).abs() < eps);
  assert!((projected.eta2 - original.eta2).abs() < eps);
  assert!(
    2.0 * projected.kappa * projected.theta >= projected.sigma_v * projected.sigma_v,
    "Feller should hold for SHOP"
  );
}

/// Self-consistency test against the published SHOP calibration
/// (Agazzotti et al. 2025, Table 1). We generate synthetic market prices
/// from the paper's SHOP parameters and verify the calibrator recovers
/// them starting from a perturbed guess.
#[test]
fn paper_table1_shop_self_consistency() {
  let truth = paper_table1::SHOP;
  let r = 0.05;
  let q = 0.0;
  let s0 = 100.0;
  let tau = 0.5;
  let strikes = [80.0_f64, 90.0, 100.0, 110.0, 120.0];

  let model = HKDEFourier {
    v0: truth.v0,
    kappa: truth.kappa,
    theta: truth.theta,
    sigma_v: truth.sigma_v,
    rho: truth.rho,
    r,
    q,
    lam: truth.lambda,
    p_up: truth.p_up,
    eta1: truth.eta1,
    eta2: truth.eta2,
  };
  let market: Vec<f64> = strikes
    .iter()
    .map(|&k| model.price_call(s0, k, r, q, tau).max(0.0))
    .collect();

  let initial = HKDEParams {
    v0: 0.12,
    kappa: 0.3,
    theta: 0.6,
    sigma_v: 0.25,
    rho: -0.55,
    lambda: 0.8,
    p_up: 0.9,
    eta1: 10.0,
    eta2: 1.0,
  };

  let calibrator = HKDECalibrator::new(
    Some(initial),
    market.clone().into(),
    vec![s0; strikes.len()].into(),
    strikes.to_vec().into(),
    r,
    Some(q),
    tau,
    OptionType::Call,
    false,
  );
  let result = calibrator.calibrate(None);
  assert!(
    result.loss.get(LossMetric::Rmse) < 0.5,
    "SHOP self-consistency RMSE={:.6}",
    result.loss.get(LossMetric::Rmse)
  );
  let refined = HKDECalibrator::new(
    Some(HKDEParams {
      v0: result.v0,
      kappa: result.kappa,
      theta: result.theta,
      sigma_v: result.sigma_v,
      rho: result.rho,
      lambda: result.lambda,
      p_up: result.p_up,
      eta1: result.eta1,
      eta2: result.eta2,
    }),
    market.into(),
    vec![s0; strikes.len()].into(),
    strikes.to_vec().into(),
    r,
    Some(q),
    tau,
    OptionType::Call,
    false,
  )
  .calibrate(None);
  assert!(
    refined.loss.get(LossMetric::Rmse) <= result.loss.get(LossMetric::Rmse) + 1e-6,
    "refit should not degrade the fit"
  );
}
