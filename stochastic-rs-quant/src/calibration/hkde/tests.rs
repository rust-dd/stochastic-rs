use super::*;
use crate::LossMetric;
use crate::OptionQuote;
use crate::OptionType;
use crate::QuantError;
use crate::calibration::CalibrationHistory;
use crate::calibration::CalibrationOptions;
use crate::pricing::fourier::HkdeFourier;
use crate::traits::Calibrator;
use crate::traits::ModelPricer;
use crate::traits::ToModel;

/// Analytical reference Heston prices at
/// (v0=0.04, kappa=1.5, theta=0.04, sigma_v=0.3, rho=-0.7),
/// S=100, r=0.05, q=0, T=1.
const HESTON_REF: [f64; 9] = [
  25.095178, 20.976171, 17.106937, 13.548230, 10.361869, 7.604362, 5.317953, 3.519953, 2.193310,
];
const STRIKES: [f64; 9] = [80.0, 85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0, 120.0];

fn ref_hkde_params() -> HkdeParams {
  HkdeParams {
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

fn calls(strikes: &[f64], taus: &[f64], price: impl Fn(f64, f64) -> f64) -> Vec<OptionQuote> {
  taus
    .iter()
    .flat_map(|&tau| strikes.iter().map(move |&k| (k, tau)))
    .map(|(k, tau)| OptionQuote::new(k, tau, price(k, tau), OptionType::Call))
    .collect()
}

fn heston_ref_quotes() -> Vec<OptionQuote> {
  let price = |k: f64, _| HESTON_REF[STRIKES.iter().position(|&x| x == k).unwrap()];
  calls(&STRIKES, &[1.0], price)
}

const START: HkdeParams = HkdeParams {
  v0: 0.05,
  kappa: 1.8,
  theta: 0.05,
  sigma_v: 0.4,
  rho: -0.6,
  lambda: 0.05,
  p_up: 0.5,
  eta1: 10.0,
  eta2: 10.0,
};

#[test]
fn hkde_calibrate_recovers_heston_prices() {
  let result = HkdeCalibrator::new(100.0, 0.05, 0.0, heston_ref_quotes())
    .with_start(START)
    .calibrate(None)
    .unwrap();
  assert!(
    result.loss.get(LossMetric::Rmse) < 1.0,
    "Hkde→Heston RMSE={:.6}",
    result.loss.get(LossMetric::Rmse)
  );
}

#[test]
fn hkde_calibrate_self_consistency() {
  let truth = ref_hkde_params();
  let model = HkdeFourier::new(truth);

  let s_val = 100.0;
  let r = 0.03;
  let q = 0.0;
  let tau = 0.75;
  let market: Vec<f64> = STRIKES
    .iter()
    .map(|&k| model.price_call(s_val, k, r, q, tau).max(0.0))
    .collect();

  let initial = HkdeParams {
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

  let quotes = STRIKES
    .iter()
    .zip(market)
    .map(|(&k, price)| OptionQuote::new(k, tau, price, OptionType::Call))
    .collect();
  let result = HkdeCalibrator::new(s_val, r, q, quotes)
    .with_start(initial)
    .calibrate(None)
    .unwrap();
  assert!(
    result.loss.get(LossMetric::Rmse) < 1.0,
    "Hkde self-consistency RMSE={:.6}",
    result.loss.get(LossMetric::Rmse)
  );
}

#[test]
fn hkde_calibrate_multi_maturity() {
  let model = HkdeFourier::new(ref_hkde_params());
  let (r, q, s0) = (0.02, 0.0, 100.0);
  let quotes = calls(
    &[90.0, 95.0, 100.0, 105.0, 110.0],
    &[0.25, 0.5, 1.0],
    |k, tau| model.price_call(s0, k, r, q, tau).max(0.0),
  );
  let result = HkdeCalibrator::new(s0, r, q, quotes)
    .with_start(HkdeParams {
      v0: 0.05,
      kappa: 1.8,
      theta: 0.05,
      sigma_v: 0.25,
      rho: -0.5,
      lambda: 1.5,
      p_up: 0.5,
      eta1: 12.0,
      eta2: 6.0,
    })
    .calibrate(None)
    .unwrap();
  assert!(
    result.loss.get(LossMetric::Rmse) < 1.0,
    "Hkde multi-maturity RMSE={:.6}",
    result.loss.get(LossMetric::Rmse)
  );
}

/// The trail opens at the projected start and records each accepted step's residuals as
/// `√w · (market − model)`, the fit's own rows, so its objective falls to the fit.
#[test]
fn the_trail_records_weighted_market_minus_model() {
  let quotes = heston_ref_quotes();
  let calibrator = HkdeCalibrator::new(100.0, 0.05, 0.0, quotes.clone())
    .with_start(START)
    .with_options(
      CalibrationOptions::default()
        .with_record_history(true)
        .with_max_iterations(6),
    );
  let fit = calibrator.calibrate(None).unwrap();
  assert_eq!(fit.history[0].params, START.projected());
  assert_eq!(
    fit.history.last().map(|entry| entry.params),
    Some(fit.params)
  );
  let objective = |entry: &CalibrationHistory<HkdeParams>| entry.residuals.dot(&entry.residuals);
  assert!(
    fit
      .history
      .windows(2)
      .all(|pair| objective(&pair[1]) < objective(&pair[0]))
  );
  let problem = super::loss::HkdeProblem::new(&calibrator, START);
  for entry in &fit.history {
    let fit_rows = problem.weighted_residuals_for(&entry.params);
    for (i, quote) in quotes.iter().enumerate() {
      let model = HkdeFourier::new(entry.params).price_call(100.0, quote.strike, 0.05, 0.0, 1.0);
      let weight = fit_rows[i] / (model - quote.price);
      assert_eq!(entry.residuals[i], -fit_rows[i]);
      assert!(
        weight.is_finite() && weight > 0.0 && (weight - 1.0).abs() > 1e-3,
        "{weight}"
      );
    }
  }
}

#[test]
fn a_start_outside_the_hkde_domain_is_an_error() {
  let calibrator = HkdeCalibrator::new(100.0, 0.05, 0.0, heston_ref_quotes());
  for start in [
    HkdeParams { eta1: 0.9, ..START },
    HkdeParams {
      p_up: f64::NAN,
      ..START
    },
  ] {
    let err = calibrator.calibrate(Some(start)).unwrap_err();
    assert!(
      matches!(err, QuantError::InvalidInput { what: "start", .. }),
      "{err}"
    );
  }
}

#[test]
#[should_panic(expected = "eta2 must satisfy `eta2.is_finite() && eta2 > 0.0`, got eta2 = 0.0")]
fn with_start_panics_outside_the_hkde_domain() {
  let _ = HkdeCalibrator::new(100.0, 0.05, 0.0, heston_ref_quotes())
    .with_start(HkdeParams { eta2: 0.0, ..START });
}

#[test]
fn the_fit_converts_to_the_hkde_model_of_its_parameters() {
  let fit = HkdeCalibrator::new(100.0, 0.05, 0.0, heston_ref_quotes())
    .with_start(START)
    .with_options(CalibrationOptions::default().with_max_iterations(2))
    .calibrate(None)
    .unwrap();
  let model = fit.to_model();
  let p = fit.params;
  assert_eq!(
    [model.v0, model.kappa, model.theta, model.sigma_v, model.rho],
    [p.v0, p.kappa, p.theta, p.sigma_v, p.rho]
  );
  assert_eq!(
    [model.lam, model.p_up, model.eta1, model.eta2],
    [p.lambda, p.p_up, p.eta1, p.eta2]
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

  let model = HkdeFourier::new(truth);
  let market: Vec<f64> = strikes
    .iter()
    .map(|&k| model.price_call(s0, k, r, q, tau).max(0.0))
    .collect();

  let initial = HkdeParams {
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

  let quotes: Vec<OptionQuote> = strikes
    .iter()
    .zip(market)
    .map(|(&k, price)| OptionQuote::new(k, tau, price, OptionType::Call))
    .collect();
  let calibrator = HkdeCalibrator::new(s0, r, q, quotes);
  let result = calibrator.calibrate(Some(initial)).unwrap();
  assert!(
    result.loss.get(LossMetric::Rmse) < 0.5,
    "SHOP self-consistency RMSE={:.6}",
    result.loss.get(LossMetric::Rmse)
  );
  let refined = calibrator.calibrate(Some(result.params)).unwrap();
  assert!(
    refined.loss.get(LossMetric::Rmse) <= result.loss.get(LossMetric::Rmse) + 1e-6,
    "refit should not degrade the fit"
  );
}
