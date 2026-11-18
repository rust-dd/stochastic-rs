use super::*;
use crate::LossMetric;
use crate::OptionQuote;
use crate::OptionType;
use crate::QuantError;
use crate::calibration::CalibrationHistory;
use crate::calibration::CalibrationOptions;
use crate::pricing::bsm::BsmCoc;
use crate::pricing::bsm::BsmPricer;
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

/// SHOP on 2024-02-20, Agazzotti et al. (2025), Table 1: Feller holds and every value lies in the
/// default projection box.
const SHOP: HkdeParams = HkdeParams {
  v0: 0.176,
  kappa: 0.191,
  theta: 0.728,
  sigma_v: 0.194,
  rho: -0.718,
  lambda: 1.009,
  p_up: 0.958,
  eta1: 8.739,
  eta2: 0.733,
};

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
  assert!(fit.history.len() > 1, "{} entries", fit.history.len());
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
  let problem = super::loss::HkdeProblem::new(&calibrator, START).unwrap();
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

/// Eq. 13 at the carry `r − q`: a call priced at σ = 0.25 (K = 90, T = 1, r = 3 %, q = 2 %),
/// against mpmath's 50-digit `1/(S ψ(d₁) √T)`.
#[test]
fn a_weight_at_a_dividend_yield_inverts_at_the_carry_r_minus_q() {
  let quote = OptionQuote::new(90.0, 1.0, 15.51751499245581, OptionType::Call);
  let weight = super::loss::market_weight(100.0, 0.03, 0.02, &quote).unwrap();
  let reference = 0.029769392626231853;
  assert!(
    (weight / reference - 1.0).abs() < 1e-12,
    "{weight} against {reference}"
  );
}

/// At `q = 0` the carry `r − q` is `r` to the bit, so every weight is the one inverted at `r`.
#[test]
fn without_a_dividend_yield_each_weight_is_the_one_inverted_at_r() {
  let model = HkdeFourier::new(ref_hkde_params());
  let carry_r = BsmPricer::new(0.2, BsmCoc::Bsm1973);
  for tau in [0.25, 1.0] {
    for (k, side) in [
      (80.0, OptionType::Put),
      (95.0, OptionType::Put),
      (100.0, OptionType::Call),
      (120.0, OptionType::Call),
    ] {
      let price = model.price_option(100.0, k, 0.03, 0.0, tau, side);
      let quote = OptionQuote::new(k, tau, price, side);
      let sigma = carry_r.implied_volatility(price, 100.0, k, 0.03, 0.0, tau, side);
      let at_r = super::loss::weight_at(100.0, 0.03, 0.0, &quote, sigma).unwrap();
      let weight = super::loss::market_weight(100.0, 0.03, 0.0, &quote).unwrap();
      assert_eq!(weight.to_bits(), at_r.to_bits(), "K = {k}, T = {tau}");
    }
  }
}

/// A zero price has no implied volatility, and σ = 0.3 at K = 400 leaves a vega of about 1e-17.
#[test]
fn a_quote_the_weights_cannot_use_stops_the_fit_before_it_starts() {
  let far = |price: f64| OptionQuote::new(400.0, 0.25, price, OptionType::Call);
  let tiny = BsmPricer::new(0.3, BsmCoc::Merton1973).price_call(100.0, 400.0, 0.05, 0.0, 0.25);
  for (quote, reason) in [
    (far(0.0), "no Black volatility above 1e-8 reprices"),
    (far(tiny), "its vega"),
  ] {
    let mut quotes = heston_ref_quotes();
    quotes.push(quote);
    let calibrator = HkdeCalibrator::new(100.0, 0.05, 0.0, quotes).with_start(START);
    match calibrator.calibrate(None) {
      Err(QuantError::InvalidQuote {
        index: 9,
        reason: got,
      }) => {
        assert!(got.starts_with(reason), "{got}")
      }
      other => panic!("{other:?}"),
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
fn the_published_shop_set_is_admissible() {
  let original = SHOP;
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
fn the_published_shop_set_is_recovered_from_its_own_prices() {
  let truth = SHOP;
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

#[test]
fn the_jacobian_is_the_slope_of_the_residuals_across_the_box() {
  use super::loss::HkdeProblem;
  use super::params::*;
  use crate::calibration::least_squares::LeastSquaresProblem;
  use crate::calibration::least_squares::tests::box_points;
  use crate::calibration::least_squares::tests::jacobian_error;
  let calibrator = HkdeCalibrator::new(100.0, 0.05, 0.0, heston_ref_quotes());
  let mut problem = HkdeProblem::new(&calibrator, START).unwrap();
  let bounds = [
    P_V0, P_KAPPA, P_THETA, P_SIGMA_V, P_RHO, P_LAMBDA, P_P_UP, P_ETA1, P_ETA2,
  ];
  let far = ndarray::array![0.2, 12.0, 0.3, 0.9, 0.6, 4.0, 0.2, 30.0, 25.0];
  for base in [problem.params(), ndarray::Array1::from(SHOP), far] {
    for (label, point) in box_points(&base, &bounds, &[true; 9]) {
      // Where Feller's projection binds the stencil runs from the projected point, and at
      // ρ = −0.98 the Gil–Pelaez price jumps inside it (the p_up column reads 2.4 times its slope).
      let capped = 2.0 * point[1] * point[2] < point[3].powi(2);
      if capped || label == "lower 4" {
        continue;
      }
      let error = jacobian_error(&mut problem, &point, |_, x| 2e-3 * x.abs().max(1e-2));
      assert!(error < 2e-2, "{label} of {base}: {error:e}");
    }
  }
}
