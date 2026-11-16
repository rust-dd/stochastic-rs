use ndarray::Array1;
use ndarray::s;
use stochastic_rs_quant::OptionType;
use stochastic_rs_quant::calibration::bsm::BsmCalibrator;
use stochastic_rs_quant::calibration::bsm::BsmParams;
use stochastic_rs_quant::calibration::hkde::HkdeCalibrator;
use stochastic_rs_quant::calibration::least_squares::LeastSquaresProblem;
use stochastic_rs_quant::pricing::bsm::BsmCoc;
use stochastic_rs_quant::pricing::bsm::BsmPricer;
use stochastic_rs_quant::traits::Calibrator;
use stochastic_rs_quant::traits::ModelPricer;

fn strided(values: impl IntoIterator<Item = f64>) -> Array1<f64> {
  Array1::from_iter(values.into_iter().flat_map(|v| [v, f64::NAN])).slice_move(s![..;2])
}

fn reversed(values: impl IntoIterator<Item = f64>) -> Array1<f64> {
  let values = values.into_iter().collect::<Vec<_>>();
  Array1::from_iter(values.into_iter().rev()).slice_move(s![..;-1])
}

#[test]
fn bsm_calibration_accepts_strided_quotes() {
  let strikes = strided([90.0, 100.0, 110.0]);
  let pricer = BsmPricer::new(0.25, BsmCoc::Bsm1973);
  let market = strided(
    strikes
      .iter()
      .map(|&k| pricer.price_call(100.0, k, 0.03, 0.0, 1.0)),
  );
  assert!(market.as_slice().is_none());
  let model = BsmCalibrator::new(
    BsmParams { v: 0.3 },
    market,
    strided([100.0; 3]),
    strikes,
    0.03,
    None,
    None,
    None,
    1.0,
    OptionType::Call,
  );
  let result = model.calibrate(None).unwrap();
  assert!((result.v - 0.25).abs() < 1e-6);
}

#[test]
fn bsm_calibration_preserves_reversed_quote_order() {
  let pricer = BsmPricer::new(0.25, BsmCoc::Bsm1973);
  let strikes = Array1::from_vec(vec![90.0, 100.0, 110.0]);
  let market = reversed(
    strikes
      .iter()
      .map(|&k| pricer.price_call(100.0, k, 0.03, 0.0, 1.0)),
  );
  assert!(market.as_slice().is_none());
  let model = BsmCalibrator::new(
    BsmParams { v: 0.3 },
    market,
    Array1::from_elem(3, 100.0),
    strikes,
    0.03,
    None,
    None,
    None,
    1.0,
    OptionType::Call,
  );
  let result = model.calibrate(None).unwrap();
  assert!((result.v - 0.25).abs() < 1e-6);
  assert!(result.loss.get(stochastic_rs_quant::LossMetric::Rmse) < 1e-6);
}

#[test]
fn hkde_weights_accept_strided_quotes() {
  let mut model = HkdeCalibrator::new(
    None,
    strided([15.0, 9.0, 4.0]),
    strided([100.0; 3]),
    strided([90.0, 100.0, 110.0]),
    0.03,
    None,
    1.0,
    OptionType::Call,
    true,
  );
  assert_eq!(model.c_market.len(), 3);
  assert!(model.residuals().unwrap().iter().all(|x| x.is_finite()));
}

#[test]
fn hkde_weights_preserve_reversed_quote_order() {
  let mut contiguous = HkdeCalibrator::new(
    None,
    Array1::from_vec(vec![15.0, 9.0, 4.0]),
    Array1::from_elem(3, 100.0),
    Array1::from_vec(vec![90.0, 100.0, 110.0]),
    0.03,
    None,
    1.0,
    OptionType::Call,
    true,
  );
  let mut reversed = HkdeCalibrator::new(
    None,
    reversed([15.0, 9.0, 4.0]),
    reversed([100.0; 3]),
    reversed([90.0, 100.0, 110.0]),
    0.03,
    None,
    1.0,
    OptionType::Call,
    true,
  );
  assert_eq!(reversed.sqrt_weights, contiguous.sqrt_weights);
  assert_eq!(reversed.residuals(), contiguous.residuals());
}

#[test]
fn the_calibration_core_types_have_the_unified_shape() {
  use stochastic_rs_quant::CalibrationLossScore;
  use stochastic_rs_quant::OptionQuote;
  use stochastic_rs_quant::QuantError;
  use stochastic_rs_quant::calibration::Calibrated;
  use stochastic_rs_quant::calibration::CalibrationHistory;
  use stochastic_rs_quant::calibration::CalibrationOptions;
  use stochastic_rs_quant::traits::CalibrationResult;
  let loss = CalibrationLossScore::compute(&[1.0, 2.0], &[1.0, 2.5]);
  let fit = Calibrated::new(0.25_f64, loss, true)
    .with_iterations(7)
    .with_message("converged")
    .with_history(vec![CalibrationHistory::new(
      0,
      0.3,
      ndarray::array![0.5],
      CalibrationLossScore::default(),
    )])
    .with_diagnostics("anything");
  assert_eq!(fit.params(), 0.25);
  assert_eq!(fit.iterations(), Some(7));
  assert_eq!(fit.message(), Some("converged"));
  assert_eq!(fit.max_error(), None);
  assert_eq!(fit.history().len(), 1);
  assert_eq!(fit.diagnostics, "anything");
  assert!((fit.rmse() - 0.125_f64.sqrt()).abs() < 1e-15);
  let options = CalibrationOptions::default()
    .with_max_iterations(50)
    .with_record_history(true);
  assert_eq!(options.max_iterations, Some(50));
  assert!(options.record_history && options.tolerance.is_none() && options.patience == 100);
  let quote = OptionQuote::new(f64::NAN, 1.0, 5.0, OptionType::Call);
  assert!(matches!(
    quote.validate(2),
    Err(QuantError::InvalidQuote { index: 2, .. })
  ));
}

#[test]
fn the_tree_calibrators_carry_their_fixed_inputs_in_params() {
  use stochastic_rs_quant::calibration::CalibrationOptions;
  use stochastic_rs_quant::calibration::hw_swaption::HullWhiteParams;
  use stochastic_rs_quant::calibration::hw_swaption::HullWhiteSwaptionCalibrator;
  use stochastic_rs_quant::calibration::hw_swaption::SwaptionQuote;
  use stochastic_rs_quant::calibration::tree_swaption::BlackKarasinskiParams;
  use stochastic_rs_quant::calibration::tree_swaption::BlackKarasinskiSwaptionCalibrator;
  use stochastic_rs_quant::calibration::tree_swaption::G2ppSwaptionCalibrator;
  use stochastic_rs_quant::curves::DiscountCurve;
  use stochastic_rs_quant::curves::InterpolationMethod;
  use stochastic_rs_quant::instruments::option::types::SwaptionDirection;
  use stochastic_rs_quant::traits::CalibrationResult;
  use stochastic_rs_quant::traits::ToShortRateModel;
  fn assert_send_sync<T: Send + Sync>() {}
  assert_send_sync::<HullWhiteSwaptionCalibrator<'static>>();
  assert_send_sync::<BlackKarasinskiSwaptionCalibrator<'static>>();
  assert_send_sync::<G2ppSwaptionCalibrator<'static>>();
  let curve = DiscountCurve::from_zero_rates(
    &Array1::from_vec(vec![0.5, 1.0, 5.0, 10.0]),
    &Array1::from_vec(vec![0.03; 4]),
    InterpolationMethod::LogLinearOnDiscountFactors,
  );
  let quotes =
    [(1.0, 2.0, 0.22), (2.0, 2.0, 0.20)].map(|(expiry, tenor, black_vol)| SwaptionQuote {
      expiry,
      tenor,
      black_vol,
      fixed_accrual: 0.5,
      direction: SwaptionDirection::Payer,
      weight: None,
    });
  let start = HullWhiteParams {
    mean_reversion: 0.05,
    sigma: 0.01,
    initial_rate: 0.03,
    theta: 0.03,
  };
  let fit = HullWhiteSwaptionCalibrator::new(&quotes, &curve, 1.0, start)
    .calibrate(None)
    .unwrap();
  assert!(fit.max_error.is_some() && fit.iterations.is_some() && fit.rmse().is_finite());
  let model = fit.to_short_rate_model();
  assert_eq!(model.initial_rate, 0.03);
  assert_eq!(model.sigma, fit.params.sigma);
  let other = HullWhiteSwaptionCalibrator::new(&quotes, &curve, 1.0, start)
    .calibrate(Some(HullWhiteParams {
      initial_rate: 0.02,
      ..start
    }))
    .unwrap();
  assert_eq!(
    other.params.initial_rate, 0.02,
    "a guess replaces the tree inputs too"
  );
  let bk = BlackKarasinskiSwaptionCalibrator::new(
    &quotes,
    &curve,
    1.0,
    8,
    BlackKarasinskiParams {
      mean_reversion: 0.1,
      sigma: 0.2,
      initial_rate: 0.03,
      long_run_rate: 0.03,
    },
  )
  .with_options(CalibrationOptions::default().with_max_iterations(200))
  .calibrate(None)
  .unwrap();
  assert_eq!(bk.to_short_rate_model().theta_log, 0.03_f64.ln());
}

#[test]
fn the_sabr_family_shares_the_shape_and_the_smile_implements_calibrator() {
  use stochastic_rs_quant::OptionQuote;
  use stochastic_rs_quant::QuantError;
  use stochastic_rs_quant::calibration::CalibrationOptions;
  use stochastic_rs_quant::calibration::sabr::SabrCalibrator;
  use stochastic_rs_quant::calibration::sabr_caplet::SabrCapletCalibrator;
  use stochastic_rs_quant::calibration::sabr_caplet::SabrCapletParams;
  use stochastic_rs_quant::pricing::sabr::SabrParams;
  use stochastic_rs_quant::pricing::sabr::SabrPricer;
  use stochastic_rs_quant::pricing::sabr::hagan_implied_vol;
  use stochastic_rs_quant::traits::CalibrationResult;
  use stochastic_rs_quant::vol_surface::sabr_smile::SabrSmileCalibrator;
  use stochastic_rs_quant::vol_surface::sabr_smile::SabrSmileQuotes;
  fn assert_send_sync<T: Send + Sync>() {}
  assert_send_sync::<SabrCalibrator>();
  assert_send_sync::<SabrCapletCalibrator>();
  assert_send_sync::<SabrSmileCalibrator>();
  let truth = SabrParams {
    alpha: 0.2,
    beta: 1.0,
    rho: -0.3,
    nu: 0.6,
  };
  let strikes = [80.0, 90.0, 95.0, 100.0, 105.0, 110.0, 120.0];
  let quotes: Vec<OptionQuote> = strikes
    .iter()
    .map(|&k| {
      let price = SabrPricer::new(truth).price_call(100.0, k, 0.01, 0.0, 1.0);
      OptionQuote::new(k, 1.0, price, OptionType::Call)
    })
    .collect();
  let fit = SabrCalibrator::new(100.0, 0.01, 0.0, quotes.clone())
    .calibrate(None)
    .unwrap();
  assert!((fit.params.nu - 0.6).abs() < 0.05 && fit.max_error.is_some());
  let mut bad = quotes;
  bad[2].strike = -1.0;
  assert!(matches!(
    SabrCalibrator::new(100.0, 0.01, 0.0, bad)
      .calibrate(None)
      .unwrap_err(),
    QuantError::InvalidQuote { index: 2, .. }
  ));
  let caplet_truth = SabrParams {
    alpha: 0.03,
    beta: 0.5,
    rho: -0.2,
    nu: 0.4,
  };
  let vols: Vec<f64> = strikes
    .iter()
    .map(|&k| hagan_implied_vol(k / 1000.0, 0.1, 2.0, &caplet_truth))
    .collect();
  let start = SabrCapletParams {
    sabr: SabrParams {
      alpha: 0.02,
      beta: 0.5,
      rho: 0.0,
      nu: 0.3,
    },
    shift: 0.0,
  };
  let caplet = SabrCapletCalibrator::new(
    0.1,
    2.0,
    strikes.iter().map(|k| k / 1000.0).collect(),
    vols,
    start,
  )
  .with_options(CalibrationOptions::default().with_tolerance(1e-14))
  .calibrate(None)
  .unwrap();
  assert!(
    caplet.rmse() < 1e-6 && (caplet.params.sabr.nu - 0.4).abs() < 1e-2,
    "rmse {}",
    caplet.rmse()
  );
  let smile = SabrSmileCalibrator::new(
    1.1,
    0.02,
    0.01,
    1.0,
    SabrSmileQuotes {
      tau: 1.0,
      sigma_atm: 0.1,
      sigma_rr: -0.01,
      sigma_bf: 0.003,
    },
  );
  let out = smile.calibrate(None).unwrap();
  assert!(out.converged && out.diagnostics.k_atm > 1.0 && out.params.beta == 1.0);
}

#[cfg(feature = "viz")]
#[test]
fn the_smile_plots_go_where_they_are_told() {
  use stochastic_rs_quant::vol_surface::sabr_smile::SabrSmileCalibrator;
  use stochastic_rs_quant::vol_surface::sabr_smile::SabrSmileQuotes;
  let dir = std::path::Path::new(env!("CARGO_TARGET_TMPDIR"));
  let cases = [(
    "1Y",
    SabrSmileQuotes {
      tau: 1.0,
      sigma_atm: 0.1,
      sigma_rr: -0.01,
      sigma_bf: 0.003,
    },
  )];
  let fits = SabrSmileCalibrator::calibrate_many(1.1, 0.02, 0.01, 1.0, &cases).unwrap();
  let many = dir.join("sabr_smiles.html");
  SabrSmileCalibrator::plot_many(&many, 1.1, 0.02, 0.01, 1.0, &cases, &fits).unwrap();
  assert!(
    std::fs::read_to_string(&many)
      .unwrap()
      .contains("Plotly.newPlot")
  );
  let one = dir.join("sabr_smile.html");
  SabrSmileCalibrator::new(1.1, 0.02, 0.01, 1.0, cases[0].1)
    .plot(&one, &fits[0])
    .unwrap();
  assert!(
    std::fs::read_to_string(&one)
      .unwrap()
      .contains("Plotly.newPlot")
  );
  let nowhere = dir.join("no such directory").join("smile.html");
  assert!(SabrSmileCalibrator::plot_many(nowhere, 1.1, 0.02, 0.01, 1.0, &cases, &fits).is_err());
}
