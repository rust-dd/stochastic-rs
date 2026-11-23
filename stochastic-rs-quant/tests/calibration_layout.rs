use ndarray::Array1;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_quant::OptionQuote;
use stochastic_rs_quant::OptionType;
use stochastic_rs_quant::QuantError;
use stochastic_rs_quant::calibration::CalibrationOptions;
use stochastic_rs_quant::calibration::bsm::BsmCalibrator;
use stochastic_rs_quant::calibration::heston::HestonCalibrator;
use stochastic_rs_quant::pricing::bsm::BsmCoc;
use stochastic_rs_quant::pricing::bsm::BsmPricer;
use stochastic_rs_quant::pricing::fourier::HestonFourier;
use stochastic_rs_quant::pricing::heston::HestonParams;
use stochastic_rs_quant::traits::CalibrationResult;
use stochastic_rs_quant::traits::Calibrator;
use stochastic_rs_quant::traits::ModelPricer;
use stochastic_rs_quant::traits::ToModel;

#[test]
fn a_reversed_quote_list_gives_the_same_bsm_fit() {
  let pricer = BsmPricer::new(0.25, BsmCoc::Merton1973);
  let quotes = [90.0, 100.0, 110.0].map(|k| {
    let price = pricer.price_call(100.0, k, 0.03, 0.0, 1.0);
    OptionQuote::new(k, 1.0, price, OptionType::Call)
  });
  let fit = |quotes: Vec<OptionQuote>| {
    BsmCalibrator::new(100.0, 0.03, 0.0, quotes)
      .calibrate(None)
      .unwrap()
  };
  let forward = fit(quotes.to_vec());
  let reversed = fit(quotes.iter().rev().copied().collect());
  assert!((forward.params.sigma - 0.25).abs() < 1e-6 && forward.rmse() < 1e-6);
  assert!(
    (reversed.params.sigma - forward.params.sigma).abs() < 1e-12,
    "{} against {}",
    reversed.params.sigma,
    forward.params.sigma
  );
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
    Array1::from_vec(vec![0.5, 1.0, 5.0, 10.0]).view(),
    Array1::from_vec(vec![0.03; 4]).view(),
    InterpolationMethod::LogLinearOnDiscountFactors,
  );
  let quotes = [(1.0, 2.0, 0.22), (2.0, 2.0, 0.20)].map(|(tau, tenor, black_vol)| SwaptionQuote {
    tau,
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

fn heston_quotes(truth: HestonParams, s: f64, r: f64, q: f64) -> Vec<OptionQuote> {
  let model = HestonFourier::new(truth);
  [0.25, 0.5, 1.0]
    .into_iter()
    .flat_map(|tau| (0..9).map(move |i| (80.0 + 5.0 * i as f64, tau)))
    .map(|(k, tau)| OptionQuote::new(k, tau, model.price_call(s, k, r, q, tau), OptionType::Call))
    .collect()
}

#[test]
fn heston_recovers_its_parameters_on_the_unified_shape() {
  let truth = HestonParams {
    v0: 0.05,
    kappa: 1.8,
    theta: 0.06,
    sigma: 0.5,
    rho: -0.6,
  };
  let calibrator = HestonCalibrator::new(100.0, 0.02, 0.0, heston_quotes(truth, 100.0, 0.02, 0.0))
    .with_options(CalibrationOptions::default().with_record_history(true));
  let fit = calibrator.calibrate(None).unwrap();
  assert!(fit.converged && fit.rmse() < 1e-3, "rmse {}", fit.rmse());
  assert!((fit.params.v0 - 0.05).abs() < 1e-3 && (fit.params.rho + 0.6).abs() < 1e-2);
  assert!(fit.iterations.is_some() && fit.max_error.unwrap() < 1e-2 && !fit.history.is_empty());
  assert_eq!(fit.history[0].iteration, 0);
  assert!(
    calibrator.calibrate(None).unwrap().history.len() == fit.history.len(),
    "the trail is per call, not accumulated"
  );
  let model = fit.to_model();
  let truth_price = HestonFourier::new(truth).price_call(100.0, 100.0, 0.02, 0.0, 0.5);
  assert!((model.price_call(100.0, 100.0, 0.02, 0.0, 0.5) - truth_price).abs() < 2e-2);
}

#[test]
fn calibrators_are_send_sync_and_a_shared_calibrator_replays() {
  fn assert_send_sync<T: Send + Sync>() {}
  assert_send_sync::<BsmCalibrator>();
  assert_send_sync::<HestonCalibrator>();
  let truth = HestonParams {
    v0: 0.05,
    kappa: 1.8,
    theta: 0.06,
    sigma: 0.5,
    rho: -0.6,
  };
  let calibrator = HestonCalibrator::new(100.0, 0.02, 0.0, heston_quotes(truth, 100.0, 0.02, 0.0));
  let (a, b) = std::thread::scope(|s| {
    let a = s.spawn(|| calibrator.calibrate(None).unwrap().params);
    let b = s.spawn(|| calibrator.calibrate(None).unwrap().params);
    (a.join().unwrap(), b.join().unwrap())
  });
  assert_eq!(a, b);
}

#[test]
fn a_bad_quote_is_an_error_and_a_bad_spot_a_panic() {
  let truth = HestonParams {
    v0: 0.04,
    kappa: 1.5,
    theta: 0.04,
    sigma: 0.3,
    rho: -0.7,
  };
  let mut quotes = heston_quotes(truth, 100.0, 0.02, 0.0);
  quotes[4].price = f64::NAN;
  let err = HestonCalibrator::new(100.0, 0.02, 0.0, quotes)
    .calibrate(None)
    .unwrap_err();
  assert!(
    matches!(err, QuantError::InvalidQuote { index: 4, .. }),
    "{err}"
  );
  let err = BsmCalibrator::new(100.0, 0.02, 0.0, Vec::new())
    .calibrate(None)
    .unwrap_err();
  assert!(
    matches!(err, QuantError::InsufficientData { got: 0, .. }),
    "{err}"
  );
  assert!(std::panic::catch_unwind(|| BsmCalibrator::new(-1.0, 0.02, 0.0, Vec::new())).is_err());
}

#[test]
fn the_jump_and_two_factor_calibrators_share_the_shape() {
  use stochastic_rs_quant::calibration::double_heston::DoubleHestonCalibrator;
  use stochastic_rs_quant::calibration::hkde::HkdeCalibrator;
  use stochastic_rs_quant::calibration::svj::SvjCalibrator;
  use stochastic_rs_quant::pricing::fourier::SvjParams;
  fn assert_send_sync<T: Send + Sync>() {}
  assert_send_sync::<SvjCalibrator>();
  assert_send_sync::<DoubleHestonCalibrator>();
  assert_send_sync::<HkdeCalibrator>();
  assert_send_sync::<stochastic_rs_quant::calibration::RoughHestonCalibrator>();
  let truth = HestonParams {
    v0: 0.04,
    kappa: 1.5,
    theta: 0.04,
    sigma: 0.3,
    rho: -0.7,
  };
  let quotes = heston_quotes(truth, 100.0, 0.02, 0.0);
  let svj = SvjCalibrator::new(100.0, 0.02, 0.0, quotes.clone())
    .with_start(SvjParams {
      v0: 0.04,
      kappa: 1.5,
      theta: 0.04,
      sigma: 0.3,
      rho: -0.7,
      lambda: 0.1,
      mu_j: -0.05,
      sigma_j: 0.1,
    })
    .with_options(CalibrationOptions::default().with_max_iterations(60))
    .calibrate(None)
    .unwrap();
  assert!(svj.rmse().is_finite() && svj.iterations.is_some());
  let model = svj.to_model();
  assert!(model.price_call(100.0, 100.0, 0.02, 0.0, 0.5).is_finite());
  let hkde = HkdeCalibrator::new(100.0, 0.02, 0.0, quotes.clone())
    .with_options(CalibrationOptions::default().with_max_iterations(40))
    .calibrate(None)
    .unwrap();
  assert!(hkde.rmse().is_finite() && hkde.max_error.is_some());
  let dh = DoubleHestonCalibrator::new(100.0, 0.02, 0.0, quotes)
    .with_options(CalibrationOptions::default().with_max_iterations(40))
    .calibrate(None)
    .unwrap();
  assert!(dh.rmse() < 0.1, "rmse {}", dh.rmse());
  let atm = |tau| HestonFourier::new(truth).price_call(100.0, 100.0, 0.02, 0.0, tau);
  for tau in [0.25, 1.0] {
    let gap = dh.to_model().price_call(100.0, 100.0, 0.02, 0.0, tau) - atm(tau);
    assert!(gap.abs() < 5e-2, "tau {tau}: {gap}");
  }
}

#[test]
fn the_levy_and_cgmysv_calibrators_take_quotes_and_params() {
  use stochastic_rs_quant::calibration::cgmysv::CgmysvCalibrator;
  use stochastic_rs_quant::calibration::levy::LevyCalibrator;
  use stochastic_rs_quant::calibration::levy::LevyModelType;
  use stochastic_rs_quant::calibration::levy::LevyParams;
  use stochastic_rs_quant::pricing::fourier::MertonJdFourier;
  fn assert_send_sync<T: Send + Sync>() {}
  assert_send_sync::<LevyCalibrator>();
  assert_send_sync::<CgmysvCalibrator>();
  let truth = MertonJdFourier {
    sigma: 0.15,
    lambda: 0.4,
    mu_j: -0.1,
    sigma_j: 0.2,
  };
  let quotes: Vec<OptionQuote> = [0.5, 1.0]
    .into_iter()
    .flat_map(|tau| (0..7).map(move |i| (85.0 + 5.0 * i as f64, tau)))
    .map(|(k, tau)| {
      let price = truth.price_call(100.0, k, 0.02, 0.0, tau);
      OptionQuote::new(k, tau, price, OptionType::Call)
    })
    .collect();
  let fit = LevyCalibrator::new(LevyModelType::MertonJd, 100.0, 0.02, 0.0, quotes.clone())
    .with_start(LevyParams {
      values: vec![0.2, 0.3, -0.05, 0.15],
      model_type: LevyModelType::MertonJd,
    })
    .calibrate(None)
    .unwrap();
  assert!(
    fit.rmse() < 1e-2 && fit.iterations.is_some(),
    "rmse {}",
    fit.rmse()
  );
  let gap = fit.to_model().price_call(100.0, 100.0, 0.02, 0.0, 1.0)
    - truth.price_call(100.0, 100.0, 0.02, 0.0, 1.0);
  assert!(gap.abs() < 5e-2, "{gap}");
  let wrong = std::panic::catch_unwind(|| {
    LevyCalibrator::new(LevyModelType::Kou, 100.0, 0.02, 0.0, Vec::new()).with_start(LevyParams {
      values: vec![0.2],
      model_type: LevyModelType::Kou,
    })
  });
  assert!(
    wrong.is_err(),
    "a start with the wrong length is a programmer error"
  );
  let cg = CgmysvCalibrator::new(100.0, 0.02, 0.0, quotes)
    .with_options(CalibrationOptions::default().with_max_iterations(5))
    .calibrate(None)
    .unwrap();
  assert!(cg.rmse().is_finite() && cg.iterations.is_some_and(|n| n <= 5));
}

#[test]
fn hscm_rbergomi_and_slv_share_the_shape() {
  use ndarray::Array2;
  use stochastic_rs_quant::calibration::heston_slv::HestonSlvCalibrator;
  use stochastic_rs_quant::calibration::heston_stoch_corr::HscmCalibrator;
  use stochastic_rs_quant::calibration::rbergomi::RBergomiCalibrator;
  use stochastic_rs_quant::pricing::slv::ParticleMethod;
  fn assert_send_sync<T: Send + Sync>() {}
  assert_send_sync::<HscmCalibrator>();
  assert_send_sync::<RBergomiCalibrator>();
  assert_send_sync::<HestonSlvCalibrator>();
  let heston = HestonParams {
    v0: 0.04,
    kappa: 2.0,
    theta: 0.05,
    sigma: 0.4,
    rho: -0.6,
  };
  let (s, r, q) = (100.0, 0.02, 0.0);
  let strikes = ndarray::Array1::linspace(70.0, 140.0, 36).to_vec();
  let maturities = vec![0.25, 0.5, 0.75, 1.0];
  let model = HestonFourier::new(heston);
  let calls = Array2::from_shape_fn((4, 36), |(j, i)| {
    model.price_call(s, strikes[i], r, q, maturities[j])
  });
  let fit = HestonSlvCalibrator::new(s, r, q, strikes, maturities, calls)
    .with_mixing(0.5)
    .with_heston_params(heston)
    .with_particle_method(
      ParticleMethod::default()
        .with_particles(5_000)
        .with_steps_per_year(50)
        .with_seed(Deterministic::new(7)),
    )
    .calibrate(None)
    .unwrap();
  assert!(fit.converged && fit.rmse() < 1.0 && fit.diagnostics.is_none());
  assert_eq!((fit.params.r, fit.params.q), (r, q));
  let pricer = fit.to_model().with_paths(4_000).with_steps_per_year(50);
  assert!(pricer.price_call(s, 100.0, r, q, 0.5).is_finite());
  let hscm = HscmCalibrator::new(s, r, q, heston_quotes(heston, s, r, q)[..9].to_vec())
    .with_options(CalibrationOptions::default().with_max_iterations(3))
    .calibrate(None)
    .unwrap();
  let message = hscm.message.as_deref().unwrap();
  assert!(!hscm.converged, "{message}");
  assert!(message.starts_with("slsqp max evaluations reached, objective "));
}
