use stochastic_rs_quant::OptionQuote;
use stochastic_rs_quant::OptionType;
use stochastic_rs_quant::QuantError;
use stochastic_rs_quant::calibration::CalibrationOptions;
use stochastic_rs_quant::calibration::Regularization;
use stochastic_rs_quant::calibration::bsm::BsmCalibrator;
use stochastic_rs_quant::calibration::cgmysv::CgmysvCalibrator;
use stochastic_rs_quant::calibration::double_heston::DoubleHestonCalibrator;
use stochastic_rs_quant::calibration::double_heston::DoubleHestonParams;
use stochastic_rs_quant::calibration::heston::HestonCalibrator;
use stochastic_rs_quant::calibration::hkde::HkdeCalibrator;
use stochastic_rs_quant::calibration::levy::LevyCalibrator;
use stochastic_rs_quant::calibration::levy::LevyModelType;
use stochastic_rs_quant::calibration::levy::LevyParams;
use stochastic_rs_quant::calibration::rough_heston::RoughHestonCalibrator;
use stochastic_rs_quant::calibration::sabr::SabrCalibrator;
use stochastic_rs_quant::calibration::svj::SvjCalibrator;
use stochastic_rs_quant::pricing::bsm::BsmCoc;
use stochastic_rs_quant::pricing::bsm::BsmPricer;
use stochastic_rs_quant::pricing::cgmysv::CgmysvModel;
use stochastic_rs_quant::pricing::cgmysv::CgmysvParams;
use stochastic_rs_quant::pricing::fourier::BatesFourier;
use stochastic_rs_quant::pricing::fourier::DoubleHestonFourier;
use stochastic_rs_quant::pricing::fourier::HestonFourier;
use stochastic_rs_quant::pricing::fourier::HkdeFourier;
use stochastic_rs_quant::pricing::fourier::HkdeParams;
use stochastic_rs_quant::pricing::fourier::MertonJdFourier;
use stochastic_rs_quant::pricing::fourier::RoughHeston;
use stochastic_rs_quant::pricing::fourier::RoughHestonCf;
use stochastic_rs_quant::pricing::fourier::RoughHestonParams;
use stochastic_rs_quant::pricing::fourier::SvjParams;
use stochastic_rs_quant::pricing::heston::HestonParams;
use stochastic_rs_quant::pricing::sabr::SabrParams;
use stochastic_rs_quant::pricing::sabr::SabrPricer;
use stochastic_rs_quant::traits::Calibrator;
use stochastic_rs_quant::traits::ModelPricer;

const S: f64 = 100.0;
const R: f64 = 0.02;
const Q: f64 = 0.0;

/// A weight of `1e8` on parameter `at` of `n`, anchored at `anchor`: heavy against any price row.
fn pull(n: usize, at: usize, anchor: f64) -> CalibrationOptions {
  let (mut anchors, mut weights) = (vec![0.0; n], vec![0.0; n]);
  anchors[at] = anchor;
  weights[at] = 1e8;
  CalibrationOptions::default()
    .with_regularization(Regularization::new(anchors, weights))
    .with_max_iterations(30)
}

fn quotes(model: &impl ModelPricer, strikes: &[f64], tau: f64) -> Vec<OptionQuote> {
  let price = |k| model.price_call(S, k, R, Q, tau);
  let quote = |&k: &f64| OptionQuote::new(k, tau, price(k), OptionType::Call);
  strikes.iter().map(quote).collect()
}

fn near(name: &str, fitted: f64, anchor: f64) {
  assert!(
    (fitted - anchor).abs() < 1e-3,
    "{name}: fitted {fitted}, anchor {anchor}"
  );
}

const STRIKES: [f64; 5] = [85.0, 95.0, 100.0, 105.0, 115.0];

const HESTON: HestonParams = HestonParams {
  v0: 0.04,
  kappa: 1.5,
  theta: 0.04,
  sigma: 0.3,
  rho: -0.7,
};

#[test]
fn bsm_holds_sigma_at_its_anchor() {
  let market = quotes(&BsmPricer::new(0.25, BsmCoc::Merton1973), &STRIKES, 0.5);
  let fit = BsmCalibrator::new(S, R, Q, market)
    .with_options(pull(1, 0, 0.4))
    .calibrate(None)
    .unwrap();
  near("sigma", fit.params.sigma, 0.4);
}

#[test]
fn heston_holds_kappa_at_its_anchor() {
  let market = quotes(&HestonFourier::new(HESTON), &STRIKES, 0.5);
  let fit = HestonCalibrator::new(S, R, Q, market)
    .with_start(HESTON)
    .with_options(pull(5, 1, 4.0))
    .calibrate(None)
    .unwrap();
  near("kappa", fit.params.kappa, 4.0);
}

#[test]
fn sabr_holds_nu_at_its_anchor_in_hagan_order() {
  let truth = SabrParams {
    alpha: 0.2,
    beta: 1.0,
    rho: -0.3,
    nu: 0.6,
  };
  let fit = SabrCalibrator::new(S, R, Q, quotes(&SabrPricer::new(truth), &STRIKES, 1.0))
    .with_start(truth)
    .with_options(pull(3, 2, 0.9))
    .calibrate(None)
    .unwrap();
  near("nu", fit.params.nu, 0.9);
}

#[test]
fn svj_holds_lambda_at_its_anchor() {
  let truth = SvjParams {
    v0: 0.04,
    kappa: 1.5,
    theta: 0.04,
    sigma: 0.3,
    rho: -0.7,
    lambda: 0.3,
    mu_j: -0.05,
    sigma_j: 0.1,
  };
  let fit = SvjCalibrator::new(S, R, Q, quotes(&BatesFourier::new(truth), &STRIKES, 0.5))
    .with_start(truth)
    .with_options(pull(8, 5, 2.0))
    .calibrate(None)
    .unwrap();
  near("lambda", fit.params.lambda, 2.0);
}

#[test]
fn double_heston_holds_the_second_kappa_at_its_anchor() {
  let second = HestonParams {
    v0: 0.02,
    kappa: 1.0,
    theta: 0.03,
    sigma: 0.2,
    rho: -0.3,
  };
  let truth = DoubleHestonParams {
    first: HESTON,
    second,
  };
  let model = DoubleHestonFourier::new(HESTON, second);
  let fit = DoubleHestonCalibrator::new(S, R, Q, quotes(&model, &STRIKES, 0.5))
    .with_start(truth)
    .with_options(pull(10, 6, 4.0))
    .calibrate(None)
    .unwrap();
  near("second.kappa", fit.params.second.kappa, 4.0);
}

#[test]
fn hkde_holds_lambda_at_its_anchor() {
  let truth = HkdeParams {
    v0: 0.04,
    kappa: 1.5,
    theta: 0.04,
    sigma: 0.3,
    rho: -0.7,
    lambda: 0.5,
    p_up: 0.4,
    eta1: 10.0,
    eta2: 5.0,
  };
  let fit = HkdeCalibrator::new(S, R, Q, quotes(&HkdeFourier::new(truth), &STRIKES, 0.5))
    .with_start(truth)
    .with_options(pull(9, 5, 3.0))
    .calibrate(None)
    .unwrap();
  near("lambda", fit.params.lambda, 3.0);
}

#[test]
fn levy_holds_the_jump_intensity_at_its_anchor() {
  let truth = MertonJdFourier {
    sigma: 0.15,
    lambda: 0.4,
    mu_j: -0.1,
    sigma_j: 0.2,
  };
  let market = quotes(&truth, &STRIKES, 0.5);
  let fit = LevyCalibrator::new(LevyModelType::MertonJd, S, R, Q, market)
    .with_start(LevyParams {
      values: vec![0.15, 0.4, -0.1, 0.2],
      model_type: LevyModelType::MertonJd,
    })
    .with_options(pull(4, 1, 1.5))
    .calibrate(None)
    .unwrap();
  near("lambda", fit.params.values[1], 1.5);
}

#[test]
fn cgmysv_holds_v0_at_its_anchor_through_its_logistic_coordinates() {
  let truth = CgmysvParams {
    alpha: 0.5184,
    lambda_plus: 25.4592,
    lambda_minus: 4.6040,
    kappa: 1.0029,
    eta: 0.0711,
    zeta: 0.3443,
    rho: -2.0283,
    v0: 0.01115,
  };
  let model = CgmysvModel {
    params: truth.clone(),
  };
  let fit = CgmysvCalibrator::new(S, R, Q, quotes(&model, &[90.0, 100.0, 110.0], 0.25))
    .with_start(truth)
    .with_options(pull(8, 7, 0.05))
    .calibrate(None)
    .unwrap();
  near("v0", fit.params.v0, 0.05);
}

#[test]
fn rough_heston_holds_its_hurst_exponent_at_its_anchor() {
  let truth = RoughHestonParams::new(0.04, 1.0, 0.04, 0.3, -0.65, 0.1);
  let model = RoughHeston::new(truth, RoughHestonCf::default());
  let fit = RoughHestonCalibrator::new(S, R, Q, quotes(&model, &[90.0, 100.0, 110.0], 0.25))
    .with_start(truth)
    .with_options(pull(6, 5, 0.2))
    .calibrate(None)
    .unwrap();
  near("hurst", fit.params.hurst, 0.2);
}

#[test]
fn a_pull_of_another_length_is_each_fits_dimension_mismatch() {
  let wrong = || {
    CalibrationOptions::default()
      .with_regularization(Regularization::uniform(vec![0.0; 11], 1.0))
      .with_max_iterations(1)
  };
  let market = quotes(&HestonFourier::new(HESTON), &STRIKES, 0.5);
  let fits = [
    (
      1,
      BsmCalibrator::new(S, R, Q, market.clone())
        .with_options(wrong())
        .calibrate(None)
        .map(drop),
    ),
    (
      5,
      HestonCalibrator::new(S, R, Q, market.clone())
        .with_options(wrong())
        .calibrate(None)
        .map(drop),
    ),
    (
      3,
      SabrCalibrator::new(S, R, Q, market.clone())
        .with_options(wrong())
        .calibrate(None)
        .map(drop),
    ),
    (
      8,
      SvjCalibrator::new(S, R, Q, market.clone())
        .with_options(wrong())
        .calibrate(None)
        .map(drop),
    ),
    (
      10,
      DoubleHestonCalibrator::new(S, R, Q, market.clone())
        .with_options(wrong())
        .calibrate(None)
        .map(drop),
    ),
    (
      9,
      HkdeCalibrator::new(S, R, Q, market.clone())
        .with_options(wrong())
        .calibrate(None)
        .map(drop),
    ),
    (
      4,
      LevyCalibrator::new(LevyModelType::MertonJd, S, R, Q, market.clone())
        .with_options(wrong())
        .calibrate(None)
        .map(drop),
    ),
    (
      8,
      CgmysvCalibrator::new(S, R, Q, market.clone())
        .with_options(wrong())
        .calibrate(None)
        .map(drop),
    ),
    (
      6,
      RoughHestonCalibrator::new(S, R, Q, market)
        .with_options(wrong())
        .calibrate(None)
        .map(drop),
    ),
  ];
  for (expected, fit) in fits {
    assert_eq!(
      fit,
      Err(QuantError::DimensionMismatch {
        what: "regularization anchors",
        expected,
        got: 11
      })
    );
  }
}
