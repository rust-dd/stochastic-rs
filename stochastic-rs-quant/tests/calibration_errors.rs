use stochastic_rs_quant::OptionQuote;
use stochastic_rs_quant::OptionType;
use stochastic_rs_quant::QuantError;
use stochastic_rs_quant::calibration::CalibrationOptions;
use stochastic_rs_quant::calibration::bsm::BsmCalibrator;
use stochastic_rs_quant::calibration::bsm::BsmParams;
use stochastic_rs_quant::calibration::cgmysv::CgmysvCalibrator;
use stochastic_rs_quant::calibration::double_heston::DoubleHestonCalibrator;
use stochastic_rs_quant::calibration::double_heston::DoubleHestonParams;
use stochastic_rs_quant::calibration::heston::HestonCalibrator;
use stochastic_rs_quant::calibration::heston_stoch_corr::HscmCalibrator;
use stochastic_rs_quant::calibration::heston_stoch_corr::HscmParams;
use stochastic_rs_quant::calibration::hkde::HkdeCalibrator;
use stochastic_rs_quant::calibration::levy::LevyCalibrator;
use stochastic_rs_quant::calibration::levy::LevyModelType;
use stochastic_rs_quant::calibration::levy::LevyParams;
use stochastic_rs_quant::calibration::sabr::SabrCalibrator;
use stochastic_rs_quant::calibration::svj::SvjCalibrator;
use stochastic_rs_quant::pricing::bsm::BsmCoc;
use stochastic_rs_quant::pricing::bsm::BsmPricer;
use stochastic_rs_quant::pricing::cgmysv::CgmysvParams;
use stochastic_rs_quant::pricing::fourier::HkdeParams;
use stochastic_rs_quant::pricing::fourier::SvjParams;
use stochastic_rs_quant::pricing::heston::HestonParams;
use stochastic_rs_quant::pricing::sabr::SabrParams;
use stochastic_rs_quant::traits::Calibrator;
use stochastic_rs_quant::traits::ModelPricer;

const S: f64 = 100.0;
const R: f64 = 0.02;

/// One calibrator on `quotes` at spot `s` and dividend yield `q`, from a non-finite start when
/// `nan_start`.
type Fit = fn(f64, f64, Vec<OptionQuote>, bool) -> Result<(), QuantError>;

const NAN: f64 = f64::NAN;

const HESTON_NAN: HestonParams = HestonParams {
  v0: NAN,
  kappa: 1.5,
  theta: 0.04,
  sigma: 0.3,
  rho: -0.7,
};

fn bsm(s: f64, q: f64, quotes: Vec<OptionQuote>, nan_start: bool) -> Result<(), QuantError> {
  let start = nan_start.then_some(BsmParams { sigma: NAN });
  BsmCalibrator::new(s, R, q, quotes)
    .calibrate(start)
    .map(drop)
}

fn heston(s: f64, q: f64, quotes: Vec<OptionQuote>, nan_start: bool) -> Result<(), QuantError> {
  let start = nan_start.then_some(HESTON_NAN);
  HestonCalibrator::new(s, R, q, quotes)
    .calibrate(start)
    .map(drop)
}

fn sabr(s: f64, q: f64, quotes: Vec<OptionQuote>, nan_start: bool) -> Result<(), QuantError> {
  let start = nan_start.then_some(SabrParams {
    alpha: NAN,
    beta: 1.0,
    rho: 0.0,
    nu: 0.5,
  });
  SabrCalibrator::new(s, R, q, quotes)
    .calibrate(start)
    .map(drop)
}

fn svj(s: f64, q: f64, quotes: Vec<OptionQuote>, nan_start: bool) -> Result<(), QuantError> {
  let start = nan_start.then_some(SvjParams {
    v0: 0.04,
    kappa: 1.5,
    theta: 0.04,
    sigma: 0.3,
    rho: -0.7,
    lambda: NAN,
    mu_j: -0.05,
    sigma_j: 0.1,
  });
  SvjCalibrator::new(s, R, q, quotes)
    .calibrate(start)
    .map(drop)
}

fn double_heston(s: f64, q: f64, quotes: Vec<OptionQuote>, nan: bool) -> Result<(), QuantError> {
  let first = HestonParams {
    v0: 0.04,
    ..HESTON_NAN
  };
  let start = nan.then_some(DoubleHestonParams {
    first,
    second: HESTON_NAN,
  });
  DoubleHestonCalibrator::new(s, R, q, quotes)
    .calibrate(start)
    .map(drop)
}

fn hkde(s: f64, q: f64, quotes: Vec<OptionQuote>, nan_start: bool) -> Result<(), QuantError> {
  let start = nan_start.then_some(HkdeParams {
    v0: 0.04,
    kappa: 1.5,
    theta: 0.04,
    sigma: 0.3,
    rho: -0.7,
    lambda: 0.5,
    p_up: NAN,
    eta1: 10.0,
    eta2: 5.0,
  });
  HkdeCalibrator::new(s, R, q, quotes)
    .calibrate(start)
    .map(drop)
}

fn levy(s: f64, q: f64, quotes: Vec<OptionQuote>, nan_start: bool) -> Result<(), QuantError> {
  let start = nan_start.then_some(LevyParams {
    values: vec![0.15, NAN, -0.1, 0.2],
    model_type: LevyModelType::MertonJd,
  });
  let calibrator = LevyCalibrator::new(LevyModelType::MertonJd, s, R, q, quotes);
  calibrator.calibrate(start).map(drop)
}

fn cgmysv(s: f64, q: f64, quotes: Vec<OptionQuote>, nan_start: bool) -> Result<(), QuantError> {
  let start = nan_start.then_some(CgmysvParams {
    alpha: 0.5,
    lambda_plus: 20.0,
    lambda_minus: 5.0,
    kappa: 1.0,
    eta: NAN,
    zeta: 0.3,
    rho: -1.0,
    v0: 0.01,
  });
  CgmysvCalibrator::new(s, R, q, quotes)
    .calibrate(start)
    .map(drop)
}

fn hscm(s: f64, q: f64, quotes: Vec<OptionQuote>, nan_start: bool) -> Result<(), QuantError> {
  let start = nan_start.then_some(HscmParams {
    kappa_v: 2.0,
    theta_v: 0.04,
    sigma_v: 0.3,
    v0: NAN,
    kappa_r: 5.0,
    mu_r: -0.5,
    sigma_r: 0.2,
    rho0: -0.7,
    rho2: 0.3,
  });
  HscmCalibrator::new(s, R, q, quotes)
    .calibrate(start)
    .map(drop)
}

const TABLE: [(&str, Fit); 9] = [
  ("bsm", bsm),
  ("heston", heston),
  ("sabr", sabr),
  ("svj", svj),
  ("double heston", double_heston),
  ("hkde", hkde),
  ("levy", levy),
  ("cgmysv", cgmysv),
  ("hscm", hscm),
];

fn quotes() -> Vec<OptionQuote> {
  let pricer = BsmPricer::new(0.2, BsmCoc::Merton1973);
  let quote = |k| {
    OptionQuote::new(
      k,
      0.5,
      pricer.price_call(S, k, R, 0.0, 0.5),
      OptionType::Call,
    )
  };
  [90.0, 95.0, 100.0, 105.0, 110.0].map(quote).to_vec()
}

#[test]
fn every_quote_calibrator_answers_bad_market_data_with_err_and_a_bad_spot_with_a_panic() {
  for (name, fit) in TABLE {
    let empty = fit(S, 0.0, Vec::new(), false);
    assert!(
      matches!(empty, Err(QuantError::InsufficientData { got: 0, .. })),
      "{name}: {empty:?}"
    );
    let mut nan_price = quotes();
    nan_price[2].price = NAN;
    let bad = fit(S, 0.0, nan_price, false);
    assert!(
      matches!(bad, Err(QuantError::InvalidQuote { index: 2, .. })),
      "{name}: {bad:?}"
    );
    let start = fit(S, 0.0, quotes(), true);
    assert!(
      matches!(start, Err(QuantError::InvalidInput { what: "start", .. })),
      "{name}: {start:?}"
    );
    let spot = std::panic::catch_unwind(|| fit(-1.0, 0.0, quotes(), false));
    assert!(
      spot.is_err(),
      "{name}: a negative spot is the caller's, not the market's"
    );
  }
}

/// A dividend yield of −1e4 overflows `S·e^{−qτ}`: no price exists at the start. HKDE reads it off
/// its weights, which need a Black volatility, before the fit starts.
#[test]
fn a_carry_no_model_can_price_leaves_the_optimiser_without_an_iterate() {
  for (name, fit) in TABLE {
    let err = fit(S, -1e4, quotes(), false);
    let expected = match name {
      "hkde" => matches!(err, Err(QuantError::InvalidQuote { index: 0, .. })),
      _ => matches!(err, Err(QuantError::Optimizer { .. })),
    };
    assert!(expected, "{name}: {err:?}");
  }
}

type WithOptions = fn(CalibrationOptions) -> Result<(), QuantError>;

/// A knob written through its field skips its builder's panic, so the fit answers it.
#[test]
fn a_levenberg_marquardt_knob_written_outside_its_builders_domain_is_invalid_input() {
  let fits: [(&str, WithOptions); 8] = [
    ("bsm", |o| {
      let c = BsmCalibrator::new(S, R, 0.0, quotes());
      c.with_options(o).calibrate(None).map(drop)
    }),
    ("heston", |o| {
      let c = HestonCalibrator::new(S, R, 0.0, quotes());
      c.with_options(o).calibrate(None).map(drop)
    }),
    ("sabr", |o| {
      let c = SabrCalibrator::new(S, R, 0.0, quotes());
      c.with_options(o).calibrate(None).map(drop)
    }),
    ("svj", |o| {
      let c = SvjCalibrator::new(S, R, 0.0, quotes());
      c.with_options(o).calibrate(None).map(drop)
    }),
    ("double heston", |o| {
      let c = DoubleHestonCalibrator::new(S, R, 0.0, quotes());
      c.with_options(o).calibrate(None).map(drop)
    }),
    ("hkde", |o| {
      let c = HkdeCalibrator::new(S, R, 0.0, quotes());
      c.with_options(o).calibrate(None).map(drop)
    }),
    ("levy", |o| {
      let c = LevyCalibrator::new(LevyModelType::MertonJd, S, R, 0.0, quotes());
      c.with_options(o).calibrate(None).map(drop)
    }),
    ("cgmysv", |o| {
      let c = CgmysvCalibrator::new(S, R, 0.0, quotes());
      c.with_options(o).calibrate(None).map(drop)
    }),
  ];
  let mut patience = CalibrationOptions::default();
  patience.patience = 0;
  let mut tolerance = CalibrationOptions::default();
  tolerance.tolerance = Some(NAN);
  let mut cap = CalibrationOptions::default();
  cap.max_iterations = Some(0);
  for (name, fit) in fits {
    for (knob, options) in [
      ("patience", &patience),
      ("tolerance", &tolerance),
      ("max_iterations", &cap),
    ] {
      let err = fit(options.clone());
      assert!(
        matches!(&err, Err(QuantError::InvalidInput { what: "options", reason }) if reason.starts_with(knob)),
        "{name}, {knob}: {err:?}"
      );
    }
  }
}

#[test]
fn every_calibrator_is_send_and_sync() {
  use stochastic_rs_quant::calibration::heston_slv::HestonSlvCalibrator;
  use stochastic_rs_quant::calibration::hw_swaption::HullWhiteSwaptionCalibrator;
  use stochastic_rs_quant::calibration::rbergomi::RBergomiCalibrator;
  use stochastic_rs_quant::calibration::sabr_caplet::SabrCapletCalibrator;
  use stochastic_rs_quant::calibration::tree_swaption::BlackKarasinskiSwaptionCalibrator;
  use stochastic_rs_quant::calibration::tree_swaption::G2ppSwaptionCalibrator;
  use stochastic_rs_quant::vol_surface::sabr_smile::SabrSmileCalibrator;
  fn assert_send_sync<T: Send + Sync>() {}
  assert_send_sync::<BsmCalibrator>();
  assert_send_sync::<HestonCalibrator>();
  assert_send_sync::<SabrCalibrator>();
  assert_send_sync::<SabrCapletCalibrator>();
  assert_send_sync::<SabrSmileCalibrator>();
  assert_send_sync::<HullWhiteSwaptionCalibrator<'static>>();
  assert_send_sync::<BlackKarasinskiSwaptionCalibrator<'static>>();
  assert_send_sync::<G2ppSwaptionCalibrator<'static>>();
  assert_send_sync::<SvjCalibrator>();
  assert_send_sync::<DoubleHestonCalibrator>();
  assert_send_sync::<HkdeCalibrator>();
  assert_send_sync::<LevyCalibrator>();
  assert_send_sync::<CgmysvCalibrator>();
  assert_send_sync::<HscmCalibrator>();
  assert_send_sync::<RBergomiCalibrator>();
  assert_send_sync::<HestonSlvCalibrator>();
}
