//! Too little or non-finite data is an `Err`, never a panic inside a regression, and every test's
//! answer to a constant series is pinned.

use ndarray::Array1;
use ndarray::Array2;
use ndarray::ArrayView1;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_stats::StatsError;
use stochastic_rs_stats::acf::LjungBoxConfig;
use stochastic_rs_stats::acf::acf;
use stochastic_rs_stats::acf::ljung_box;
use stochastic_rs_stats::acf::pacf;
use stochastic_rs_stats::bayesian_diffusion::fit_diffusion_bayesian;
use stochastic_rs_stats::distfit::fit_johnson_su;
use stochastic_rs_stats::distfit::fit_skew_t;
use stochastic_rs_stats::distfit::fit_variance_gamma;
use stochastic_rs_stats::econometrics::changepoint::cusum;
use stochastic_rs_stats::econometrics::changepoint::pelt;
use stochastic_rs_stats::econometrics::cointegration::EngleGrangerConfig;
use stochastic_rs_stats::econometrics::cointegration::engle_granger_test;
use stochastic_rs_stats::econometrics::cointegration::fit_vecm;
use stochastic_rs_stats::econometrics::cointegration::johansen_test;
use stochastic_rs_stats::econometrics::granger::granger_causality_test;
use stochastic_rs_stats::econometrics::hmm::GaussianHmm;
use stochastic_rs_stats::evt::fit_gev;
use stochastic_rs_stats::evt::fit_gpd;
use stochastic_rs_stats::evt::fit_gpd_pwm;
use stochastic_rs_stats::evt::fit_hill;
use stochastic_rs_stats::evt::fit_pot;
use stochastic_rs_stats::garch::GarchSpec;
use stochastic_rs_stats::garch::fit_garch;
use stochastic_rs_stats::gmm_cir::fit_cir_gmm;
use stochastic_rs_stats::goodness_of_fit::chi_square::ChiSquareGofConfig;
use stochastic_rs_stats::goodness_of_fit::chi_square::chi_square_gof_test;
use stochastic_rs_stats::goodness_of_fit::kolmogorov_smirnov::KolmogorovSmirnovConfig;
use stochastic_rs_stats::goodness_of_fit::kolmogorov_smirnov::kolmogorov_smirnov_test;
use stochastic_rs_stats::heston_mle::fit_heston_nmle;
use stochastic_rs_stats::heston_mle::fit_heston_nmle_with_delta;
use stochastic_rs_stats::heston_mle::fit_heston_pmle;
use stochastic_rs_stats::heston_mle::fit_heston_pmle_with_delta;
use stochastic_rs_stats::heston_nmle_cekf::HestonNmleCekfConfig;
use stochastic_rs_stats::heston_nmle_cekf::fit_heston_nmle_cekf;
use stochastic_rs_stats::mle::DensityApprox;
use stochastic_rs_stats::mle::fit_mle;
use stochastic_rs_stats::normality::anderson_darling::AndersonDarlingConfig;
use stochastic_rs_stats::normality::anderson_darling::anderson_darling_normal_test;
use stochastic_rs_stats::normality::jarque_bera::JarqueBeraConfig;
use stochastic_rs_stats::normality::jarque_bera::jarque_bera_test;
use stochastic_rs_stats::normality::shapiro_francia::ShapiroFranciaConfig;
use stochastic_rs_stats::normality::shapiro_francia::shapiro_francia_test;
use stochastic_rs_stats::particle_mle::fit_sv_particle_mle;
use stochastic_rs_stats::qmle::DiffusionKind;
use stochastic_rs_stats::qmle::fit_diffusion_qmle;
use stochastic_rs_stats::qmle::fit_ou_mle;
use stochastic_rs_stats::realized::HarRv;
use stochastic_rs_stats::realized::KernelType;
use stochastic_rs_stats::realized::bipower::bns_jump_test;
use stochastic_rs_stats::realized::bipower_variation;
use stochastic_rs_stats::realized::ewma_variance;
use stochastic_rs_stats::realized::lee_mykland::lee_mykland_test;
use stochastic_rs_stats::realized::medrv;
use stochastic_rs_stats::realized::minrv;
use stochastic_rs_stats::realized::multi_scale_rv;
use stochastic_rs_stats::realized::pre_averaged_variance;
use stochastic_rs_stats::realized::pre_averaging::pre_averaged_variance_with_block;
use stochastic_rs_stats::realized::realized_kernel;
use stochastic_rs_stats::realized::realized_kurtosis;
use stochastic_rs_stats::realized::realized_quarticity;
use stochastic_rs_stats::realized::realized_semivariance;
use stochastic_rs_stats::realized::realized_skewness;
use stochastic_rs_stats::realized::realized_variance;
use stochastic_rs_stats::realized::realized_volatility;
use stochastic_rs_stats::realized::riskmetrics_variance;
use stochastic_rs_stats::realized::tripower_quarticity;
use stochastic_rs_stats::realized::two_scale_rv;
use stochastic_rs_stats::spectral::PeriodogramConfig;
use stochastic_rs_stats::spectral::SpectrumSearchConfig;
use stochastic_rs_stats::spectral::dominant_frequency_fft;
use stochastic_rs_stats::spectral::fft_spectrum_search;
use stochastic_rs_stats::spectral::periodogram_fft;
use stochastic_rs_stats::stationarity::adf::AdfConfig;
use stochastic_rs_stats::stationarity::adf::adf_test;
use stochastic_rs_stats::stationarity::andrews_ploberger::AndrewsPlobergerConfig;
use stochastic_rs_stats::stationarity::andrews_ploberger::andrews_ploberger_test;
use stochastic_rs_stats::stationarity::cusum::CusumConfig;
use stochastic_rs_stats::stationarity::cusum::cusum_test;
use stochastic_rs_stats::stationarity::ers_dfgls::ErsConfig;
use stochastic_rs_stats::stationarity::ers_dfgls::ers_dfgls_test;
use stochastic_rs_stats::stationarity::kpss::KpssConfig;
use stochastic_rs_stats::stationarity::kpss::kpss_test;
use stochastic_rs_stats::stationarity::leybourne_mccabe::LeybourneMcCabeConfig;
use stochastic_rs_stats::stationarity::leybourne_mccabe::leybourne_mccabe_test;
use stochastic_rs_stats::stationarity::lo_mackinlay::LoMacKinlayConfig;
use stochastic_rs_stats::stationarity::lo_mackinlay::lo_mackinlay_test;
use stochastic_rs_stats::stationarity::phillips_perron::PhillipsPerronConfig;
use stochastic_rs_stats::stationarity::phillips_perron::phillips_perron_test;
use stochastic_rs_stats::stationarity::reset::ResetConfig;
use stochastic_rs_stats::stationarity::reset::reset_test;
use stochastic_rs_stochastic::diffusion::ou::Ou;

type Answer = Box<dyn Fn(ArrayView1<f64>) -> Option<StatsError>>;

type Entry = (&'static str, Answer, Option<StatsError>);

const OLS: Option<StatsError> = Some(StatsError::Singular("OLS design"));

const SERIES: Option<StatsError> = Some(StatsError::DegenerateData("constant series"));

const SAMPLE: Option<StatsError> = Some(StatsError::DegenerateData("constant sample"));

const EXCESSES: Option<StatsError> = Some(StatsError::DegenerateData("constant excesses"));

const JOHANSEN: Option<StatsError> = Some(StatsError::Singular("S00"));

const VARIANCE: Option<StatsError> = Some(StatsError::DegenerateData("constant variance series"));

const DT: f64 = 1.0 / 252.0;

fn err<T, F: Fn(ArrayView1<f64>) -> Result<T, StatsError> + 'static>(f: F) -> Answer {
  Box::new(move |x| f(x).err())
}

fn design(n: usize) -> Array2<f64> {
  Array2::<f64>::ones((n, 1))
}

/// An intercept and a trend: RESET's powers of an intercept-only fit are collinear with it.
fn trend_design(n: usize) -> Array2<f64> {
  Array2::<f64>::from_shape_fn((n, 2), |(t, j)| if j == 0 { 1.0 } else { t as f64 })
}

/// The series and its square as the two columns of a matrix-input test.
fn pair(x: ArrayView1<f64>) -> Array2<f64> {
  Array2::<f64>::from_shape_fn(
    (x.len(), 2),
    |(t, j)| if j == 0 { x[t] } else { x[t] * x[t] },
  )
}

fn prices(x: ArrayView1<f64>) -> Array1<f64> {
  x.mapv(|v| v + 100.0)
}

fn variances(x: ArrayView1<f64>) -> Array1<f64> {
  x.mapv(|v| v.abs() + 0.01)
}

fn two_state_hmm() -> GaussianHmm {
  GaussianHmm::new(
    ndarray::array![0.5, 0.5],
    ndarray::array![[0.9, 0.1], [0.1, 0.9]],
    ndarray::array![-0.5, 0.5],
    ndarray::array![1.0, 1.0],
  )
}

fn har_model() -> HarRv {
  HarRv::fit(Array1::from_iter((0..60).map(|t| 1.0 + 0.5 * (t as f64 * 0.7).sin())).view()).unwrap()
}

pub fn hypothesis_tests() -> Vec<Entry> {
  vec![
    ("adf", err(|x| adf_test(x, AdfConfig::default())), OLS),
    ("kpss", err(|x| kpss_test(x, KpssConfig::default())), SERIES),
    ("ers", err(|x| ers_dfgls_test(x, ErsConfig::default())), OLS),
    (
      "pp",
      err(|x| phillips_perron_test(x, PhillipsPerronConfig::default())),
      SERIES,
    ),
    (
      "lm",
      err(|x| leybourne_mccabe_test(x, LeybourneMcCabeConfig::default())),
      SERIES,
    ),
    (
      "lo_mackinlay",
      err(|x| lo_mackinlay_test(x, LoMacKinlayConfig::default())),
      Some(StatsError::DegenerateData("zero-variance increments")),
    ),
    (
      "cusum",
      err(|x| cusum_test(x, design(x.len()).view(), CusumConfig::default())),
      SERIES,
    ),
    (
      "andrews_ploberger",
      err(|x| andrews_ploberger_test(x, design(x.len()).view(), AndrewsPlobergerConfig::default())),
      SERIES,
    ),
    (
      "reset",
      err(|x| reset_test(x, trend_design(x.len()).view(), ResetConfig::default())),
      SERIES,
    ),
    (
      "anderson_darling",
      err(|x| anderson_darling_normal_test(x, AndersonDarlingConfig::default())),
      SAMPLE,
    ),
    (
      "jarque_bera",
      err(|x| jarque_bera_test(x, JarqueBeraConfig::default())),
      SAMPLE,
    ),
    (
      "shapiro_francia",
      err(|x| shapiro_francia_test(x, ShapiroFranciaConfig::default())),
      SAMPLE,
    ),
    (
      "kolmogorov_smirnov",
      err(|x| {
        kolmogorov_smirnov_test(x, |v| v.clamp(0.0, 1.0), KolmogorovSmirnovConfig::default())
      }),
      None,
    ),
    (
      "ljung_box",
      err(|x| ljung_box(x, LjungBoxConfig::default())),
      SERIES,
    ),
    (
      "engle_granger",
      err(|x| engle_granger_test(x, x, EngleGrangerConfig::default())),
      OLS,
    ),
    (
      "granger",
      err(|x| granger_causality_test(x, x, 1, 0.05)),
      SERIES,
    ),
    ("bns", err(|x| bns_jump_test(x, 0.05)), None),
    ("lee_mykland", err(|x| lee_mykland_test(x, 16, 0.05)), None),
    ("acf", err(|x| acf(x, 20)), SERIES),
    ("pacf", err(|x| pacf(x, 20)), SERIES),
    (
      "johansen",
      err(|x| johansen_test(pair(x).view(), 2)),
      JOHANSEN,
    ),
    ("vecm", err(|x| fit_vecm(pair(x).view(), 2, 1)), JOHANSEN),
  ]
}

pub fn estimators_a() -> Vec<Entry> {
  vec![
    (
      "fit_gev",
      err(fit_gev),
      Some(StatsError::DegenerateData("constant block maxima")),
    ),
    (
      "fit_gpd",
      err(|x| fit_gpd(x.mapv(f64::abs).view())),
      EXCESSES,
    ),
    (
      "fit_gpd_pwm",
      err(|x| fit_gpd_pwm(x.mapv(f64::abs).view())),
      None,
    ),
    ("fit_pot", err(|x| fit_pot(x, 0.0)), EXCESSES),
    (
      "fit_hill",
      err(|x| fit_hill(x.mapv(|v| v.abs() + 1.0).view(), 5)),
      None,
    ),
    ("fit_johnson_su", err(fit_johnson_su), SAMPLE),
    ("fit_skew_t", err(fit_skew_t), SAMPLE),
    ("fit_variance_gamma", err(fit_variance_gamma), SAMPLE),
    (
      "fit_garch",
      err(|x| fit_garch(x, GarchSpec::garch(1, 1))),
      Some(StatsError::DegenerateData("zero-variance returns")),
    ),
    (
      "har_fit",
      err(|x| HarRv::fit(x.mapv(f64::abs).view())),
      SERIES,
    ),
    (
      "har_forecast",
      err(|x| har_model().forecast(x.mapv(f64::abs).view())),
      None,
    ),
    ("ewma", err(|x| ewma_variance(x, 0.94)), None),
    ("riskmetrics", err(riskmetrics_variance), None),
  ]
}

fn check(entries: Vec<Entry>) {
  let empty = Array1::<f64>::zeros(0);
  let mut with_nan = Array1::<f64>::linspace(-1.0, 1.0, 400);
  with_nan[7] = f64::NAN;
  let constant = Array1::<f64>::from_elem(400, 0.3);
  for (name, f, on_constant) in entries {
    assert!(
      matches!(f(empty.view()), Some(StatsError::InsufficientData { .. })),
      "{name}: an empty series must be InsufficientData, got {:?}",
      f(empty.view())
    );
    assert!(
      matches!(
        f(with_nan.view()),
        Some(StatsError::InvalidObservation { index: 7, .. })
      ),
      "{name}: a NaN must be InvalidObservation at its index, got {:?}",
      f(with_nan.view())
    );
    assert_eq!(
      f(constant.view()),
      on_constant,
      "{name}: the answer to a constant series"
    );
  }
}

pub fn estimators_b() -> Vec<Entry> {
  let ou = DiffusionKind::OrnsteinUhlenbeck;
  vec![
    (
      "fit_diffusion_qmle",
      err(move |x| fit_diffusion_qmle(x, DT, ou)),
      SERIES,
    ),
    ("fit_ou_mle", err(|x| fit_ou_mle(x, DT)), SERIES),
    (
      "fit_cir_gmm",
      err(|x| fit_cir_gmm(variances(x).view(), DT)),
      SERIES,
    ),
    (
      "fit_diffusion_bayesian",
      err(move |x| fit_diffusion_bayesian(x, DT, ou, [0.1; 3], 50, 10, 1)),
      SERIES,
    ),
    (
      "fit_heston_nmle",
      err(|x| fit_heston_nmle(prices(x).view(), variances(x).view(), 0.0)),
      VARIANCE,
    ),
    (
      "fit_heston_nmle_with_delta",
      err(|x| fit_heston_nmle_with_delta(prices(x).view(), variances(x).view(), 0.0, DT)),
      VARIANCE,
    ),
    (
      "fit_heston_pmle",
      err(|x| fit_heston_pmle(prices(x).view(), variances(x).view(), 0.0)),
      VARIANCE,
    ),
    (
      "fit_heston_pmle_with_delta",
      err(|x| fit_heston_pmle_with_delta(prices(x).view(), variances(x).view(), 0.0, DT)),
      VARIANCE,
    ),
    (
      "fit_heston_nmle_cekf",
      err(|x| {
        let cfg = HestonNmleCekfConfig {
          max_iters: 2,
          ..HestonNmleCekfConfig::default()
        };
        fit_heston_nmle_cekf(prices(x).view(), cfg)
      }),
      None,
    ),
    (
      "fit_sv_particle_mle",
      err(|x| fit_sv_particle_mle(x, 50, 7)),
      SERIES,
    ),
    (
      "fit_mle",
      err(|x| {
        let mut ou = Ou::new(
          1.0,
          0.5,
          0.5,
          100,
          Some(1.0),
          Some(1.0),
          Deterministic::new(0),
        );
        fit_mle(&mut ou, x, DT, DensityApprox::Euler, None)
      }),
      None,
    ),
    (
      "periodogram",
      err(|x| periodogram_fft(x, PeriodogramConfig::default())),
      None,
    ),
    (
      "fft_spectrum_search",
      err(|x| fft_spectrum_search(x, SpectrumSearchConfig::default())),
      None,
    ),
    (
      "dominant_frequency_fft",
      err(|x| dominant_frequency_fft(x, SpectrumSearchConfig::default())),
      None,
    ),
    (
      "baum_welch",
      err(|x| two_state_hmm().baum_welch(x, 5, 1e-6)),
      SERIES,
    ),
    ("cusum_chart", err(|x| cusum(x, 0.5, 4.0)), SERIES),
    ("pelt", err(|x| pelt(x, 10.0, 2)), None),
  ]
}

#[test]
fn estimators_b_answer_err_on_degenerate_input() {
  check(estimators_b());
}

#[test]
#[should_panic(expected = "min_size must satisfy `min_size >= 1`")]
fn pelt_rejects_a_zero_segment_size() {
  let _ = pelt(Array1::<f64>::linspace(0.0, 1.0, 50).view(), 10.0, 0);
}

#[test]
fn hypothesis_tests_answer_err_on_degenerate_input() {
  check(hypothesis_tests());
}

/// Counts cannot be NaN, so the chi-square test has the empty and constant rows only.
#[test]
fn chi_square_answers_err_on_too_few_bins_and_a_value_on_flat_counts() {
  assert_eq!(
    chi_square_gof_test(&[], &[], ChiSquareGofConfig::default()).unwrap_err(),
    StatsError::InsufficientData { needed: 2, got: 0 }
  );
  let flat = chi_square_gof_test(
    &[3; 400],
    &[1.0 / 400.0; 400],
    ChiSquareGofConfig::default(),
  );
  assert!(flat.is_ok(), "{flat:?}");
}

#[test]
#[should_panic(expected = "alpha must satisfy `0.0 < alpha < 1.0`, got alpha = 1.5")]
fn an_alpha_outside_the_unit_interval_panics() {
  let _ = bns_jump_test(Array1::<f64>::linspace(-1.0, 1.0, 50).view(), 1.5);
}

#[test]
#[should_panic(expected = "lags must satisfy `lags >= 1`, got lags = 0")]
fn zero_lags_panic() {
  let x = Array1::<f64>::linspace(-1.0, 1.0, 50);
  let _ = granger_causality_test(x.view(), x.view(), 0, 0.05);
}

#[test]
fn estimators_a_answer_err_on_degenerate_input() {
  check(estimators_a());
}

#[test]
fn realized_measures_answer_nan_below_their_minimum_length() {
  let empty = Array1::<f64>::zeros(0);
  let one = Array1::<f64>::from_vec(vec![0.01]);
  let two = Array1::<f64>::from_vec(vec![0.01, -0.02]);
  let three = Array1::<f64>::from_vec(vec![0.01, -0.02, 0.015]);
  assert!(realized_variance(empty.view()).is_nan());
  assert!(realized_volatility(empty.view(), 252.0).is_nan());
  let (down, up) = realized_semivariance(empty.view(), 0.0);
  assert!(down.is_nan() && up.is_nan());
  assert!(realized_quarticity(empty.view()).is_nan());
  assert!(realized_skewness(one.view()).is_nan() && realized_kurtosis(one.view()).is_nan());
  let zeros = Array1::<f64>::zeros(5);
  assert!(realized_skewness(zeros.view()).is_nan() && realized_kurtosis(zeros.view()).is_nan());
  assert!(bipower_variation(one.view()).is_nan() && minrv(one.view()).is_nan());
  assert!(medrv(two.view()).is_nan() && tripower_quarticity(two.view()).is_nan());
  assert!(realized_kernel(one.view(), KernelType::Parzen, 1).is_nan());
  assert!(pre_averaged_variance(three.view(), 0.5).is_nan());
  assert!(pre_averaged_variance_with_block(three.view(), 4).is_nan());
  assert!(two_scale_rv(two.view(), 1).is_nan() && multi_scale_rv(three.view(), 2).is_nan());
  let mut with_nan = Array1::<f64>::linspace(-0.01, 0.01, 50);
  with_nan[3] = f64::NAN;
  assert!(medrv(with_nan.view()).is_nan() && bipower_variation(with_nan.view()).is_nan());
  assert!(minrv(with_nan.view()).is_nan());
}
