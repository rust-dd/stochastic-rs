//! Too little, non-finite or constant data is an `Err`, never a panic inside a regression or a sentinel.

use ndarray::Array1;
use ndarray::Array2;
use ndarray::ArrayView1;
use stochastic_rs_stats::StatsError;
use stochastic_rs_stats::acf::LjungBoxConfig;
use stochastic_rs_stats::acf::ljung_box;
use stochastic_rs_stats::econometrics::cointegration::EngleGrangerConfig;
use stochastic_rs_stats::econometrics::cointegration::engle_granger_test;
use stochastic_rs_stats::econometrics::granger::granger_causality_test;
use stochastic_rs_stats::goodness_of_fit::kolmogorov_smirnov::KolmogorovSmirnovConfig;
use stochastic_rs_stats::goodness_of_fit::kolmogorov_smirnov::kolmogorov_smirnov_test;
use stochastic_rs_stats::normality::anderson_darling::AndersonDarlingConfig;
use stochastic_rs_stats::normality::anderson_darling::anderson_darling_normal_test;
use stochastic_rs_stats::normality::jarque_bera::JarqueBeraConfig;
use stochastic_rs_stats::normality::jarque_bera::jarque_bera_test;
use stochastic_rs_stats::normality::shapiro_francia::ShapiroFranciaConfig;
use stochastic_rs_stats::normality::shapiro_francia::shapiro_francia_test;
use stochastic_rs_stats::realized::bipower::bns_jump_test;
use stochastic_rs_stats::realized::lee_mykland::lee_mykland_test;
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

type Answer = Box<dyn Fn(ArrayView1<f64>) -> Option<StatsError>>;

type Entry = (&'static str, Answer);

fn err<T, F: Fn(ArrayView1<f64>) -> Result<T, StatsError> + 'static>(f: F) -> Answer {
  Box::new(move |x| f(x).err())
}

fn design(n: usize) -> Array2<f64> {
  Array2::<f64>::ones((n, 1))
}

pub fn hypothesis_tests() -> Vec<Entry> {
  vec![
    ("adf", err(|x| adf_test(x, AdfConfig::default()))),
    ("kpss", err(|x| kpss_test(x, KpssConfig::default()))),
    ("ers", err(|x| ers_dfgls_test(x, ErsConfig::default()))),
    (
      "pp",
      err(|x| phillips_perron_test(x, PhillipsPerronConfig::default())),
    ),
    (
      "lm",
      err(|x| leybourne_mccabe_test(x, LeybourneMcCabeConfig::default())),
    ),
    (
      "lo_mackinlay",
      err(|x| lo_mackinlay_test(x, LoMacKinlayConfig::default())),
    ),
    (
      "cusum",
      err(|x| cusum_test(x, design(x.len()).view(), CusumConfig::default())),
    ),
    (
      "andrews_ploberger",
      err(|x| andrews_ploberger_test(x, design(x.len()).view(), AndrewsPlobergerConfig::default())),
    ),
    (
      "reset",
      err(|x| reset_test(x, design(x.len()).view(), ResetConfig::default())),
    ),
    (
      "anderson_darling",
      err(|x| anderson_darling_normal_test(x, AndersonDarlingConfig::default())),
    ),
    (
      "jarque_bera",
      err(|x| jarque_bera_test(x, JarqueBeraConfig::default())),
    ),
    (
      "shapiro_francia",
      err(|x| shapiro_francia_test(x, ShapiroFranciaConfig::default())),
    ),
    (
      "kolmogorov_smirnov",
      err(|x| {
        kolmogorov_smirnov_test(x, |v| v.clamp(0.0, 1.0), KolmogorovSmirnovConfig::default())
      }),
    ),
    (
      "ljung_box",
      err(|x| ljung_box(x, LjungBoxConfig::default())),
    ),
    (
      "engle_granger",
      err(|x| engle_granger_test(x, x, EngleGrangerConfig::default())),
    ),
    ("granger", err(|x| granger_causality_test(x, x, 1, 0.05))),
    ("bns", err(|x| bns_jump_test(x, 0.05))),
    ("lee_mykland", err(|x| lee_mykland_test(x, 16, 0.05))),
  ]
}

fn check(entries: Vec<Entry>) {
  let empty = Array1::<f64>::zeros(0);
  let mut with_nan = Array1::<f64>::linspace(-1.0, 1.0, 400);
  with_nan[7] = f64::NAN;
  let constant = Array1::<f64>::from_elem(400, 0.3);
  for (name, f) in entries {
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
    assert!(
      matches!(
        f(constant.view()),
        Some(
          StatsError::DegenerateData(_) | StatsError::Singular(_) | StatsError::Numerical { .. }
        ) | None
      ),
      "{name}: a constant series must be DegenerateData, Singular, Numerical or a value, got {:?}",
      f(constant.view())
    );
  }
}

#[test]
fn hypothesis_tests_answer_err_on_degenerate_input() {
  check(hypothesis_tests());
}
