//! A strided view and its contiguous copy give the same statistic.

use ndarray::Array1;
use ndarray::Array2;
use ndarray::s;
use stochastic_rs_stats::normality::anderson_darling::AndersonDarlingConfig;
use stochastic_rs_stats::normality::anderson_darling::anderson_darling_normal_test;
use stochastic_rs_stats::normality::jarque_bera::JarqueBeraConfig;
use stochastic_rs_stats::normality::jarque_bera::jarque_bera_test;
use stochastic_rs_stats::normality::shapiro_francia::ShapiroFranciaConfig;
use stochastic_rs_stats::normality::shapiro_francia::shapiro_francia_test;
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

fn series() -> Array1<f64> {
  Array1::from_iter((0..800).map(|i| ((i as f64) * 0.37).sin() + 0.001 * i as f64))
}

#[test]
fn strided_views_match_their_contiguous_copies() {
  let full = series();
  let strided = full.slice(s![..;2]);
  assert!(strided.as_slice().is_none());
  let contiguous = strided.to_owned();
  let design = Array2::<f64>::ones((strided.len(), 1));
  // RESET's powers of a fit on the intercept alone would be collinear with it.
  let trend = Array2::<f64>::from_shape_fn(
    (strided.len(), 2),
    |(t, j)| if j == 0 { 1.0 } else { t as f64 },
  );
  macro_rules! same {
    ($name:literal, $f:expr) => {{
      let a = $f(strided).unwrap().statistic;
      let b = $f(contiguous.view()).unwrap().statistic;
      assert_eq!(a, b, $name);
    }};
  }
  same!("adf", |x| adf_test(x, AdfConfig::default()));
  same!("kpss", |x| kpss_test(x, KpssConfig::default()));
  same!("ers", |x| ers_dfgls_test(x, ErsConfig::default()));
  same!("pp", |x| phillips_perron_test(
    x,
    PhillipsPerronConfig::default()
  ));
  same!("lm", |x| leybourne_mccabe_test(
    x,
    LeybourneMcCabeConfig::default()
  ));
  same!("lo_mackinlay", |x| lo_mackinlay_test(
    x,
    LoMacKinlayConfig::default()
  ));
  same!("cusum", |x| cusum_test(
    x,
    design.view(),
    CusumConfig::default()
  ));
  same!("andrews_ploberger", |x| andrews_ploberger_test(
    x,
    design.view(),
    AndrewsPlobergerConfig::default()
  ));
  same!("reset", |x| reset_test(
    x,
    trend.view(),
    ResetConfig::default()
  ));
  same!("anderson_darling", |x| anderson_darling_normal_test(
    x,
    AndersonDarlingConfig::default()
  ));
  same!("jarque_bera", |x| jarque_bera_test(
    x,
    JarqueBeraConfig::default()
  ));
  same!("shapiro_francia", |x| shapiro_francia_test(
    x,
    ShapiroFranciaConfig::default()
  ));
}
