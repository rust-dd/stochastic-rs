//! Every hypothesis test answers `statistic`, `p_value`, `critical_values`, `rejects_null`, and the trait agrees.

use ndarray::Array1;
use ndarray::Array2;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_distributions::DistributionSampler;
use stochastic_rs_distributions::SimdDistribution;
use stochastic_rs_distributions::normal::SimdNormal;
use stochastic_rs_stats::acf::LjungBoxConfig;
use stochastic_rs_stats::acf::ljung_box;
use stochastic_rs_stats::critical_values::CriticalValues;
use stochastic_rs_stats::econometrics::cointegration::EngleGrangerConfig;
use stochastic_rs_stats::econometrics::cointegration::engle_granger_test;
use stochastic_rs_stats::econometrics::granger::granger_causality_test;
use stochastic_rs_stats::goodness_of_fit::chi_square::ChiSquareGofConfig;
use stochastic_rs_stats::goodness_of_fit::chi_square::chi_square_gof_test;
use stochastic_rs_stats::goodness_of_fit::kolmogorov_smirnov::KolmogorovSmirnovConfig;
use stochastic_rs_stats::goodness_of_fit::kolmogorov_smirnov::kolmogorov_smirnov_test;
use stochastic_rs_stats::normality::anderson_darling::AndersonDarlingConfig;
use stochastic_rs_stats::normality::anderson_darling::anderson_darling_normal_test;
use stochastic_rs_stats::normality::jarque_bera::JarqueBeraConfig;
use stochastic_rs_stats::normality::jarque_bera::jarque_bera_test;
use stochastic_rs_stats::normality::shapiro_francia::ShapiroFranciaConfig;
use stochastic_rs_stats::normality::shapiro_francia::shapiro_francia_test;
use stochastic_rs_stats::realized::bipower::bns_jump_test;
use stochastic_rs_stats::stationarity::Trend;
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
use stochastic_rs_stats::traits::HypothesisTest;

fn ar1(seed: u64, n: usize, phi: f64) -> Array1<f64> {
  let mut dist = SimdNormal::<f64>::new(0.0, 1.0).seeded(&Deterministic::new(seed));
  let mut eps = vec![0.0_f64; n];
  dist.fill_slice(&mut eps);
  let mut x = Array1::<f64>::zeros(n);
  for i in 1..n {
    x[i] = phi * x[i - 1] + eps[i];
  }
  x
}

fn agrees<T: HypothesisTest>(result: &T, statistic: f64, rejects: bool) {
  assert_eq!(result.statistic(), statistic);
  assert_eq!(result.null_rejected(), Some(rejects));
}

#[test]
fn critical_values_pick_the_conservative_tabulated_level() {
  let cv = CriticalValues::new([(0.10, -2.57), (0.01, -3.43), (0.05, -2.86)]);
  assert_eq!(cv.entries(), &[(0.01, -3.43), (0.05, -2.86), (0.10, -2.57)]);
  assert_eq!(cv.at(0.05), Some(-2.86));
  assert_eq!(cv.at(0.03), None);
  assert_eq!(cv.value_at(0.01), -3.43);
  assert_eq!(cv.value_at(0.03), -3.43);
  assert_eq!(cv.value_at(0.05), -2.86);
  assert_eq!(cv.value_at(0.07), -2.86);
  assert_eq!(cv.value_at(0.10), -2.57);
}

#[test]
fn a_level_above_the_table_decides_at_its_largest_level() {
  let cv = CriticalValues::new([(0.01, -3.43), (0.05, -2.86), (0.10, -2.57)]);
  assert_eq!(cv.value_at(0.2), -2.57);
}

#[test]
#[should_panic(
  expected = "alpha must satisfy `alpha >= 0.01` (the smallest tabulated level), got alpha = 0.005"
)]
fn a_level_below_the_table_is_a_precondition() {
  let _ = CriticalValues::new([(0.01, -3.43), (0.05, -2.86), (0.10, -2.57)]).value_at(0.005);
}

#[test]
#[should_panic(expected = "level must satisfy `0.0 < level < 1.0`")]
fn a_level_outside_the_unit_interval_is_a_precondition() {
  let _ = CriticalValues::new([(1.5, 0.0)]);
}

#[test]
#[should_panic(expected = "entries must satisfy `!entries.is_empty()`, got entries = []")]
fn an_empty_table_is_a_precondition() {
  let _ = CriticalValues::new([]);
}

#[test]
fn every_test_result_answers_through_the_trait() {
  let x = ar1(1, 400, 0.5);
  let adf = adf_test(x.view(), AdfConfig::default()).unwrap();
  agrees(&adf, adf.statistic, adf.rejects_null);
  assert!(adf.p_value.is_none() && adf.critical_values.is_some());
  let kpss = kpss_test(x.view(), KpssConfig::default()).unwrap();
  agrees(&kpss, kpss.statistic, kpss.rejects_null);
  assert_eq!(kpss.critical_values.as_ref().unwrap().entries().len(), 4);
  let ers = ers_dfgls_test(x.view(), ErsConfig::default()).unwrap();
  agrees(&ers, ers.statistic, ers.rejects_null);
  let pp = phillips_perron_test(x.view(), PhillipsPerronConfig::default()).unwrap();
  assert_eq!(pp.null_rejected(), pp.rejects_null);
  let lm = leybourne_mccabe_test(x.view(), LeybourneMcCabeConfig::default()).unwrap();
  agrees(&lm, lm.statistic, lm.rejects_null);
  assert!(lm.p_value.is_some() && lm.critical_values.is_none());
  let lomk = lo_mackinlay_test(x.view(), LoMacKinlayConfig::default()).unwrap();
  agrees(&lomk, lomk.statistic, lomk.rejects_null);
  assert!(lomk.p_value.unwrap() >= 0.0 && lomk.p_value_iid >= 0.0);
  let design = Array2::<f64>::ones((x.len(), 1));
  let cusum = cusum_test(x.view(), design.view(), CusumConfig::default()).unwrap();
  agrees(&cusum, cusum.statistic, cusum.rejects_null);
  let ap =
    andrews_ploberger_test(x.view(), design.view(), AndrewsPlobergerConfig::default()).unwrap();
  assert_eq!(ap.statistic(), ap.statistic);
  assert_eq!(ap.null_rejected(), ap.rejects_null);
  assert!(ap.rejects_null.is_some());
  // RESET's powers of a fit on the intercept alone would be collinear with it.
  let trend =
    Array2::<f64>::from_shape_fn((x.len(), 2), |(t, j)| if j == 0 { 1.0 } else { t as f64 });
  let reset = reset_test(x.view(), trend.view(), ResetConfig::default()).unwrap();
  agrees(&reset, reset.statistic, reset.rejects_null);
  let ad = anderson_darling_normal_test(x.view(), AndersonDarlingConfig::default()).unwrap();
  agrees(&ad, ad.statistic, ad.rejects_null);
  let jb = jarque_bera_test(x.view(), JarqueBeraConfig::default()).unwrap();
  agrees(&jb, jb.statistic, jb.rejects_null);
  let sf = shapiro_francia_test(x.view(), ShapiroFranciaConfig::default()).unwrap();
  agrees(&sf, sf.statistic, sf.rejects_null);
  let ks = kolmogorov_smirnov_test(
    x.view(),
    |v| 0.5 * (1.0 + libm_erf(v / 2.0_f64.sqrt())),
    KolmogorovSmirnovConfig::default(),
  )
  .unwrap();
  agrees(&ks, ks.statistic, ks.rejects_null);
  assert_eq!(ks.critical_values.as_ref().unwrap().entries().len(), 1);
  let chi = chi_square_gof_test(
    &[40, 35, 25],
    &[0.4, 0.35, 0.25],
    ChiSquareGofConfig::default(),
  )
  .unwrap();
  agrees(&chi, chi.statistic, chi.rejects_null);
  let lb = ljung_box(
    x.view(),
    LjungBoxConfig {
      lags: 10,
      fit_df: 0,
      alpha: 0.05,
    },
  )
  .unwrap();
  agrees(&lb, lb.statistic, lb.rejects_null);
  assert_eq!(lb.rejects_null, lb.p_value.unwrap() < 0.05);
  let y = ar1(2, 400, 0.5);
  let granger = granger_causality_test(y.view(), x.view(), 2, 0.05).unwrap();
  agrees(&granger, granger.statistic, granger.rejects_null);
  let eg = engle_granger_test(y.view(), x.view(), EngleGrangerConfig::default()).unwrap();
  agrees(&eg, eg.statistic, eg.rejects_null);
  assert!(eg.p_value.is_none() && eg.critical_values.as_ref().unwrap().at(0.05).is_some());
  let bns = bns_jump_test(x.view(), 0.05).unwrap();
  agrees(&bns, bns.statistic, bns.rejects_null);
  assert!(bns.p_value.is_some());
}

fn libm_erf(x: f64) -> f64 {
  stochastic_rs_distributions::special::erf(x)
}

#[test]
fn engle_granger_reports_the_intercept() {
  let x = ar1(7, 500, 1.0);
  let mut dist = SimdNormal::<f64>::new(0.0, 0.05).seeded(&Deterministic::new(11));
  let mut eps = vec![0.0_f64; 500];
  dist.fill_slice(&mut eps);
  let y = Array1::from_iter((0..500).map(|i| 2.0 + 0.7 * x[i] + eps[i]));
  let eg = engle_granger_test(y.view(), x.view(), EngleGrangerConfig { alpha: 0.05 }).unwrap();
  assert!((eg.intercept - 2.0).abs() < 0.1 && (eg.beta - 0.7).abs() < 0.05);
}

#[test]
#[should_panic(expected = "trend must satisfy `trend != Trend::None`")]
fn kpss_has_no_trend_free_regression() {
  let x = ar1(3, 200, 0.3);
  let _ = kpss_test(
    x.view(),
    KpssConfig {
      trend: Trend::None,
      ..KpssConfig::default()
    },
  );
}
