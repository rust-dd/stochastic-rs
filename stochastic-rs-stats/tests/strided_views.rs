//! A strided view and its contiguous copy give the same statistic.

use ndarray::Array1;
use ndarray::Array2;
use ndarray::ArrayView1;
use ndarray::s;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_distributions::DistributionSampler;
use stochastic_rs_distributions::SimdDistribution;
use stochastic_rs_distributions::normal::SimdNormal;
use stochastic_rs_stats::StatsError;
use stochastic_rs_stats::fou::FilterType;
use stochastic_rs_stats::fou::FouFit;
use stochastic_rs_stats::fou::fit_fou_central_differences;
use stochastic_rs_stats::fou::fit_fou_filtered_variations;
use stochastic_rs_stats::fou::fit_fou_power_variation;
use stochastic_rs_stats::hurst::Whittle;
use stochastic_rs_stats::kde::silverman_bandwidth;
use stochastic_rs_stats::normality::anderson_darling::AndersonDarlingConfig;
use stochastic_rs_stats::normality::anderson_darling::anderson_darling_normal_test;
use stochastic_rs_stats::normality::jarque_bera::JarqueBeraConfig;
use stochastic_rs_stats::normality::jarque_bera::jarque_bera_test;
use stochastic_rs_stats::normality::shapiro_francia::ShapiroFranciaConfig;
use stochastic_rs_stats::normality::shapiro_francia::shapiro_francia_test;
use stochastic_rs_stats::spectral::PeriodogramConfig;
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
  let a = periodogram_fft(strided, PeriodogramConfig::default())
    .unwrap()
    .spectrum;
  let b = periodogram_fft(contiguous.view(), PeriodogramConfig::default())
    .unwrap()
    .spectrum;
  assert_eq!(a, b);
  let short = full.slice(s![..80;2]);
  let whittle = Whittle::new(72, 1.0 / 250.0);
  let a = whittle.fit(short).unwrap();
  let b = whittle.fit(short.to_owned().view()).unwrap();
  assert_eq!(
    (a.hurst, a.eta, a.neg_log_lik),
    (b.hurst, b.eta, b.neg_log_lik)
  );
}

#[test]
fn silverman_bandwidth_of_a_strided_or_reversed_view_matches_its_copy() {
  let mut normal = SimdNormal::<f64>::new(100.0, 3.0).seeded(&Deterministic::new(7));
  let (mut strided_moves, mut reversed_moves) = (0, 0);
  for trial in 0..2000 {
    let n = 9 + trial % 192;
    let mut copy = Array1::<f64>::zeros(n);
    normal.fill_slice(copy.as_slice_mut().unwrap());
    let mut interleaved = Array1::<f64>::zeros(2 * n);
    interleaved.slice_mut(s![..;2]).assign(&copy);
    let backwards = Array1::from_iter(copy.iter().rev().copied());
    let want = silverman_bandwidth(copy.view()).unwrap().to_bits();
    let strided = silverman_bandwidth(interleaved.slice(s![..;2])).unwrap();
    let reversed = silverman_bandwidth(backwards.slice(s![..;-1])).unwrap();
    strided_moves += usize::from(strided.to_bits() != want);
    reversed_moves += usize::from(reversed.to_bits() != want);
  }
  assert_eq!(
    (strided_moves, reversed_moves),
    (0, 0),
    "(strided, reversed) bandwidths of 2000 that differ from the copy's"
  );
}

#[test]
fn fou_fits_of_a_strided_or_reversed_view_match_their_copy() {
  fn bits(fit: Result<FouFit, StatsError>) -> Result<[u64; 4], StatsError> {
    fit.map(|f| [f.hurst, f.sigma, f.mu, f.theta].map(f64::to_bits))
  }
  let fits = |x: ArrayView1<f64>| {
    [
      bits(fit_fou_filtered_variations(
        x,
        FilterType::Daubechies,
        None,
        None,
      )),
      bits(fit_fou_central_differences(x, None, None)),
      bits(fit_fou_power_variation(x, None, 2, 2.0, None, None)),
    ]
  };
  let mut normal = SimdNormal::<f64>::new(0.0, 0.1).seeded(&Deterministic::new(11));
  let mut moves = [[0usize; 3]; 2];
  for trial in 0..600 {
    let n = 9 + trial % 192;
    let mut steps = vec![0.0; n];
    normal.fill_slice(&mut steps);
    let copy = Array1::from_iter(steps.iter().scan(2.0, |level, &z| {
      *level += z;
      Some(*level)
    }));
    let mut interleaved = Array1::<f64>::zeros(2 * n);
    interleaved.slice_mut(s![..;2]).assign(&copy);
    let backwards = Array1::from_iter(copy.iter().rev().copied());
    let want = fits(copy.view());
    for (view, row) in [interleaved.slice(s![..;2]), backwards.slice(s![..;-1])]
      .into_iter()
      .zip(&mut moves)
    {
      for (got, (want, count)) in fits(view).into_iter().zip(want.iter().zip(row)) {
        *count += usize::from(got != *want);
      }
    }
  }
  assert_eq!(
    moves, [[0; 3]; 2],
    "[strided, reversed] x [filtered, central, power] fits of 600 that differ from the copy's"
  );
}
