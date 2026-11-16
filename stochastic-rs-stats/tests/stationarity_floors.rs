//! The stationarity statistics keep their value when the series is rescaled, and a variance that is
//! zero or not finite is an `Err`.

use ndarray::Array1;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_distributions::DistributionSampler;
use stochastic_rs_distributions::SimdDistribution;
use stochastic_rs_distributions::normal::SimdNormal;
use stochastic_rs_stats::StatsError;
use stochastic_rs_stats::stationarity::Trend;
use stochastic_rs_stats::stationarity::kpss::KpssConfig;
use stochastic_rs_stats::stationarity::kpss::kpss_test;
use stochastic_rs_stats::stationarity::leybourne_mccabe::LeybourneMcCabeConfig;
use stochastic_rs_stats::stationarity::leybourne_mccabe::leybourne_mccabe_test;
use stochastic_rs_stats::stationarity::phillips_perron::PhillipsPerronConfig;
use stochastic_rs_stats::stationarity::phillips_perron::PpTestType;
use stochastic_rs_stats::stationarity::phillips_perron::phillips_perron_test;

/// A random walk of step 0.01 plus white noise of 0.02, 400 points.
fn series(seed: u64) -> Array1<f64> {
  let mut steps = vec![0.0; 400];
  let mut noise = vec![0.0; 400];
  SimdNormal::<f64>::new(0.0, 1.0)
    .seeded(&Deterministic::new(seed))
    .fill_slice(&mut steps);
  SimdNormal::<f64>::new(0.0, 1.0)
    .seeded(&Deterministic::new(seed + 1))
    .fill_slice(&mut noise);
  let mut walk = 0.0;
  Array1::from_iter(steps.iter().zip(&noise).map(|(step, e)| {
    walk += step;
    0.01 * walk + 0.02 * e
  }))
}

const SEEDS: [u64; 3] = [7, 11, 13];

/// Where an absolute variance floor of `1e-12` bound these statistics.
const SCALES: [f64; 3] = [1e-5, 1e-6, 1e-7];

fn assert_rescaled(what: &str, statistic: impl Fn(&Array1<f64>) -> f64, scales: &[f64]) {
  for seed in SEEDS {
    let y = series(seed);
    let unit = statistic(&y);
    for &c in scales {
      let scaled = statistic(&y.mapv(|v| v * c));
      assert!(
        (scaled - unit).abs() <= 1e-12 * unit.abs(),
        "{what}, seed {seed}, c = {c:e}: {scaled} on c·y against {unit} on y"
      );
    }
  }
}

#[test]
fn kpss_keeps_its_statistic_under_rescaling() {
  for trend in [Trend::Constant, Trend::ConstantTrend] {
    let cfg = KpssConfig {
      trend,
      ..KpssConfig::default()
    };
    assert_rescaled(
      &format!("KPSS {trend:?}"),
      |y| kpss_test(y.view(), cfg).unwrap().statistic,
      &SCALES,
    );
  }
}

#[test]
fn phillips_perron_keeps_its_statistics_under_rescaling() {
  for trend in [Trend::None, Trend::Constant, Trend::ConstantTrend] {
    for test_type in [PpTestType::Tau, PpTestType::Rho] {
      let cfg = PhillipsPerronConfig {
        trend,
        test_type,
        ..PhillipsPerronConfig::default()
      };
      assert_rescaled(
        &format!("PP {test_type:?} {trend:?}"),
        |y| phillips_perron_test(y.view(), cfg).unwrap().statistic,
        &[1e-5, 1e-6, 1e-7, 1e4],
      );
    }
  }
}

/// The bootstrap paths scale with the series, so its p-value is a count that does not move.
#[test]
fn leybourne_mccabe_keeps_its_statistic_and_p_value_under_rescaling() {
  let scales = [1e-5, 1e-6, 1e-7, 1e7];
  for trend in [Trend::Constant, Trend::ConstantTrend] {
    let cfg = LeybourneMcCabeConfig {
      trend,
      bootstrap_samples: 99,
      ..LeybourneMcCabeConfig::default()
    };
    let lm = |y: &Array1<f64>| leybourne_mccabe_test(y.view(), cfg).unwrap();
    assert_rescaled(&format!("LM {trend:?}"), |y| lm(y).statistic, &scales);
    for seed in SEEDS {
      let y = series(seed);
      for c in scales {
        assert_eq!(
          lm(&y.mapv(|v| v * c)).p_value,
          lm(&y).p_value,
          "LM {trend:?}, seed {seed}, c = {c:e}"
        );
      }
    }
  }
}

#[test]
fn kpss_answers_a_long_run_variance_that_overflows_with_numerical() {
  let y = series(7).mapv(|v| v * 1e200);
  assert_eq!(
    kpss_test(y.view(), KpssConfig::default()).unwrap_err(),
    StatsError::Numerical {
      stage: "long-run variance"
    }
  );
}

/// A line is fitted exactly by the Dickey–Fuller regression, so its residuals are all zero.
#[test]
fn phillips_perron_answers_zero_residuals_with_degenerate_data() {
  let y = Array1::from_iter((0..30).map(|t| t as f64));
  for test_type in [PpTestType::Tau, PpTestType::Rho] {
    let cfg = PhillipsPerronConfig {
      test_type,
      ..PhillipsPerronConfig::default()
    };
    assert_eq!(
      phillips_perron_test(y.view(), cfg).unwrap_err(),
      StatsError::DegenerateData("zero long-run variance")
    );
  }
}

/// An alternating series is an exact AR(1) with coefficient −1 about its zero mean.
#[test]
fn leybourne_mccabe_answers_an_exact_ar_fit_with_degenerate_data() {
  let y = Array1::from_iter((0..40).map(|t| if t % 2 == 0 { 1.0 } else { -1.0 }));
  assert_eq!(
    leybourne_mccabe_test(y.view(), LeybourneMcCabeConfig::default()).unwrap_err(),
    StatsError::DegenerateData("zero residual variance")
  );
}
