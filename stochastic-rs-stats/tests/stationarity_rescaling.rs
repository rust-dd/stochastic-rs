//! The stationarity statistics keep their value when the series is rescaled, and a variance that is
//! zero or not finite is an `Err`.

use ndarray::Array1;
use ndarray::Array2;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_distributions::DistributionSampler;
use stochastic_rs_distributions::SimdDistribution;
use stochastic_rs_distributions::normal::SimdNormal;
use stochastic_rs_stats::StatsError;
use stochastic_rs_stats::econometrics::cointegration::EngleGrangerConfig;
use stochastic_rs_stats::econometrics::cointegration::engle_granger_test;
use stochastic_rs_stats::econometrics::granger::granger_causality_test;
use stochastic_rs_stats::stationarity::LagSelection;
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
use stochastic_rs_stats::stationarity::phillips_perron::PhillipsPerronConfig;
use stochastic_rs_stats::stationarity::phillips_perron::PpTestType;
use stochastic_rs_stats::stationarity::phillips_perron::phillips_perron_test;
use stochastic_rs_stats::stationarity::reset::ResetConfig;
use stochastic_rs_stats::stationarity::reset::reset_test;

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

/// Scales several decades below unit variance.
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

/// From the foot of the double range to its top, where a square under- or overflows unscaled, and
/// `1e±6`, where a criterion compared over samples of different sizes would move the lag.
const FAR_SCALES: [f64; 9] = [
  1e-300, 1e-200, 1e-160, 1e-150, 1e-6, 1e6, 1e154, 1e156, 1e300,
];

#[test]
fn adf_and_df_gls_keep_their_statistic_under_rescaling() {
  for lag_selection in [LagSelection::Aic, LagSelection::Bic, LagSelection::TStat] {
    for trend in [Trend::None, Trend::Constant, Trend::ConstantTrend] {
      let cfg = AdfConfig {
        trend,
        lag_selection,
        ..AdfConfig::default()
      };
      assert_rescaled(
        &format!("ADF {trend:?} {lag_selection:?}"),
        |y| adf_test(y.view(), cfg).unwrap().statistic,
        &FAR_SCALES,
      );
    }
    for trend in [Trend::Constant, Trend::ConstantTrend] {
      let cfg = ErsConfig {
        trend,
        lag_selection,
        ..ErsConfig::default()
      };
      assert_rescaled(
        &format!("ERS {trend:?} {lag_selection:?}"),
        |y| ers_dfgls_test(y.view(), cfg).unwrap().statistic,
        &FAR_SCALES,
      );
    }
  }
}

/// `x` is `y` plus noise of a fifth of its largest step, which no square underflows at `1e-300`.
#[test]
fn engle_granger_keeps_its_statistic_under_rescaling() {
  let mut noise = vec![0.0; 400];
  SimdNormal::<f64>::new(0.0, 1.0)
    .seeded(&Deterministic::new(99))
    .fill_slice(&mut noise);
  let engle_granger = |y: &Array1<f64>| {
    let steps = y.slice(ndarray::s![1..]).to_owned() - y.slice(ndarray::s![..-1]);
    let spread = 0.4 * steps.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let x = Array1::from_iter(y.iter().zip(&noise).map(|(v, e)| v + 0.5 * spread * e));
    engle_granger_test(y.view(), x.view(), EngleGrangerConfig::default())
      .unwrap()
      .statistic
  };
  assert_rescaled("Engle-Granger", engle_granger, &FAR_SCALES);
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
        &[1e-300, 1e-150, 1e-7, 1e-6, 1e-5, 1e4, 1e7, 1e150, 1e300],
      );
    }
  }
}

/// The bootstrap paths scale with the series, so its p-value is a count that does not move.
#[test]
fn leybourne_mccabe_keeps_its_statistic_and_p_value_under_rescaling() {
  let scales = [1e-300, 1e-160, 1e-5, 1e-6, 1e-7, 1e7, 1e154, 1e300];
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

/// KPSS is homogeneous in the series and scales it by a power of two, so no scale is out of range.
#[test]
fn kpss_keeps_its_statistic_from_1e_minus_300_to_1e300() {
  let scales = [
    1e-300, 1e-200, 1e-162, 1e-160, 1e-156, 1e-150, 1e150, 1e154, 1e200, 1e300,
  ];
  for trend in [Trend::Constant, Trend::ConstantTrend] {
    let cfg = KpssConfig {
      trend,
      ..KpssConfig::default()
    };
    assert_rescaled(
      &format!("KPSS {trend:?}"),
      |y| kpss_test(y.view(), cfg).unwrap().statistic,
      &scales,
    );
  }
}

/// A mean of a series this large overflows unless the series is scaled first.
#[test]
fn a_finite_series_near_the_largest_double_is_centred_after_it_is_scaled() {
  let scale = 1e306;
  for trend in [Trend::Constant, Trend::ConstantTrend] {
    let pp = PhillipsPerronConfig {
      trend,
      ..PhillipsPerronConfig::default()
    };
    let kpss = KpssConfig {
      trend,
      ..KpssConfig::default()
    };
    let shifted = |f: &dyn Fn(&Array1<f64>) -> f64| {
      let unit = f(&series(7).mapv(|v| v + 5.0));
      let big = f(&series(7).mapv(|v| (v + 5.0) * scale));
      assert!(
        (big - unit).abs() <= 1e-9 * unit.abs(),
        "{trend:?}: {big} at 1e306 against {unit}"
      );
    };
    shifted(&|y| phillips_perron_test(y.view(), pp).unwrap().statistic);
    shifted(&|y| kpss_test(y.view(), kpss).unwrap().statistic);
  }
}

/// A lagged level that is a constant, or a line, leaves no design whatever the last point is.
#[test]
fn a_flat_or_linear_lagged_level_is_degenerate_data_whatever_the_last_point() {
  let flat_then = |level: f64, n: usize, last: f64| {
    Array1::from_iter((0..n).map(|t| if t + 1 == n { last } else { level }))
  };
  let line_then = Array1::from_iter((0..400).map(|t| {
    if t == 399 {
      -2.0
    } else {
      3.0 + 0.37 * t as f64
    }
  }));
  let pp = |trend| PhillipsPerronConfig {
    trend,
    ..PhillipsPerronConfig::default()
  };
  let zero = StatsError::DegenerateData("zero residual variance");
  for (what, y, trend) in [
    (
      "399 × 0.1 then 0.3",
      flat_then(0.1, 400, 0.3),
      Trend::Constant,
    ),
    ("3 + 0.37 t then −2", line_then, Trend::ConstantTrend),
    ("29 × 5 then 7", flat_then(5.0, 30, 7.0), Trend::Constant),
  ] {
    assert_eq!(
      phillips_perron_test(y.view(), pp(trend)).unwrap_err(),
      zero,
      "{what}, {trend:?}"
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

/// Each series is its test's regression to rounding, at a level of 0 or `1e5` and at every scale:
/// the residuals then carry no variance against the series' own.
#[test]
fn an_exact_fit_is_degenerate_data_at_every_level_and_scale() {
  let line = |level: f64| Array1::from_iter((0..30).map(|t| level + t as f64));
  let power = Array1::from_iter((0..80).map(|t| 1.5f64.powi(t)));
  let square = Array1::from_iter((0..100).map(|t| (t * t) as f64));
  let pp = |trend| PhillipsPerronConfig {
    trend,
    ..PhillipsPerronConfig::default()
  };
  let kpss = KpssConfig {
    trend: Trend::ConstantTrend,
    ..KpssConfig::default()
  };
  let zero = StatsError::DegenerateData("zero residual variance");
  for c in [1e-6, 1.0, 1e6] {
    for level in [0.0, 1e5] {
      let y = line(level).mapv(|v| v * c);
      for trend in [Trend::Constant, Trend::ConstantTrend] {
        let got = phillips_perron_test(y.view(), pp(trend));
        assert_eq!(
          got.unwrap_err(),
          zero,
          "PP {trend:?}, line at {level:e}, c = {c:e}"
        );
      }
      let got = kpss_test(y.view(), kpss);
      assert_eq!(got.unwrap_err(), zero, "KPSS, line at {level:e}, c = {c:e}");
    }
    for trend in [Trend::None, Trend::Constant, Trend::ConstantTrend] {
      let got = phillips_perron_test(power.mapv(|v| v * c).view(), pp(trend));
      assert_eq!(got.unwrap_err(), zero, "PP {trend:?}, 1.5^t, c = {c:e}");
    }
    let got = phillips_perron_test(square.mapv(|v| v * c).view(), pp(Trend::ConstantTrend));
    assert_eq!(got.unwrap_err(), zero, "PP ConstantTrend, t^2, c = {c:e}");
  }
}

/// Doubles within an ulp of `0.7` are the rounding of a constant to every test, and the same steps
/// eight ulps apart are a series to test.
#[test]
fn a_series_of_neighbouring_doubles_is_a_constant_series_for_every_test() {
  let ulp = f64::EPSILON / 2.0;
  let mut noise = vec![0.0; 200];
  SimdNormal::<f64>::new(0.0, 1.0)
    .seeded(&Deterministic::new(5))
    .fill_slice(&mut noise);
  let steps = Array1::from_iter(noise.iter().map(|e| e.signum()));
  let y = steps.mapv(|j| 0.7 + j * ulp);
  let x = series(7).slice(ndarray::s![..200]).to_owned();
  let design = Array2::from_shape_fn((200, 2), |(i, j)| if j == 0 { 1.0 } else { x[i] });
  let answers = [
    ("ADF", adf_test(y.view(), AdfConfig::default()).err()),
    ("ERS", ers_dfgls_test(y.view(), ErsConfig::default()).err()),
    (
      "PP",
      phillips_perron_test(y.view(), PhillipsPerronConfig::default()).err(),
    ),
    ("KPSS", kpss_test(y.view(), KpssConfig::default()).err()),
    (
      "LM",
      leybourne_mccabe_test(y.view(), LeybourneMcCabeConfig::default()).err(),
    ),
    (
      "Engle-Granger y",
      engle_granger_test(y.view(), x.view(), EngleGrangerConfig::default()).err(),
    ),
    (
      "Engle-Granger x",
      engle_granger_test(x.view(), y.view(), EngleGrangerConfig::default()).err(),
    ),
    (
      "Granger y",
      granger_causality_test(y.view(), x.view(), 2, 0.05).err(),
    ),
    (
      "Granger x",
      granger_causality_test(x.view(), y.view(), 2, 0.05).err(),
    ),
    (
      "RESET",
      reset_test(y.view(), design.view(), ResetConfig::default()).err(),
    ),
    (
      "CUSUM",
      cusum_test(y.view(), design.view(), CusumConfig::default()).err(),
    ),
    (
      "Andrews-Ploberger",
      andrews_ploberger_test(y.view(), design.view(), AndrewsPlobergerConfig::default()).err(),
    ),
  ];
  for (what, answer) in answers {
    assert_eq!(
      answer,
      Some(StatsError::DegenerateData("constant series")),
      "{what}"
    );
  }
  let wide = steps.mapv(|j| 0.7 + 4.0 * j * ulp);
  assert!(adf_test(wide.view(), AdfConfig::default()).is_ok());
  assert!(kpss_test(wide.view(), KpssConfig::default()).is_ok());
}

/// Noise of `1e-7` about `1e8` tests as the same points less `1e8`, a subtraction that is exact.
#[test]
fn a_level_far_above_the_variation_keeps_the_statistic() {
  let mut noise = vec![0.0; 200];
  SimdNormal::<f64>::new(0.0, 1.0)
    .seeded(&Deterministic::new(3))
    .fill_slice(&mut noise);
  let y = Array1::from_iter(noise.iter().map(|e| 1e8 + 1e-7 * e));
  let w = y.mapv(|v| v - 1e8);
  for trend in [Trend::Constant, Trend::ConstantTrend] {
    let cfg = PhillipsPerronConfig {
      trend,
      ..PhillipsPerronConfig::default()
    };
    let (at_level, less_level) = (
      phillips_perron_test(y.view(), cfg).map(|r| r.statistic),
      phillips_perron_test(w.view(), cfg).unwrap().statistic,
    );
    assert!(
      at_level
        .as_ref()
        .is_ok_and(|s| (s - less_level).abs() <= 1e-12 * less_level.abs()),
      "PP {trend:?}: {at_level:?} at 1e8 against {less_level}"
    );
    let cfg = KpssConfig {
      trend,
      ..KpssConfig::default()
    };
    let (at_level, less_level) = (
      kpss_test(y.view(), cfg).unwrap().statistic,
      kpss_test(w.view(), cfg).unwrap().statistic,
    );
    assert!(
      (at_level - less_level).abs() <= 1e-12 * less_level.abs(),
      "KPSS {trend:?}: {at_level} at 1e8 against {less_level}"
    );
    for lag_selection in [LagSelection::Aic, LagSelection::Fixed(2)] {
      let cfg = AdfConfig {
        trend,
        lag_selection,
        ..AdfConfig::default()
      };
      let (at_level, less_level) = (
        adf_test(y.view(), cfg).map(|r| r.statistic),
        adf_test(w.view(), cfg).unwrap().statistic,
      );
      assert!(
        at_level
          .as_ref()
          .is_ok_and(|s| (s - less_level).abs() <= 1e-12 * less_level.abs()),
        "ADF {trend:?} {lag_selection:?}: {at_level:?} at 1e8 against {less_level}"
      );
    }
  }
}

/// ERS and Leybourne–McCabe test noise of `1e-7` about a level from `1e3` to `1e8` as the same
/// points less the level: their detrending meets the series less its mean.
#[test]
fn ers_and_leybourne_mccabe_keep_the_statistic_at_a_level_from_1e3_to_1e8() {
  let mut noise = vec![0.0; 200];
  SimdNormal::<f64>::new(0.0, 1.0)
    .seeded(&Deterministic::new(3))
    .fill_slice(&mut noise);
  for level in [1e3, 1e4, 1e5, 1e6, 1e7, 1e8] {
    let y = Array1::from_iter(noise.iter().map(|e| level + 1e-7 * e));
    let w = y.mapv(|v| v - level);
    for trend in [Trend::Constant, Trend::ConstantTrend] {
      for lag_selection in [LagSelection::Aic, LagSelection::Fixed(2)] {
        let cfg = ErsConfig {
          trend,
          lag_selection,
          ..ErsConfig::default()
        };
        let (at_level, less_level) = (
          ers_dfgls_test(y.view(), cfg).map(|r| r.statistic),
          ers_dfgls_test(w.view(), cfg).unwrap().statistic,
        );
        assert!(
          at_level
            .as_ref()
            .is_ok_and(|s| (s - less_level).abs() <= 1e-12 * less_level.abs()),
          "ERS {trend:?} {lag_selection:?}: {at_level:?} at {level:e} against {less_level}"
        );
      }
      for ar_lags in [1, 2] {
        let cfg = LeybourneMcCabeConfig {
          trend,
          ar_lags,
          ..LeybourneMcCabeConfig::default()
        };
        let (at_level, less_level) = (
          leybourne_mccabe_test(y.view(), cfg).map(|r| (r.statistic, r.p_value.unwrap())),
          leybourne_mccabe_test(w.view(), cfg).unwrap(),
        );
        assert!(
          at_level.as_ref().is_ok_and(|(s, p)| {
            (s - less_level.statistic).abs() <= 1e-12 * less_level.statistic
              && (p - less_level.p_value.unwrap()).abs() <= 0.005
          }),
          "LM {trend:?} at {ar_lags} lags: {at_level:?} at {level:e} against {less_level:?}"
        );
      }
    }
  }
}
