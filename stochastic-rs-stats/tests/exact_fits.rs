//! A regression whose residuals keep no more of the series' variance than rounding is refused by
//! every unit-root and cointegration test that runs one; a statistic of its residuals is noise.

use ndarray::Array1;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_distributions::DistributionSampler;
use stochastic_rs_distributions::SimdDistribution;
use stochastic_rs_distributions::normal::SimdNormal;
use stochastic_rs_stats::StatsError;
use stochastic_rs_stats::econometrics::cointegration::EngleGrangerConfig;
use stochastic_rs_stats::econometrics::cointegration::engle_granger_test;
use stochastic_rs_stats::stationarity::LagSelection;
use stochastic_rs_stats::stationarity::Trend;
use stochastic_rs_stats::stationarity::adf::AdfConfig;
use stochastic_rs_stats::stationarity::adf::adf_test;
use stochastic_rs_stats::stationarity::ers_dfgls::ErsConfig;
use stochastic_rs_stats::stationarity::ers_dfgls::ers_dfgls_test;
use stochastic_rs_stats::stationarity::leybourne_mccabe::LeybourneMcCabeConfig;
use stochastic_rs_stats::stationarity::leybourne_mccabe::leybourne_mccabe_test;

const SCALES: [f64; 3] = [1e-6, 1.0, 1e6];

const EXACT: StatsError = StatsError::DegenerateData("zero residual variance");

const RULES: [LagSelection; 3] = [LagSelection::Aic, LagSelection::Bic, LagSelection::TStat];

fn normals(seed: u64, n: usize) -> Vec<f64> {
  let mut z = vec![0.0; n];
  SimdNormal::<f64>::new(0.0, 1.0)
    .seeded(&Deterministic::new(seed))
    .fill_slice(&mut z);
  z
}

/// Lines at levels 0 and `1e5`, `t²`, `1.5^t` and `2^t`, each its own ADF regression to rounding.
fn exact_series() -> [(&'static str, Array1<f64>); 5] {
  [
    ("line 0..99", Array1::from_iter((0..100).map(|t| t as f64))),
    (
      "1e5 + 0.37 t",
      Array1::from_iter((0..100).map(|t| 1e5 + 0.37 * t as f64)),
    ),
    ("t²", Array1::from_iter((0..100).map(|t| (t * t) as f64))),
    ("1.5^t", Array1::from_iter((0..80).map(|t| 1.5f64.powi(t)))),
    ("2^t", Array1::from_iter((0..60).map(|t| 2f64.powi(t)))),
  ]
}

/// Refused as an exact fit or the singular design a collinear lag leaves, whose `t` the t rule
/// cannot read.
fn refused<T>(got: &Result<T, StatsError>, lag_selection: LagSelection) -> bool {
  match got {
    Err(
      StatsError::DegenerateData("zero residual variance") | StatsError::Singular("OLS design"),
    ) => true,
    Err(StatsError::Numerical {
      stage: "lag selection",
    }) => lag_selection == LagSelection::TStat,
    _ => false,
  }
}

#[test]
fn adf_and_df_gls_refuse_an_exact_fit_at_every_scale() {
  for (what, y) in exact_series() {
    for c in SCALES {
      let y = y.mapv(|v| v * c);
      for lag_selection in RULES {
        for trend in [Trend::None, Trend::Constant, Trend::ConstantTrend] {
          let cfg = AdfConfig {
            trend,
            lag_selection,
            max_lags: Some(4),
            ..AdfConfig::default()
          };
          let got = adf_test(y.view(), cfg).map(|r| r.statistic);
          assert!(
            refused(&got, lag_selection),
            "ADF {trend:?} {lag_selection:?}, {what}, c = {c:e}: {got:?}"
          );
        }
        for trend in [Trend::Constant, Trend::ConstantTrend] {
          let cfg = ErsConfig {
            trend,
            lag_selection,
            max_lags: Some(4),
            ..ErsConfig::default()
          };
          let got = ers_dfgls_test(y.view(), cfg).map(|r| r.statistic);
          assert!(
            refused(&got, lag_selection),
            "ERS {trend:?} {lag_selection:?}, {what}, c = {c:e}: {got:?}"
          );
        }
      }
    }
  }
}

/// `y = 2x + 1` on a random walk, exactly and with noise of `1e-13`, below the rule's `1e-12`.
#[test]
fn engle_granger_refuses_an_exact_cointegrating_regression_at_every_scale() {
  let mut walk = 0.0;
  let x = Array1::from_iter(normals(3, 200).into_iter().map(|e| {
    walk += e;
    walk
  }));
  let noise = normals(4, 200);
  for (what, size) in [("y = 2x + 1", 0.0), ("y = 2x + 1 + 1e-13 z", 1e-13)] {
    let y = Array1::from_iter(x.iter().zip(&noise).map(|(v, e)| 2.0 * v + 1.0 + size * e));
    for c in SCALES {
      let got = engle_granger_test(
        y.mapv(|v| v * c).view(),
        x.mapv(|v| v * c).view(),
        EngleGrangerConfig::default(),
      )
      .map(|r| r.statistic);
      assert_eq!(got, Err(EXACT), "{what}, c = {c:e}");
    }
  }
}

/// An alternating series is an exact AR(1) about its mean, and with noise far below `1e-12` too.
#[test]
fn leybourne_mccabe_refuses_an_exact_ar_fit_at_every_scale() {
  let alternating = |size: f64, noise: f64, seed: u64| {
    let z = normals(seed, 80);
    Array1::from_iter((0..80).map(|t| size * if t % 2 == 0 { 1.0 } else { -1.0 } + noise * z[t]))
  };
  let cfg = LeybourneMcCabeConfig {
    bootstrap_samples: 19,
    ..LeybourneMcCabeConfig::default()
  };
  for (what, y) in [
    ("alternating", alternating(1.0, 0.0, 5)),
    ("alternating + 1e-15 z", alternating(1.0, 1e-15, 5)),
    ("0.1 × alternating + 1e-17 z", alternating(0.1, 1e-17, 6)),
  ] {
    for c in SCALES {
      let got = leybourne_mccabe_test(y.mapv(|v| v * c).view(), cfg).map(|r| r.statistic);
      assert_eq!(got, Err(EXACT), "{what}, c = {c:e}");
    }
  }
}

/// Detrended, the alternating series is not an AR(1): its residuals carry the fitted line's slope.
#[test]
fn leybourne_mccabe_with_a_trend_keeps_the_alternating_series_statistic() {
  let y = Array1::from_iter((0..80).map(|t| if t % 2 == 0 { 1.0 } else { -1.0 }));
  let cfg = LeybourneMcCabeConfig {
    trend: Trend::ConstantTrend,
    bootstrap_samples: 19,
    ..LeybourneMcCabeConfig::default()
  };
  let unit = leybourne_mccabe_test(y.view(), cfg).unwrap().statistic;
  for c in SCALES {
    let got = leybourne_mccabe_test(y.mapv(|v| v * c).view(), cfg)
      .unwrap()
      .statistic;
    assert!(
      (got - unit).abs() <= 1e-12 * unit,
      "c = {c:e}: {got} against {unit}"
    );
  }
}
