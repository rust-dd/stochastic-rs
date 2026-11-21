//! Leybourne–McCabe's bootstrap paths take the condition bound and not the exact-fit rule, which
//! guards the caller's series: short persistent series whose fitted AR is explosive.

use ndarray::Array1;
use stochastic_rs_stats::StatsError;
use stochastic_rs_stats::stationarity::Trend;
use stochastic_rs_stats::stationarity::leybourne_mccabe::LeybourneMcCabeConfig;
use stochastic_rs_stats::stationarity::leybourne_mccabe::leybourne_mccabe_test;

/// `label | trend ar_lags | exact statistic and exact-path p-value, or refused | series`: walks of
/// 40 to 60 points from numpy's `default_rng([4242, n, seed])` whose AR fit has a root above 1.03.
const CASES: &str = include_str!("fixtures/lm_short_persistent.txt");

const PATHS: usize = 49;

const PATH_REFUSED: StatsError = StatsError::Numerical {
  stage: "bootstrap path",
};

struct Case {
  label: &'static str,
  cfg: LeybourneMcCabeConfig,
  answer: Option<(f64, f64)>,
  y: Array1<f64>,
}

fn cases() -> impl Iterator<Item = Case> {
  CASES.lines().map(|line| {
    let fields = line.split(" | ").collect::<Vec<_>>();
    let (trend, ar_lags) = fields[1].split_once(' ').unwrap();
    let answer = fields[2]
      .split_once(' ')
      .map(|(s, p)| (s.parse().unwrap(), p.parse().unwrap()));
    Case {
      label: fields[0],
      cfg: LeybourneMcCabeConfig {
        trend: if trend == "trend" {
          Trend::ConstantTrend
        } else {
          Trend::Constant
        },
        ar_lags: ar_lags.parse().unwrap(),
        bootstrap_samples: PATHS,
        ..LeybourneMcCabeConfig::default()
      },
      answer,
      y: fields[3].split(' ').map(|v| v.parse().unwrap()).collect(),
    }
  })
}

/// Within one path's weight of the p-value its paths give in exact arithmetic, or refused where a
/// path's design passes the bound, past which its statistic is rounding.
#[test]
fn short_persistent_series_answer_their_exact_path_p_value_or_refuse() {
  let mut failures = Vec::new();
  for case in cases() {
    let got = leybourne_mccabe_test(case.y.view(), case.cfg);
    let held = match (case.answer, &got) {
      (Some((statistic, p_value)), Ok(r)) => {
        (r.statistic - statistic).abs() <= 1e-9 * statistic
          && r
            .p_value
            .is_some_and(|p| (p - p_value).abs() <= 1.0 / PATHS as f64)
      }
      (None, Err(e)) => *e == PATH_REFUSED,
      _ => false,
    };
    if !held {
      failures.push(format!(
        "{} {:?} ar {}: {got:?} against {:?}",
        case.label, case.cfg.trend, case.cfg.ar_lags, case.answer
      ));
    }
  }
  assert!(failures.is_empty(), "{failures:#?}");
}

/// A 40-point random walk whose AR(4) fit has a root of 1.069 has paths at a condition of `1e15`.
#[test]
fn a_short_random_walk_with_an_explosive_ar_four_fit_is_refused() {
  let case = cases().find(|c| c.label.starts_with("walk ")).unwrap();
  assert_eq!(
    leybourne_mccabe_test(case.y.view(), case.cfg)
      .map(|r| r.statistic)
      .unwrap_err(),
    PATH_REFUSED
  );
}
