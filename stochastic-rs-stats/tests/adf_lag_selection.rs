//! ADF's automatic lag choice agrees with statsmodels' `adfuller(autolag=…)`, which fits every
//! candidate lag on the rows of the largest.

use ndarray::ArrayView1;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_distributions::DistributionSampler;
use stochastic_rs_distributions::SimdDistribution;
use stochastic_rs_distributions::normal::SimdNormal;
use stochastic_rs_stats::stationarity::LagSelection;
use stochastic_rs_stats::stationarity::Trend;
use stochastic_rs_stats::stationarity::adf::AdfConfig;
use stochastic_rs_stats::stationarity::adf::adf_test;

const N: usize = 400;

fn normals(seed: u64) -> Vec<f64> {
  let mut e = vec![0.0; N];
  SimdNormal::<f64>::new(0.0, 1.0)
    .seeded(&Deterministic::new(seed))
    .fill_slice(&mut e);
  e
}

/// A walk of steps 0.01 plus noise of 0.02, an AR(1) of 0.5, or a walk of AR(1) 0.4 steps.
fn series(kind: usize, seed: u64) -> Vec<f64> {
  let e = normals(seed);
  let mut level = 0.0;
  match kind {
    0 => {
      let noise = normals(seed + 1000);
      (0..N)
        .map(|t| {
          level += 0.01 * e[t];
          level + 0.02 * noise[t]
        })
        .collect()
    }
    1 => e
      .iter()
      .map(|v| {
        level = 0.5 * level + v;
        level
      })
      .collect(),
    _ => {
      let mut step = 0.0;
      e.iter()
        .map(|v| {
          step = 0.4 * step + v;
          level += step;
          level
        })
        .collect()
    }
  }
}

/// statsmodels 0.15.0's `adfuller(y, maxlag=16, regression, autolag)` lags (rows `n`, `c`, `ct`;
/// columns AIC, BIC, t-stat), each decided by at least 0.02 in the criterion or 0.001 in `|t|`.
const STATSMODELS: [(usize, u64, [[usize; 3]; 3]); 12] = [
  (0, 1, [[4, 2, 15], [4, 2, 15], [4, 2, 15]]),
  (0, 2, [[9, 2, 8], [2, 2, 8], [2, 2, 8]]),
  (0, 3, [[5, 2, 5], [5, 2, 3], [3, 2, 13]]),
  (0, 4, [[5, 2, 5], [5, 2, 5], [5, 2, 5]]),
  (1, 1, [[1, 0, 15], [1, 0, 15], [1, 0, 15]]),
  (1, 2, [[0, 0, 0], [0, 0, 0], [0, 0, 0]]),
  (1, 3, [[0, 0, 0], [0, 0, 0], [0, 0, 0]]),
  (1, 4, [[0, 0, 14], [0, 0, 14], [2, 0, 14]]),
  (2, 1, [[2, 1, 16], [2, 1, 16], [2, 1, 16]]),
  (2, 2, [[2, 1, 1], [2, 1, 1], [2, 1, 2]]),
  (2, 3, [[1, 1, 1], [1, 1, 1], [1, 1, 1]]),
  (2, 4, [[1, 1, 15], [3, 1, 15], [3, 1, 15]]),
];

#[test]
fn the_lag_choice_matches_statsmodels() {
  let trends = [Trend::None, Trend::Constant, Trend::ConstantTrend];
  let selections = [LagSelection::Aic, LagSelection::Bic, LagSelection::TStat];
  let mut differ = Vec::new();
  for (kind, seed, lags) in STATSMODELS {
    let y = series(kind, seed);
    for (trend, row) in trends.into_iter().zip(lags) {
      for (lag_selection, want) in selections.into_iter().zip(row) {
        let cfg = AdfConfig {
          trend,
          lag_selection,
          max_lags: Some(16),
          alpha: 0.05,
        };
        let got = adf_test(ArrayView1::from(&y), cfg).unwrap().used_lags;
        if got != want {
          differ.push(format!(
            "kind {kind}, seed {seed}, {trend:?}, {lag_selection:?}: {got} against {want}"
          ));
        }
      }
    }
  }
  assert!(
    differ.is_empty(),
    "{} of 108 differ:\n{}",
    differ.len(),
    differ.join("\n")
  );
}
