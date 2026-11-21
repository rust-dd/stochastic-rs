//! Engle–Granger against statsmodels' `coint`, whose ADF on the residuals has no deterministic
//! terms; arch's `engle_granger` gives the same statistics.

use ndarray::Array1;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_distributions::DistributionSampler;
use stochastic_rs_distributions::SimdDistribution;
use stochastic_rs_distributions::normal::SimdNormal;
use stochastic_rs_stats::econometrics::cointegration::EngleGrangerConfig;
use stochastic_rs_stats::econometrics::cointegration::engle_granger_test;

fn normals(seed: u64) -> Vec<f64> {
  let mut z = vec![0.0; 250];
  SimdNormal::<f64>::new(0.0, 1.0)
    .seeded(&Deterministic::new(seed))
    .fill_slice(&mut z);
  z
}

/// A random walk `x`, and `y = 0.5 + 2x` plus an AR(2) (seeds 1–3) or an independent walk of
/// AR(1) steps (seeds 4–6).
fn pair(seed: u64) -> (Array1<f64>, Array1<f64>) {
  let (e, u) = (normals(seed), normals(seed + 100));
  let mut walk = 0.0;
  let x = Array1::from_iter(e.iter().map(|v| {
    walk += v;
    walk
  }));
  let (mut a, mut b) = (0.0, 0.0);
  let y = Array1::from_iter((0..250).map(|t| {
    if seed <= 3 {
      let c = 1.1 * a - 0.3 * b + u[t];
      (b, a) = (a, c);
      0.5 + 2.0 * x[t] + c
    } else {
      b = 0.5 * b + u[t];
      a += b;
      a
    }
  }));
  (y, x)
}

/// statsmodels 0.15.0's `coint(y, x, trend="c", maxlag=15, autolag="aic")`, each lag chosen by at
/// least 0.55 in the AIC.
const COINT: [f64; 6] = [
  -6.332115142836262,
  -5.72759751134103,
  -5.474540645124467,
  -2.824095831245911,
  -1.3237756367744635,
  -2.544934213046048,
];

#[test]
fn the_statistic_matches_statsmodels_coint() {
  for (seed, want) in (1..=6).zip(COINT) {
    let (y, x) = pair(seed);
    let got = engle_granger_test(y.view(), x.view(), EngleGrangerConfig::default())
      .unwrap()
      .statistic;
    assert!(
      (got - want).abs() <= 1e-9 * want.abs(),
      "seed {seed}: {got} against coint's {want}"
    );
  }
}
