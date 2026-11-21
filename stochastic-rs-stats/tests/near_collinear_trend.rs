//! Phillips–Perron with a trend, on series their trend dominates: the lagged level lies close to
//! the constant and trend, and the statistic still matches its value in exact rational arithmetic.

use ndarray::Array1;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_distributions::DistributionSampler;
use stochastic_rs_distributions::SimdDistribution;
use stochastic_rs_distributions::normal::SimdNormal;
use stochastic_rs_stats::StatsError;
use stochastic_rs_stats::stationarity::Trend;
use stochastic_rs_stats::stationarity::phillips_perron::PhillipsPerronConfig;
use stochastic_rs_stats::stationarity::phillips_perron::phillips_perron_test;

fn normals() -> Vec<f64> {
  let mut z = vec![0.0; 200];
  SimdNormal::<f64>::new(0.0, 1.0)
    .seeded(&Deterministic::new(2))
    .fill_slice(&mut z);
  z
}

fn z_tau(y: &Array1<f64>) -> Result<f64, StatsError> {
  let cfg = PhillipsPerronConfig {
    trend: Trend::ConstantTrend,
    ..PhillipsPerronConfig::default()
  };
  phillips_perron_test(y.view(), cfg).map(|r| r.statistic)
}

/// `Z_τ` of the line `0..199` plus noise of `a` times its range, from Python `Fraction`s on the
/// same doubles, with each tolerance; `a = 1e-13` falls below the exact-fit rule.
const LINE: [(f64, f64, f64); 6] = [
  (1e-6, -12.846084936733384, 1e-10),
  (1e-8, -12.846084942733107, 1e-8),
  (1e-9, -12.846084941679544, 1e-8),
  (1e-10, -12.846085257399444, 1e-6),
  (1e-11, -12.846087852221881, 1e-5),
  (1e-12, -12.846028676460177, 1e-4),
];

/// The same for `1000 + 10 t` plus noise of `sigma`.
const LEVEL: [(f64, f64, f64); 5] = [
  (1e-2, -12.846084936730044, 1e-10),
  (1e-4, -12.846084937329763, 1e-9),
  (1e-6, -12.846084846651356, 1e-6),
  (1e-7, -12.846084189684447, 1e-5),
  (1e-8, -12.84609094009974, 1e-5),
];

#[test]
fn a_trend_dominated_series_matches_its_exact_statistic() {
  let z = normals();
  for (a, exact, tolerance) in LINE {
    let y = Array1::from_iter((0..200).map(|t| t as f64 + a * 200.0 * z[t]));
    let got = z_tau(&y);
    assert!(
      got
        .as_ref()
        .is_ok_and(|g| (g - exact).abs() <= tolerance * exact.abs()),
      "line plus {a:e} of its range: {got:?} against {exact}"
    );
  }
  for (sigma, exact, tolerance) in LEVEL {
    let y = Array1::from_iter((0..200).map(|t| 1000.0 + 10.0 * t as f64 + sigma * z[t]));
    let got = z_tau(&y);
    assert!(
      got
        .as_ref()
        .is_ok_and(|g| (g - exact).abs() <= tolerance * exact.abs()),
      "1000 + 10 t plus {sigma:e}: {got:?} against {exact}"
    );
  }
  let y = Array1::from_iter((0..200).map(|t| t as f64 + 2e-11 * z[t]));
  assert_eq!(
    z_tau(&y),
    Err(StatsError::DegenerateData("zero residual variance"))
  );
}
