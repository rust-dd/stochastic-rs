// docs: stats#adf-stationarity-test
//! Backs the ADF example on the statistics catalog page. `adf_test` lives
//! under the `openblas`-gated `stationarity` module (OLS regression via
//! `ndarray-linalg`), so the whole file is gated on that feature.

#![cfg(feature = "openblas")]

use ndarray::ArrayView1;
use stochastic_rs::simd_rng::Deterministic;
use stochastic_rs::stats::stationarity::adf::AdfConfig;
use stochastic_rs::stats::stationarity::adf::adf_test;
use stochastic_rs::stochastic::diffusion::ou::Ou;
use stochastic_rs::traits::ProcessExt;

#[test]
fn adf_rejects_unit_root_on_a_mean_reverting_series() {
  // An OU path is mean-reverting by construction, so the null (unit
  // root) should be rejected — a stand-in for "prices or log-returns"
  // that avoids a live data feed.
  let series = Ou::<f64, _>::new(
    2.0,
    0.0,
    0.5,
    500,
    Some(0.0),
    Some(10.0),
    Deterministic::new(7),
  )
  .sample();

  let result = adf_test(
    ArrayView1::from(series.as_slice().unwrap()),
    AdfConfig::default(),
  );
  assert!(
    result.reject_unit_root,
    "expected the unit-root null to be rejected"
  );
}
