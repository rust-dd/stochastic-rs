// docs: stats#garch-family-qmle
//! Backs the GARCH QMLE example on the stats page.

use ndarray::Array1;
use stochastic_rs::simd_rng::Deterministic;
use stochastic_rs::stats::garch::GarchSpec;
use stochastic_rs::stats::garch::MeanSpec;
use stochastic_rs::stats::garch::garch_fit;
use stochastic_rs::stochastic::autoregressive::garch::Garch;
use stochastic_rs::traits::ProcessExt;

#[test]
fn garch_qmle_recovers_the_simulated_parameters() {
  // Two thousand daily returns from GARCH(1,1) with ω = 0.05, α = 0.10,
  // β = 0.85 (percent units, unconditional variance 1.0).
  let process = Garch::<f64, _>::new(
    0.05,
    Array1::from(vec![0.10]),
    Array1::from(vec![0.85]),
    2_000,
    Deterministic::new(42),
  );
  let returns = process.sample();

  // The simulator has no drift, so a zero-mean fit is the matching model;
  // `GarchSpec::garch(1, 1)` alone would estimate a constant mean as well.
  let fit = garch_fit(
    returns.view(),
    GarchSpec::garch(1, 1).with_mean(MeanSpec::Zero),
  );
  assert!(fit.converged);
  assert!((fit.alpha[0] - 0.10).abs() < 3.0 * fit.robust_std_errors[1]);
  assert!((fit.beta[0] - 0.85).abs() < 3.0 * fit.robust_std_errors[2]);
  assert!(fit.persistence < 1.0);
  assert_eq!(fit.conditional_variance.len(), returns.len());
}
