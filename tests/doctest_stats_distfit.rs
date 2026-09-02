// docs: stats#fitting-the-skewed-return-distributions
//! Backs the distribution-fitting example on the stats page.

use ndarray::Array1;
use stochastic_rs::distributions::skew_t::SimdSkewT;
use stochastic_rs::simd_rng::Deterministic;
use stochastic_rs::stats::distfit::johnson_su_fit;
use stochastic_rs::stats::distfit::skew_t_fit;

#[test]
fn skew_t_fit_recovers_the_simulated_shape() {
  // Returns from Hansen's skew-t with η = 6, λ = −0.3, scaled to 1.5%
  // daily volatility around a 0.05% drift.
  let dist = SimdSkewT::<f64>::new(6.0, -0.3, &Deterministic::new(11));
  let mut z = vec![0.0; 4_000];
  dist.fill_slice(&mut z);
  let returns = Array1::from_iter(z.iter().map(|v| 0.0005 + 0.015 * v));

  let fit = skew_t_fit(returns.view());
  assert!(fit.converged);
  assert!((fit.lambda + 0.3).abs() < 3.0 * fit.std_errors[3]);
  assert!((fit.sigma - 0.015).abs() < 3.0 * fit.std_errors[1]);

  // A Johnson SU fit of the same sample ranks below it on AIC.
  let jsu = johnson_su_fit(returns.view());
  assert!(jsu.converged && jsu.aic > fit.aic);
}
