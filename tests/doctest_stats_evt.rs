// docs: stats#extreme-value-theory-hill-peaks-over-threshold-and-block-maxima
//! Backs the extreme-value example on the stats page.

use ndarray::Array1;
use stochastic_rs::distributions::pareto::SimdPareto;
use stochastic_rs::simd_rng::Deterministic;
use stochastic_rs::stats::evt::block_maxima;
use stochastic_rs::stats::evt::gev_fit;
use stochastic_rs::stats::evt::hill_estimator;
use stochastic_rs::stats::evt::pot_fit;

#[test]
fn evt_tail_estimates_agree_on_a_pareto_tail() {
  // Losses with an exact Pareto(α = 3) tail, so the true tail index is
  // ξ = 1/3.
  let dist = SimdPareto::<f64>::new(1.0, 3.0, &Deterministic::new(7));
  let mut losses = vec![0.0; 20_000];
  dist.fill_slice(&mut losses);
  let losses = Array1::from(losses);

  let hill = hill_estimator(losses.view(), 500);
  assert!((hill.xi - 1.0 / 3.0).abs() < 3.0 * hill.std_error);

  // Peaks over a threshold of 3: the GPD shape estimates the same ξ, and
  // the tail quantiles follow.
  let pot = pot_fit(losses.view(), 3.0);
  assert!(pot.gpd.converged);
  assert!((pot.gpd.xi - 1.0 / 3.0).abs() < 3.0 * pot.gpd.std_errors[1]);
  let var99 = pot.quantile(0.99);
  assert!(pot.expected_shortfall(0.99) > var99 && var99 > pot.threshold);

  // Block maxima of 100 draws: GEV with ξ ≈ 1/3 again.
  let maxima = block_maxima(losses.view(), 100);
  let gev = gev_fit(maxima.view());
  assert!(gev.converged);
  assert!((gev.xi - 1.0 / 3.0).abs() < 3.0 * gev.std_errors[2]);
  assert!(gev.return_level(50.0) > gev.mu);
}
