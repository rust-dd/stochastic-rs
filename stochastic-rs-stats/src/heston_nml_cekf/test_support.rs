use ndarray::Array1;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_distributions::normal::SimdNormal;

pub(super) fn simulate_heston_prices(
  observations: usize,
  delta: f64,
  seed: u64,
) -> (Array1<f64>, Array1<f64>) {
  let r = 0.01;
  let kappa = 1.4;
  let theta = 0.06;
  let sigma = 0.28;
  let rho = -0.45;
  let mut prices = Array1::<f64>::zeros(observations);
  let mut variances = Array1::<f64>::zeros(observations);
  prices[0] = 100.0;
  variances[0] = theta;
  let normal = SimdNormal::<f64>::new(0.0, 1.0, &Deterministic::new(seed));

  for index in 1..observations {
    let independent_price: f64 = normal.sample_fast();
    let variance_shock: f64 = normal.sample_fast();
    let previous_variance = variances[index - 1].max(1e-8);
    let correlated_price_shock =
      rho * variance_shock + (1.0 - rho * rho).sqrt() * independent_price;
    let next_variance = previous_variance
      + kappa * (theta - previous_variance) * delta
      + sigma * previous_variance.sqrt() * delta.sqrt() * variance_shock;
    variances[index] = next_variance.max(1e-8);
    prices[index] = (prices[index - 1].ln()
      + (r - 0.5 * previous_variance) * delta
      + previous_variance.sqrt() * delta.sqrt() * correlated_price_shock)
      .exp();
  }

  (prices, variances)
}

pub(super) fn assert_close(actual: f64, expected: f64, tolerance: f64) {
  assert!(
    (actual - expected).abs() <= tolerance,
    "expected {expected:.16e}, got {actual:.16e}"
  );
}
