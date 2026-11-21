//! `RBergomiPricer`'s antithetic pairs against independent ones at equal cost.

use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_quant::calibration::rbergomi::RBergomiParams;
use stochastic_rs_quant::calibration::rbergomi::RBergomiXi0;
use stochastic_rs_quant::calibration::rbergomi::simulate_rbergomi_terminal_samples;
use stochastic_rs_quant::pricing::rbergomi::RBergomiPricer;
use stochastic_rs_quant::traits::ModelPricer;

fn rbergomi_params() -> RBergomiParams {
  RBergomiParams {
    hurst: 0.12,
    rho: -0.72,
    eta: 1.6,
    xi0: RBergomiXi0::Constant(0.04),
  }
}

/// Error bar of the mean of `n` pair means, the pairs `(i, i + n)` of `2n` payoffs.
fn pair_mean_and_error(payoffs: &[f64]) -> (f64, f64) {
  let n = payoffs.len() / 2;
  let pairs = (0..n).map(|i| 0.5 * (payoffs[i] + payoffs[i + n]));
  let mean = pairs.clone().sum::<f64>() / n as f64;
  let var = pairs.map(|y| (y - mean).powi(2)).sum::<f64>() / (n as f64 - 1.0);
  (mean, (var / n as f64).sqrt())
}

/// Antithetic pairs against two independent runs at the same `2n` paths: the error bar falls to
/// 0.66, 0.67, 0.78 and 0.91 of the independent pairs', and the two estimates agree within bars.
#[test]
fn rbergomi_antithetic_pairs_reduce_the_error_bar_at_equal_cost() {
  let (s, r, q, tau, n) = (100.0, 0.02, 0.0, 0.5, 4_000);
  let seed = Deterministic::new(42);
  let pricer = RBergomiPricer::new(rbergomi_params())
    .with_paths(n)
    .with_steps_per_year(50)
    .with_seed(Deterministic::new(42));
  let independent =
    simulate_rbergomi_terminal_samples(&rbergomi_params(), s, r, q, tau, 2 * n, 50, 12, &seed);
  let discount = (-r * tau).exp();
  for (k, largest_ratio) in [(80.0, 0.75), (90.0, 0.75), (100.0, 0.85), (110.0, 0.97)] {
    let paired = pricer.price_call_estimate(s, k, r, q, tau);
    let payoffs = independent
      .iter()
      .map(|st| discount * (st - k).max(0.0))
      .collect::<Vec<f64>>();
    let (mean, std_err) = pair_mean_and_error(&payoffs);
    let ratio = paired.std_err / std_err;
    assert_eq!(paired.n_samples, n);
    assert!(ratio < largest_ratio, "k = {k}: ratio {ratio}");
    let bar = (paired.std_err.powi(2) + std_err.powi(2)).sqrt();
    assert!(
      (paired.mean - mean).abs() < 4.0 * bar,
      "k = {k}: {} vs {mean}",
      paired.mean
    );
  }
  let plain = pricer.clone().with_antithetic(false);
  let fresh = Deterministic::new(42);
  let first =
    simulate_rbergomi_terminal_samples(&rbergomi_params(), s, r, q, tau, n, 50, 12, &fresh);
  let mean = discount * (first.iter().map(|st| (st - 100.0).max(0.0)).sum::<f64>() / n as f64);
  assert_eq!(plain.price_call(s, 100.0, r, q, tau), mean);
}
