// docs: processes#correlated-multi-asset-gbm
//! Backs the correlated multi-asset GBM example on the processes page.

use ndarray::array;
use stochastic_rs::simd_rng::Deterministic;
use stochastic_rs::stochastic::diffusion::multi_gbm::MultiGbm;
use stochastic_rs::traits::ProcessExt;

#[test]
fn multi_gbm_terminal_correlation_follows_rho() {
  // Two assets over one year on 252 steps with correlation −0.4.
  let model = MultiGbm::<f64, _>::new(
    array![0.05, 0.02],
    array![0.2, 0.3],
    array![[1.0, -0.4], [-0.4, 1.0]],
    253,
    array![100.0, 50.0],
    Some(1.0),
    Deterministic::new(7),
  );
  let paths = model.sample_par(8_000);
  assert_eq!(paths[0].dim(), (2, 253));

  // Correlation of the terminal log-returns tracks ρ.
  let logs: Vec<(f64, f64)> = paths
    .iter()
    .map(|p| ((p[(0, 252)] / 100.0).ln(), (p[(1, 252)] / 50.0).ln()))
    .collect();
  let n = logs.len() as f64;
  let (m0, m1) = (
    logs.iter().map(|l| l.0).sum::<f64>() / n,
    logs.iter().map(|l| l.1).sum::<f64>() / n,
  );
  let v0 = logs.iter().map(|l| (l.0 - m0).powi(2)).sum::<f64>() / n;
  let v1 = logs.iter().map(|l| (l.1 - m1).powi(2)).sum::<f64>() / n;
  let cov = logs.iter().map(|l| (l.0 - m0) * (l.1 - m1)).sum::<f64>() / n;
  let corr = cov / (v0 * v1).sqrt();
  assert!((corr + 0.4).abs() < 0.05, "terminal correlation {corr}");
}
