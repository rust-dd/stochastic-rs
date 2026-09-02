// docs: processes#wishart-process-stochastic-covariance
//! Backs the Wishart process example on the processes page.

use ndarray::Array2;
use ndarray::array;
use stochastic_rs::simd_rng::Deterministic;
use stochastic_rs::stochastic::diffusion::wishart::Wishart;
use stochastic_rs::traits::ProcessExt;

#[test]
fn wishart_terminal_law_matches_the_closed_forms() {
  // 2 × 2 Wishart with a full drift, four exact steps over one year.
  let process = Wishart::<f64, _>::new(
    2.5,
    array![[-0.5, 0.1], [0.05, -0.3]],
    array![[0.3, 0.1], [0.0, 0.2]],
    array![[1.0, 0.2], [0.2, 0.5]],
    5,
    Some(1.0),
    Deterministic::new(7),
  );
  let paths = process.sample_par(8_000);
  assert_eq!(paths[0].dim(), (5, 2, 2));

  // Terminal mean against E[X_1] = m x₀ mᵀ + α q (affine moment formula).
  let mut mean = Array2::<f64>::zeros((2, 2));
  for p in &paths {
    mean += &p.index_axis(ndarray::Axis(0), 4);
  }
  mean /= paths.len() as f64;
  let want = process.mean(1.0);
  assert!(
    (&mean - &want).iter().all(|e| e.abs() < 0.05),
    "mean {mean:?} vs {want:?}"
  );

  // Laplace transform E[exp(Tr(v X_1))] for a negative definite v, eq. (10).
  let v = array![[-0.4, -0.12], [-0.12, -0.4]];
  let mc = paths
    .iter()
    .map(|p| {
      let x = p.index_axis(ndarray::Axis(0), 4);
      (v.dot(&x)).diag().sum().exp()
    })
    .sum::<f64>()
    / paths.len() as f64;
  let exact = process.laplace_transform(&v, 1.0);
  assert!((mc - exact).abs() < 0.03, "laplace {mc} vs {exact}");
}
