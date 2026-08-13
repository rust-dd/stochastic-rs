// docs: copulas#clayton-lower-tail-dependence
//! Backs the Clayton example on the copulas catalog page.

use stochastic_rs::copulas::bivariate::clayton::Clayton;
use stochastic_rs::copulas::correlation::kendall_tau;
use stochastic_rs::traits::BivariateExt;

#[test]
fn clayton_set_tau_then_sample() {
  let mut cop = Clayton::new();
  cop.set_tau(0.5); // tau => theta via Kendall inversion
  cop.set_theta(cop.compute_theta());
  let uv = cop.sample_with_seed(10_000, 42).unwrap(); // Array2<f64>, shape (10_000, 2)
  assert_eq!(uv.dim(), (10_000, 2));
  assert!(uv.iter().all(|&x| (0.0..=1.0).contains(&x)));

  let corr = kendall_tau(&uv); // pairwise Kendall tau matrix, shape (2, 2)
  let tau_hat = corr[[0, 1]];
  assert!((tau_hat - 0.5).abs() < 0.05, "tau_hat = {tau_hat}");
}
