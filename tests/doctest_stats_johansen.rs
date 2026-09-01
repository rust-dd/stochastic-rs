// docs: stats#johansen-cointegration-and-vecm
//! Backs the Johansen / VECM example on the stats page.

use ndarray::Array2;
use stochastic_rs::distributions::normal::SimdNormal;
use stochastic_rs::simd_rng::Deterministic;
use stochastic_rs::stats::econometrics::johansen_test;
use stochastic_rs::stats::econometrics::vecm_fit;

#[test]
fn johansen_rank_then_vecm() {
  // Two prices sharing one random walk: y2 ≈ 0.7·y1, so the pair has
  // exactly one cointegrating relation.
  let steps = SimdNormal::<f64>::new(0.0, 1.0, &Deterministic::new(7));
  let noise = SimdNormal::<f64>::new(0.0, 0.1, &Deterministic::new(11));
  let mut dw = vec![0.0_f64; 500];
  let mut eps = vec![0.0_f64; 500];
  steps.fill_slice(&mut dw);
  noise.fill_slice(&mut eps);
  let mut y = Array2::<f64>::zeros((500, 2));
  let mut w = 0.0;
  for t in 0..500 {
    w += dw[t];
    y[[t, 0]] = w;
    y[[t, 1]] = 0.7 * w + eps[t];
  }

  // Rank tests at VAR order 2: both sequential procedures stop at r = 1.
  let test = johansen_test(y.view(), 2);
  assert_eq!(test.rank_trace, 1);
  assert_eq!(test.rank_max_eig, 1);
  assert!(test.max_eig_statistics[0] > test.max_eig_critical_5pct[0]);

  // The VECM at that rank recovers the relation y2 − 0.7·y1 up to scale.
  let fit = vecm_fit(y.view(), 2, test.rank_trace);
  let ratio = fit.beta[[1, 0]] / fit.beta[[0, 0]];
  assert!((ratio + 1.0 / 0.7).abs() < 0.1, "beta ratio {ratio}");
  assert_eq!(fit.gamma.len(), 1);
  assert_eq!(fit.residuals.dim(), (498, 2));
}
