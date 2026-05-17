//! Integration tests for the `stats::econometrics` module.

use ndarray::Array1;
#[cfg(feature = "openblas")]
use ndarray::Array2;
use stochastic_rs::distributions::normal::SimdNormal;
use stochastic_rs::simd_rng::Deterministic;
#[cfg(feature = "openblas")]
use stochastic_rs::stats::econometrics::GaussianHmm;
use stochastic_rs::stats::econometrics::cusum;
#[cfg(feature = "openblas")]
use stochastic_rs::stats::econometrics::engle_granger_test;
#[cfg(feature = "openblas")]
use stochastic_rs::stats::econometrics::granger_causality;
#[cfg(feature = "openblas")]
use stochastic_rs::stats::econometrics::johansen_test;
use stochastic_rs::stats::econometrics::pelt;

fn random_walk(seed: u64, n: usize, sigma: f64) -> Array1<f64> {
  let dist = SimdNormal::<f64>::new(0.0, sigma, &Deterministic::new(seed));
  let mut steps = vec![0.0_f64; n];
  dist.fill_slice_fast(&mut steps);
  let mut out = Array1::<f64>::zeros(n);
  for i in 1..n {
    out[i] = out[i - 1] + steps[i];
  }
  out
}

#[test]
fn cusum_finite_outputs_for_arbitrary_series() {
  let s = random_walk(101, 1_000, 1.0);
  let res = cusum(s.view(), 0.5, 6.0);
  assert!(res.upper.iter().all(|v| v.is_finite()));
  assert!(res.lower.iter().all(|v| v.is_finite()));
}

#[test]
fn pelt_segment_count_increases_with_lower_penalty() {
  let mut s = Array1::<f64>::zeros(300);
  for i in 0..300 {
    s[i] = match i {
      0..=99 => 0.0,
      100..=199 => 5.0,
      _ => -3.0,
    };
  }
  let cheap = pelt(s.view(), 1.0, 5);
  let expensive = pelt(s.view(), 1_000.0, 5);
  assert!(cheap.changepoints.len() >= expensive.changepoints.len());
}

#[cfg(feature = "openblas")]
#[test]
fn engle_granger_full_pipeline() {
  let x = random_walk(31, 400, 1.0);
  let dist = SimdNormal::<f64>::new(0.0, 0.05, &Deterministic::new(33));
  let mut eps = vec![0.0_f64; 400];
  dist.fill_slice_fast(&mut eps);
  let y: Array1<f64> = (0..400)
    .map(|i| 1.0 + 1.5 * x[i] + eps[i])
    .collect::<Vec<_>>()
    .into();
  let res = engle_granger_test(y.view(), x.view());
  assert!(res.reject_no_cointegration);
  assert!((res.beta - 1.5).abs() < 0.1);
}

#[cfg(feature = "openblas")]
#[test]
fn johansen_eigenvalues_decreasing_in_magnitude() {
  let mut s = Array2::<f64>::zeros((400, 4));
  let r1 = random_walk(41, 400, 1.0);
  let r2 = random_walk(43, 400, 1.0);
  let r3 = random_walk(47, 400, 1.0);
  let r4 = random_walk(53, 400, 1.0);
  for i in 0..400 {
    s[[i, 0]] = r1[i];
    s[[i, 1]] = r2[i];
    s[[i, 2]] = r3[i];
    s[[i, 3]] = r4[i];
  }
  let res = johansen_test(s.view(), 1);
  for w in res.eigenvalues.windows(2).into_iter() {
    assert!(w[0] >= w[1] - 1e-9);
  }
}

#[cfg(feature = "openblas")]
#[test]
fn granger_pvalue_in_unit_interval() {
  let dist = SimdNormal::<f64>::new(0.0, 1.0, &Deterministic::new(71));
  let mut buf_a = vec![0.0_f64; 300];
  let mut buf_b = vec![0.0_f64; 300];
  dist.fill_slice_fast(&mut buf_a);
  dist.fill_slice_fast(&mut buf_b);
  let a = Array1::from(buf_a);
  let b = Array1::from(buf_b);
  let res = granger_causality(a.view(), b.view(), 4, 0.05);
  assert!((0.0..=1.0).contains(&res.p_value));
}

#[cfg(feature = "openblas")]
#[test]
fn hmm_log_likelihood_finite_after_fit() {
  let dist0 = SimdNormal::<f64>::new(-2.0, 0.5, &Deterministic::new(1));
  let dist1 = SimdNormal::<f64>::new(2.0, 0.5, &Deterministic::new(2));
  let mut a = vec![0.0_f64; 200];
  let mut b = vec![0.0_f64; 200];
  dist0.fill_slice_fast(&mut a);
  dist1.fill_slice_fast(&mut b);
  let obs = Array1::from(a.into_iter().chain(b).collect::<Vec<_>>());
  let mut hmm = GaussianHmm::new(
    Array1::from(vec![0.5, 0.5]),
    Array2::from_shape_vec((2, 2), vec![0.9, 0.1, 0.1, 0.9]).unwrap(),
    Array1::from(vec![-1.0, 1.0]),
    Array1::from(vec![1.0, 1.0]),
  );
  let fit = hmm.baum_welch(obs.view(), 50, 1e-6);
  assert!(fit.log_likelihood.is_finite());
  let path = hmm.viterbi(obs.view());
  assert_eq!(path.len(), 400);
}
