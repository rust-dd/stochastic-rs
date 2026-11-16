use std::f64::consts::PI;

use ndarray::Array1;
use ndarray::Array2;
use ndarray::array;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_distributions::DistributionSampler;
use stochastic_rs_distributions::SimdDistribution;
use stochastic_rs_distributions::normal::SimdNormal;

use crate::econometrics::GaussianHmm;
use crate::econometrics::hmm::SCALED_PASS_MIN_PROBABILITY;

fn log_sum_exp(values: impl Iterator<Item = f64> + Clone) -> f64 {
  let max = values.clone().fold(f64::NEG_INFINITY, f64::max);
  if max == f64::NEG_INFINITY {
    return max;
  }
  max + values.map(|v| (v - max).exp()).sum::<f64>().ln()
}

fn ln_density(hmm: &GaussianHmm, x: f64, s: usize) -> f64 {
  let z = (x - hmm.means[s]) / hmm.stds[s];
  -0.5 * z * z - hmm.stds[s].ln() - 0.5 * (2.0 * PI).ln()
}

/// The unscaled `ln α_t`, `ln β_t` and `ln p(y)` of Rabiner's recursions, every sum by log-sum-exp.
fn exact_passes(hmm: &GaussianHmm, y: &[f64]) -> (Array2<f64>, Array2<f64>, f64) {
  let (n, k) = (y.len(), hmm.n_states());
  let ln_b = |t: usize, s: usize| ln_density(hmm, y[t], s);
  let ln_a = hmm.transitions.mapv(f64::ln);
  let mut alpha = Array2::<f64>::zeros((n, k));
  let mut beta = Array2::<f64>::zeros((n, k));
  for s in 0..k {
    alpha[[0, s]] = hmm.initial[s].ln() + ln_b(0, s);
  }
  for t in 1..n {
    for j in 0..k {
      alpha[[t, j]] = log_sum_exp((0..k).map(|i| alpha[[t - 1, i]] + ln_a[[i, j]])) + ln_b(t, j);
    }
  }
  for t in (0..n - 1).rev() {
    for i in 0..k {
      beta[[t, i]] = log_sum_exp((0..k).map(|j| ln_a[[i, j]] + ln_b(t + 1, j) + beta[[t + 1, j]]));
    }
  }
  let log_likelihood = log_sum_exp((0..k).map(|s| alpha[[n - 1, s]]));
  (alpha, beta, log_likelihood)
}

fn assert_close(got: f64, want: f64, relative: f64, what: &str) {
  assert!(
    (got - want).abs() <= relative * want.abs(),
    "{what}: {got} against {want}"
  );
}

/// State 0 absorbs (or leaves at `leave`), and the point at `outlier` sits among 61 near 10.
fn absorbing_with_outlier(outlier: f64, leave: f64) -> (GaussianHmm, Array1<f64>) {
  let hmm = GaussianHmm::new(
    array![0.5, 0.5],
    array![[1.0 - leave, leave], [1e-10, 1.0 - 1e-10]],
    array![0.0, 10.0],
    array![1.0, 1.0],
  );
  let mut y = Array1::from_iter((0..61).map(|t| 10.0 + 0.1 * (t as f64 * 1.3).sin()));
  y[30] = outlier;
  (hmm, y)
}

/// State 0 absorbs (or leaves at `leave`): 600 points from its law, then 3000 from state 1's.
fn absorbing_on_ordinary_data(leave: f64) -> (GaussianHmm, Array1<f64>) {
  let hmm = GaussianHmm::new(
    array![0.5, 0.5],
    array![[1.0 - leave, leave], [0.001, 0.999]],
    array![-0.5, 0.5],
    array![2.0, 1.0],
  );
  let mut z = vec![0.0; 3600];
  SimdNormal::<f64>::new(0.0, 1.0)
    .seeded(&Deterministic::new(5))
    .fill_slice(&mut z);
  let y = Array1::from_iter((0..3600).map(|t| {
    if t < 600 {
      -0.5 + 2.0 * z[t]
    } else {
      0.5 + z[t]
    }
  }));
  (hmm, y)
}

/// The outlier at −70 leaves state 1 a filtered weight near `e^{−750}`, which the points after it
/// need; `−3256.8947196354566` is the review's numpy pass on this series.
#[test]
fn an_absorbing_state_keeps_the_exact_likelihood_past_an_outlier() {
  let (hmm, y) = absorbing_with_outlier(-70.0, 0.0);
  let exact = exact_passes(&hmm, y.as_slice().unwrap()).2;
  let ll = hmm.log_likelihood(y.view());
  assert_close(ll, exact, 1e-12, "log_likelihood");
  assert_close(
    ll,
    -3256.8947196354566,
    1e-12,
    "log_likelihood against numpy",
  );
  let fit = hmm.clone().baum_welch(y.view(), 1, 0.0).unwrap();
  assert_close(fit.log_likelihood, exact, 1e-12, "baum_welch");
}

/// No outlier: state 1's exact filtered log-probability falls to −776.7 by t = 599 and ends near 0;
/// `−6255.35394742743` is a numpy exact pass on the crate's seed-5 series.
#[test]
fn an_absorbing_state_keeps_the_exact_likelihood_on_ordinary_data() {
  let (hmm, y) = absorbing_on_ordinary_data(0.0);
  let (alpha, _, exact) = exact_passes(&hmm, y.as_slice().unwrap());
  let ll = hmm.log_likelihood(y.view());
  assert_close(ll, exact, 1e-11, "log_likelihood");
  assert_close(ll, -6255.35394742743, 1e-11, "log_likelihood against numpy");
  let fit = hmm.clone().baum_welch(y.view(), 1, 0.0).unwrap();
  assert_close(fit.log_likelihood, exact, 1e-11, "baum_welch");
  let last = alpha.row(3599);
  let filtered = (last[1] - log_sum_exp(last.iter().copied())).exp();
  let got = hmm.filtered_state_probs(y.view())[[3599, 1]];
  assert_close(
    got,
    filtered,
    1e-11,
    "the last filtered probability of state 1",
  );
}

/// The model whose scaled backward pass overflowed at an outlier of −69: one Baum–Welch step
/// against Rabiner's (27) and (37) on the unscaled log passes, each normalised per step.
#[test]
fn the_backward_pass_past_the_double_range_is_exact() {
  let (mut hmm, y) = absorbing_with_outlier(-69.0, 0.0);
  let (alpha, beta, exact) = exact_passes(&hmm, y.as_slice().unwrap());
  let (n, k) = alpha.dim();
  let ln_a = hmm.transitions.mapv(f64::ln);
  let ln_b = |t: usize, s: usize| ln_density(&hmm, y[t], s);
  let gamma = |t: usize| {
    let z = log_sum_exp((0..k).map(|s| alpha[[t, s]] + beta[[t, s]]));
    Array1::from_iter((0..k).map(|s| (alpha[[t, s]] + beta[[t, s]] - z).exp()))
  };
  let mut xi = Array2::<f64>::zeros((k, k));
  for t in 0..n - 1 {
    let term =
      |i: usize, j: usize| alpha[[t, i]] + ln_a[[i, j]] + ln_b(t + 1, j) + beta[[t + 1, j]];
    let z = log_sum_exp((0..k * k).map(|ij| term(ij / k, ij % k)));
    xi += &Array2::from_shape_fn((k, k), |(i, j)| (term(i, j) - z).exp());
  }
  let rows = &xi / &xi.sum_axis(ndarray::Axis(1)).insert_axis(ndarray::Axis(1));
  let weights = Array1::from_iter((0..n).map(|t| gamma(t)[1]));
  let mean = weights.dot(&y) / weights.sum();
  let initial = gamma(0);

  let fit = hmm.baum_welch(y.view(), 1, 0.0).unwrap();

  assert_close(fit.log_likelihood, exact, 1e-12, "log-likelihood");
  for s in 0..k {
    assert_close(hmm.initial[s], initial[s], 1e-12, "initial");
    assert_close(
      hmm.transitions[[1, s]],
      rows[[1, s]],
      1e-9,
      "transitions out of state 1",
    );
  }
  assert_eq!(hmm.transitions.row(0), array![1.0, 0.0]);
  assert_close(hmm.means[1], mean, 1e-12, "the mean of state 1");
}

/// The first point favours state 1 by 800 nats, so state 0 starts at `e^{−109}` of state 1, and
/// only state 0 explains the rest: a tiny or zero `π_1` alone keeps the scaled pass.
#[test]
fn a_tiny_or_zero_initial_probability_keeps_the_scaled_pass_exact() {
  let y = Array1::from_iter((0..21).map(|t| if t == 0 { 40.0 } else { 0.1 * (t as f64).sin() }));
  for initial in [array![1.0 - 1e-300, 1e-300], array![1.0, 0.0]] {
    let hmm = GaussianHmm::new(
      initial,
      array![[0.999, 0.001], [1e-100, 1.0 - 1e-100]],
      array![0.0, 40.0],
      array![1.0, 1.0],
    );
    let exact = exact_passes(&hmm, y.as_slice().unwrap()).2;
    let what = format!("initial {:?}", hmm.initial.to_vec());
    assert_close(hmm.forward(y.view()).log_likelihood, exact, 1e-12, &what);
    assert!(!hmm.in_log_space(), "{what}");
    assert_close(hmm.log_likelihood(y.view()), exact, 1e-12, &what);
    let fit = hmm.clone().baum_welch(y.view(), 1, 0.0).unwrap();
    assert_close(fit.log_likelihood, exact, 1e-12, &what);
  }
}

/// With the absorbing transition at `2τ` both models keep the scaled pass, which then matches the
/// exact passes, and the passes on logarithms, within 1e-11 relative.
#[test]
fn a_transition_just_above_the_threshold_keeps_the_scaled_pass_exact() {
  let leave = 2.0 * SCALED_PASS_MIN_PROBABILITY;
  for (hmm, y) in [
    absorbing_with_outlier(-70.0, leave),
    absorbing_on_ordinary_data(leave),
  ] {
    assert!(!hmm.in_log_space());
    let exact = exact_passes(&hmm, y.as_slice().unwrap()).2;
    let scaled = hmm.forward(y.view());
    let logs = hmm.log_forward(y.view());
    assert_close(hmm.log_likelihood(y.view()), exact, 1e-11, "log_likelihood");
    assert_close(
      logs.log_likelihood,
      exact,
      1e-11,
      "the passes on logarithms",
    );
    for (p, ln_p) in scaled.alpha.iter().zip(&logs.log_alpha) {
      assert!(
        (p - ln_p.exp()).abs() <= 1e-11,
        "filtered: {p} against {}",
        ln_p.exp()
      );
    }
  }
}
