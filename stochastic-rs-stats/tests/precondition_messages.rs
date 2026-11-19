//! The goodness-of-fit, filtering and HMM helpers panic in the predicate form, the value last.

use std::panic::AssertUnwindSafe;
use std::panic::catch_unwind;

use ndarray::Array1;
use ndarray::Array2;
use ndarray::ArrayView1;
use ndarray::array;
use stochastic_rs_stats::econometrics::hmm::GaussianHmm;
use stochastic_rs_stats::filtering::ParticleFilter;
use stochastic_rs_stats::filtering::UkfState;
use stochastic_rs_stats::filtering::UkfTuning;
use stochastic_rs_stats::filtering::unscented_kalman_step;
use stochastic_rs_stats::goodness_of_fit::chi_square::bin_observed;
use stochastic_rs_stats::goodness_of_fit::chi_square::chi_square_critical_value;
use stochastic_rs_stats::goodness_of_fit::chi_square::pool_integer_bins;
use stochastic_rs_stats::goodness_of_fit::kolmogorov_smirnov::ks_critical_value;

fn panic_text(f: impl FnOnce()) -> String {
  let payload = catch_unwind(AssertUnwindSafe(f)).expect_err("a panic");
  payload
    .downcast_ref::<String>()
    .cloned()
    .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
    .expect("a text panic payload")
}

fn ukf(
  prior_cov: Array2<f64>,
  process: Array2<f64>,
  measured: Array2<f64>,
  out: usize,
  seen: usize,
) {
  let prior = UkfState {
    mean: Array1::zeros(2),
    covariance: prior_cov,
  };
  let _ = unscented_kalman_step(
    &prior,
    move |x: ArrayView1<f64>| Array1::zeros(out.max(1)) + x.sum() * 0.0,
    move |_: ArrayView1<f64>| Array1::zeros(seen),
    process.view(),
    measured.view(),
    Array1::<f64>::zeros(1).view(),
    UkfTuning::default(),
  );
}

/// A call that must panic, with the message it must panic with.
type Case = (Box<dyn FnOnce()>, &'static str);

#[test]
fn each_precondition_names_its_predicate_and_its_value() {
  let eye = || Array2::<f64>::eye(2);
  let cases: Vec<Case> = vec![
    (
      Box::new(|| {
        chi_square_critical_value(1.5, 3);
      }),
      "alpha must satisfy `0.0 < alpha < 1.0`, got alpha = 1.5",
    ),
    (
      Box::new(|| {
        chi_square_critical_value(0.05, 0);
      }),
      "df must satisfy `df >= 1`, got df = 0",
    ),
    (
      Box::new(|| {
        pool_integer_bins(100, 1, 0, |_| 0.5, 5.0);
      }),
      "k_hi must satisfy `k_hi >= k_lo`, got k_hi = 0, k_lo = 1",
    ),
    (
      Box::new(|| {
        pool_integer_bins(100, 0, 3, |_| 0.5, 0.0);
      }),
      "min_expected must satisfy `min_expected > 0.0`, got min_expected = 0.0",
    ),
    (
      Box::new(|| {
        bin_observed(ArrayView1::from(&[1, 2]), ArrayView1::from(&[0]));
      }),
      "edges must satisfy `edges.len() >= 2`, got edges.len() = 1",
    ),
    (
      Box::new(|| {
        ks_critical_value(0.0, 10);
      }),
      "alpha must satisfy `0.0 < alpha < 1.0`, got alpha = 0.0",
    ),
    (
      Box::new(|| {
        ks_critical_value(0.05, 0);
      }),
      "n must satisfy `n >= 1`, got n = 0",
    ),
    (
      Box::new(move || ukf(Array2::eye(1), eye(), Array2::eye(1), 2, 1)),
      "prior.covariance must satisfy `prior.covariance.dim() == (n, n)`, got prior.covariance.dim() = (1, 1), n = 2",
    ),
    (
      Box::new(move || ukf(eye(), Array2::eye(3), Array2::eye(1), 2, 1)),
      "process_cov must satisfy `process_cov.dim() == (n, n)`, got process_cov.dim() = (3, 3), n = 2",
    ),
    (
      Box::new(move || ukf(eye(), eye(), Array2::eye(1), 3, 1)),
      "transition's output must satisfy `output.len() == n`, got output.len() = 3, n = 2",
    ),
    (
      Box::new(move || ukf(eye(), eye(), Array2::eye(2), 2, 1)),
      "measurement_cov must satisfy `measurement_cov.dim() == (m, m)`, got measurement_cov.dim() = (2, 2), m = 1",
    ),
    (
      Box::new(move || ukf(eye(), eye(), Array2::eye(1), 2, 2)),
      "measurement's output must satisfy `output.len() == m`, got output.len() = 2, m = 1",
    ),
    (
      Box::new(|| {
        let hmm = GaussianHmm::new(
          array![0.5, 0.5],
          array![[0.9, 0.1], [0.1, 0.9]],
          array![0.0, 1.0],
          array![1.0, 1.0],
        );
        hmm.filter_step(array![1.0].view(), 0.3);
      }),
      "prev must satisfy `prev.len() == k`, got prev.len() = 1, k = 2",
    ),
    (
      Box::new(|| {
        let mut pf = ParticleFilter::new(
          4,
          |_| Array1::zeros(1),
          |_, _| Array1::zeros(2),
          |_, _| 0.0,
          7,
        );
        pf.step(array![0.0].view());
      }),
      "transition's output must satisfy `output.len() == d`, got output.len() = 2, d = 1",
    ),
  ];
  let total = cases.len();
  let wrong = cases
    .into_iter()
    .filter_map(|(case, message)| {
      let got = panic_text(case);
      (got != message).then(|| format!("{got:?}, not {message:?}"))
    })
    .collect::<Vec<_>>();
  assert!(
    wrong.is_empty(),
    "{} of {total}:\n{}",
    wrong.len(),
    wrong.join("\n")
  );
}
