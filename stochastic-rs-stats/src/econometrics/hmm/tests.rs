use ndarray::array;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_distributions::DistributionSampler;
use stochastic_rs_distributions::SimdDistribution;
use stochastic_rs_distributions::normal::SimdNormal;

use super::*;

fn gauss_pdf(x: f64, mean: f64, std: f64) -> f64 {
  let z = (x - mean) / std;
  (-0.5 * z * z).exp() / (std * (2.0 * std::f64::consts::PI).sqrt())
}

fn panic_message(f: impl FnOnce()) -> String {
  let payload = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).expect_err("no panic");
  payload
    .downcast_ref::<String>()
    .cloned()
    .unwrap_or_default()
}

#[test]
fn viterbi_recovers_two_state_path() {
  let init = array![0.5, 0.5];
  let trans = array![[0.95, 0.05], [0.05, 0.95]];
  let means = array![-1.0, 1.0];
  let stds = array![0.3, 0.3];
  let hmm = GaussianHmm::new(init, trans, means, stds);
  let obs = array![
    -1.0_f64, -0.9, -1.1, -1.05, -0.95, -1.0, 1.0, 1.1, 0.9, 1.05, 0.95, 1.0
  ];
  let path = hmm.viterbi(obs.view()).unwrap();
  assert!(path.iter().take(6).all(|&s| s == 0));
  assert!(path.iter().skip(6).all(|&s| s == 1));
}

#[test]
fn filtered_probs_are_normalised_and_track_the_regime() {
  let init = array![0.5, 0.5];
  let trans = array![[0.95, 0.05], [0.05, 0.95]];
  let means = array![-1.0, 1.0];
  let stds = array![0.3, 0.3];
  let hmm = GaussianHmm::new(init, trans, means, stds);
  let obs = array![
    -1.0_f64, -0.9, -1.1, -1.05, -0.95, -1.0, 1.0, 1.1, 0.9, 1.05, 0.95, 1.0
  ];
  let probs = hmm.filtered_state_probs(obs.view());

  assert_eq!(probs.dim(), (obs.len(), 2));
  for row in probs.rows() {
    assert!((row.sum() - 1.0).abs() < 1e-9);
    assert!(row.iter().all(|&p| (0.0..=1.0).contains(&p)));
  }
  assert!(probs[[5, 0]] > 0.9, "low-mean regime should dominate early");
  assert!(
    probs[[11, 1]] > 0.9,
    "high-mean regime should dominate late"
  );
}

#[test]
fn filtered_probs_depend_only_on_the_past() {
  // The defining property for online use: truncating the series after t must
  // not change the filtered distribution at t. Viterbi does not have this.
  let init = array![0.5, 0.5];
  let trans = array![[0.9, 0.1], [0.2, 0.8]];
  let means = array![-1.0, 1.0];
  let stds = array![0.5, 0.4];
  let hmm = GaussianHmm::new(init, trans, means, stds);
  let obs = array![
    -1.0_f64, -0.6, 0.9, 1.2, -0.2, -1.4, 1.1, 0.7, -0.9, 1.3, 0.4, -1.1
  ];
  let full = hmm.filtered_state_probs(obs.view());

  for t in 0..obs.len() {
    let truncated = hmm.filtered_state_probs(obs.slice(ndarray::s![..=t]));
    for k in 0..hmm.n_states() {
      assert!(
        (full[[t, k]] - truncated[[t, k]]).abs() < 1e-12,
        "filtered prob at t={t}, state={k} changed when the future was removed"
      );
    }
  }
}

#[test]
fn filter_step_matches_the_batch_forward_pass() {
  let init = array![0.5, 0.5];
  let trans = array![[0.9, 0.1], [0.2, 0.8]];
  let means = array![-1.0, 1.0];
  let stds = array![0.5, 0.4];
  let hmm = GaussianHmm::new(init, trans, means, stds);
  let obs = array![-1.0_f64, -0.6, 0.9, 1.2, -0.2, -1.4, 1.1, 0.7];
  let batch = hmm.filtered_state_probs(obs.view());

  // The first step consumes the initial distribution unchanged by any
  // transition, so it is seeded from `initial` and then normalised.
  let mut state = {
    let mut first = Array1::<f64>::zeros(hmm.n_states());
    for k in 0..hmm.n_states() {
      first[k] = hmm.initial[k] * gauss_pdf(obs[0], hmm.means[k], hmm.stds[k]);
    }
    let z: f64 = first.sum();
    first / z
  };
  for k in 0..hmm.n_states() {
    assert!((state[k] - batch[[0, k]]).abs() < 1e-12);
  }

  for (t, &observation) in obs.iter().enumerate().skip(1) {
    state = hmm.filter_step(state.view(), observation);
    for k in 0..hmm.n_states() {
      assert!(
        (state[k] - batch[[t, k]]).abs() < 1e-12,
        "streaming and batch filters diverged at t={t}, state={k}"
      );
    }
  }
}

/// Both densities underflow at 1e9, but in log space state 1 explains it better by 2e11 nats.
#[test]
fn filter_step_weighs_an_implausible_observation_by_its_densities() {
  let init = array![0.5, 0.5];
  let trans = array![[0.9, 0.1], [0.2, 0.8]];
  let means = array![-1.0, 1.0];
  let stds = array![0.1, 0.1];
  let hmm = GaussianHmm::new(init, trans, means, stds);

  let next = hmm.filter_step(hmm.initial.view(), 1e9);

  assert_eq!(next, array![0.0, 1.0]);
}

/// At 20 both densities underflow; the log-space pass keeps the likelihood summed over 32 paths.
#[test]
fn an_observation_far_from_every_state_keeps_the_exact_likelihood() {
  let hmm = GaussianHmm::new(
    array![0.5, 0.5],
    array![[0.95, 0.05], [0.05, 0.95]],
    array![-1.0, 1.0],
    array![0.3, 0.3],
  );
  let y = array![-1.0, -0.9, 20.0, 1.0, 0.9];
  let ln_density = |x: f64, s: usize| {
    let z = (x - hmm.means[s]) / hmm.stds[s];
    -0.5 * z * z - hmm.stds[s].ln() - 0.5 * (2.0 * std::f64::consts::PI).ln()
  };
  let paths = (0..1usize << y.len())
    .map(|code| {
      let state = |t: usize| (code >> t) & 1;
      (1..y.len()).fold(
        hmm.initial[state(0)].ln() + ln_density(y[0], state(0)),
        |lp, t| lp + hmm.transitions[[state(t - 1), state(t)]].ln() + ln_density(y[t], state(t)),
      )
    })
    .collect::<Vec<_>>();
  let max = paths.iter().copied().fold(f64::NEG_INFINITY, f64::max);
  let exact = max + paths.iter().map(|lp| (lp - max).exp()).sum::<f64>().ln();
  let ll = hmm.log_likelihood(y.view());
  assert!(
    (ll - exact).abs() <= 1e-9 * exact.abs(),
    "{ll} against {exact}"
  );
  for row in hmm.filtered_state_probs(y.view()).rows() {
    assert!((row.sum() - 1.0).abs() < 1e-12, "{row}");
  }
}

/// State 0 cannot leave itself, so no observation may route the path through state 1.
#[test]
fn viterbi_never_takes_a_zero_probability_transition() {
  let hmm = GaussianHmm::new(
    array![1.0, 0.0],
    array![[1.0, 0.0], [0.5, 0.5]],
    array![0.0, 10.0],
    array![1.0, 1.0],
  );
  assert_eq!(
    hmm.viterbi(array![0.0, 0.0, 100.0, 0.0].view()).unwrap(),
    array![0, 0, 0, 0]
  );
}

/// Every path through a NaN or infinite point has an undefined score, so none is the most likely.
#[test]
fn viterbi_answers_a_non_finite_observation_with_err() {
  let hmm = two_state(0.5, 0.1, 0.5, 1.0);
  assert_eq!(
    hmm.viterbi(array![-1.0, -0.9, 1.0, 1.1].view()).unwrap(),
    array![1, 1, 1, 1]
  );
  for (index, value) in [(2, f64::NAN), (1, f64::INFINITY), (3, f64::NEG_INFINITY)] {
    let mut y = array![-1.0, -0.9, 1.0, 1.1];
    y[index] = value;
    let err = hmm.viterbi(y.view()).unwrap_err();
    assert!(
      matches!(err, StatsError::InvalidObservation { index: i, constraint: "finite", .. } if i == index),
      "{err:?}"
    );
  }
}

/// With standard deviations of `1e-160` every log density overflows to `−∞`, so no path scores.
#[test]
fn viterbi_answers_a_series_no_path_scores_with_numerical() {
  let hmm = GaussianHmm::new(
    array![0.5, 0.5],
    array![[0.9, 0.1], [0.1, 0.9]],
    array![-0.5, 0.5],
    array![1e-160, 1e-160],
  );
  let observations = Array1::from_iter((0..40).map(|t| 0.3 * (t as f64 * 0.7).sin()));
  assert_eq!(
    hmm.viterbi(observations.view()).unwrap_err(),
    StatsError::Numerical {
      stage: "path score"
    }
  );
}

/// A third state 39 standard deviations from every point keeps only a subnormal posterior mass,
/// and its reestimated transition row still sums to 1.
#[test]
fn a_state_with_a_subnormal_mass_keeps_a_stochastic_row() {
  let mut hmm = GaussianHmm::new(
    array![0.4, 0.4, 0.2],
    array![[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.3, 0.3, 0.4]],
    array![-0.5, 0.5, 39.0],
    array![1.0, 1.0, 1.0],
  );
  let observations = Array1::from_iter((0..200).map(|t| 0.9 * (t as f64 * 0.37).sin()));
  hmm.baum_welch(observations.view(), 1, 1e-12).unwrap();
  for row in hmm.transitions.rows() {
    assert!((row.sum() - 1.0).abs() <= 4.0 * f64::EPSILON, "{row}");
  }
}

/// Data and model scaled by `c` shift the log-likelihood by `−n ln c` and keep every state, which
/// needs Rabiner's floor relative to the data: at `c = 1e-5` the quiet state's variance is `1e-10`.
#[test]
fn a_rescaled_fit_shifts_the_log_likelihood_and_keeps_the_viterbi_path() {
  let mut z = vec![0.0; 600];
  SimdNormal::<f64>::new(0.0, 1.0)
    .seeded(&Deterministic::new(3))
    .fill_slice(&mut z);
  let regime = (0..600).map(|t| (t / 120) % 2).collect::<Vec<_>>();
  let y = Array1::from_iter((0..600).map(|t| [1.0, 3.0][regime[t]] * z[t]));
  let fit = |c: f64| {
    let mut hmm = GaussianHmm::new(
      array![0.5, 0.5],
      array![[0.95, 0.05], [0.05, 0.95]],
      array![0.0, 0.0],
      array![0.8 * c, 2.5 * c],
    );
    let scaled = y.mapv(|v| v * c);
    let fit = hmm.baum_welch(scaled.view(), 200, 1e-10).unwrap();
    (fit, hmm.viterbi(scaled.view()).unwrap())
  };
  let (unit, unit_path) = fit(1.0);
  assert!(unit.converged);
  for c in [1e-3, 1e-5, 1e5] {
    let (scaled, path) = fit(c);
    let shifted = scaled.log_likelihood + 600.0 * c.ln();
    assert!(scaled.converged, "c = {c:e}: {scaled:?}");
    assert!(
      (shifted - unit.log_likelihood).abs() <= 1e-9 * unit.log_likelihood.abs(),
      "c = {c:e}: {shifted} against {}",
      unit.log_likelihood
    );
    assert_eq!(path, unit_path, "c = {c:e}");
  }
}

#[test]
fn a_sample_variance_that_underflows_is_a_numerical_error() {
  let mut hmm = two_state(0.5, 0.1, 0.5, 1.0);
  let observations = Array1::from_iter((0..40).map(|t| 1e-170 * (t as f64 * 0.7).sin()));
  assert_eq!(
    hmm.baum_welch(observations.view(), 5, 1e-8).unwrap_err(),
    StatsError::Numerical {
      stage: "sample variance"
    }
  );
}

/// With standard deviations of `1e-160` every squared z-score overflows, so no density is left to
/// scale by.
#[test]
fn a_log_likelihood_that_is_not_finite_is_a_numerical_error() {
  let mut hmm = GaussianHmm::new(
    array![0.5, 0.5],
    array![[0.9, 0.1], [0.1, 0.9]],
    array![-0.5, 0.5],
    array![1e-160, 1e-160],
  );
  let before = hmm.clone();
  let observations = Array1::from_iter((0..40).map(|t| 0.3 * (t as f64 * 0.7).sin()));
  assert_eq!(
    hmm.baum_welch(observations.view(), 5, 1e-8).unwrap_err(),
    StatsError::Numerical {
      stage: "forward pass"
    }
  );
  assert_eq!(hmm.initial, before.initial);
  assert_eq!(hmm.transitions, before.transitions);
  assert!(hmm.log_likelihood(observations.view()).is_nan());
}

#[test]
fn baum_welch_log_likelihood_does_not_decrease() {
  let mut dist0 = SimdNormal::<f64>::new(-1.0, 0.5).seeded(&Deterministic::new(1));
  let mut dist1 = SimdNormal::<f64>::new(1.5, 0.4).seeded(&Deterministic::new(2));
  let mut a = vec![0.0_f64; 100];
  let mut b = vec![0.0_f64; 100];
  dist0.fill_slice(&mut a);
  dist1.fill_slice(&mut b);
  let mut obs: Vec<f64> = a.into_iter().chain(b).collect();
  obs.shrink_to_fit();
  let observations = ndarray::Array1::from(obs);
  let init = array![0.5, 0.5];
  let trans = array![[0.9, 0.1], [0.1, 0.9]];
  let means = array![-0.5, 0.5];
  let stds = array![1.0, 1.0];
  let mut hmm = GaussianHmm::new(init, trans, means, stds);
  let ll0 = hmm.log_likelihood(observations.view());
  let fit = hmm.baum_welch(observations.view(), 100, 1e-6).unwrap();
  assert!(fit.log_likelihood >= ll0 - 1e-6);
  assert!(hmm.stds.iter().all(|&s| s > 0.0));
}

#[test]
#[should_panic(expected = "initial.len() must satisfy `initial.len() >= 1`, got initial.len() = 0")]
fn an_hmm_without_states_panics() {
  let _ = GaussianHmm::new(
    Array1::zeros(0),
    Array2::zeros((0, 0)),
    Array1::zeros(0),
    Array1::zeros(0),
  );
}

fn two_state(initial: f64, transition: f64, mean: f64, std: f64) -> GaussianHmm {
  GaussianHmm::new(
    array![0.5, initial],
    array![[0.9, 0.1], [transition, 0.9]],
    array![-0.5, mean],
    array![1.0, std],
  )
}

#[test]
#[should_panic(expected = "initial must satisfy `initial.iter().all(|p| p.is_finite())`")]
fn a_nan_initial_probability_panics() {
  let _ = two_state(f64::NAN, 0.1, 0.5, 1.0);
}

#[test]
#[should_panic(expected = "transitions must satisfy `transitions.iter().all(|p| p.is_finite())`")]
fn an_infinite_transition_panics() {
  let _ = two_state(0.5, f64::INFINITY, 0.5, 1.0);
}

#[test]
#[should_panic(expected = "initial must satisfy `initial.iter().all(|&p| p >= 0.0)`")]
fn a_negative_initial_probability_panics() {
  let _ = two_state(-0.5, 0.1, 0.5, 1.0);
}

#[test]
#[should_panic(
  expected = "initial must satisfy `(initial.sum() - 1.0).abs() < 1e-6`, got initial.sum() = 1.2"
)]
fn an_initial_distribution_that_does_not_sum_to_one_panics() {
  let _ = two_state(0.7, 0.1, 0.5, 1.0);
}

#[test]
#[should_panic(expected = "transitions must satisfy `transitions.iter().all(|&p| p >= 0.0)`")]
fn a_negative_transition_probability_panics() {
  let _ = GaussianHmm::new(
    array![0.5, 0.5],
    array![[1.25, -0.25], [0.1, 0.9]],
    array![-0.5, 0.5],
    array![1.0, 1.0],
  );
}

#[test]
#[should_panic(expected = "for every row, got transitions.row(1).sum() = 1.1")]
fn a_transition_row_that_does_not_sum_to_one_panics() {
  let _ = two_state(0.5, 0.2, 0.5, 1.0);
}

/// The fields are public, so every entry point repeats the checks of `new`.
#[test]
fn every_entry_point_checks_a_struct_literal() {
  let hmm = GaussianHmm {
    initial: array![0.5, 0.5],
    transitions: array![[0.9, 0.1], [0.2, 0.9]],
    means: array![-0.5, 0.5],
    stds: array![1.0, 1.0],
  };
  let y = array![0.1, 0.4, -0.2];
  let want = "got transitions.row(1).sum() = 1.1";
  let messages = [
    panic_message(|| {
      hmm.log_likelihood(y.view());
    }),
    panic_message(|| {
      hmm.filtered_state_probs(y.view());
    }),
    panic_message(|| {
      hmm.filter_step(hmm.initial.view(), 0.1);
    }),
    panic_message(|| {
      let _ = hmm.viterbi(y.view());
    }),
    panic_message(|| {
      let _ = hmm.clone().baum_welch(y.view(), 5, 1e-6);
    }),
  ];
  for message in messages {
    assert!(message.contains(want), "{message}");
  }
}

#[test]
#[should_panic(expected = "means must satisfy `means.iter().all(|m| m.is_finite())`")]
fn a_nan_mean_panics() {
  let _ = two_state(0.5, 0.1, f64::NAN, 1.0);
}

#[test]
#[should_panic(expected = "stds must satisfy `stds.iter().all(|&s| s > 0.0 && s.is_finite())`")]
fn an_infinite_std_panics() {
  let _ = two_state(0.5, 0.1, 0.5, f64::INFINITY);
}

#[test]
fn an_empty_series_gives_the_empty_answers() {
  let hmm = two_state(0.5, 0.1, 0.5, 1.0);
  let empty = Array1::<f64>::zeros(0);
  assert_eq!(
    hmm.log_likelihood(empty.view()).to_bits(),
    0.0_f64.to_bits()
  );
  assert_eq!(hmm.filtered_state_probs(empty.view()).dim(), (0, 2));
  assert!(hmm.viterbi(empty.view()).unwrap().is_empty());
}

/// The fields are public, so a model without states can bypass `new`.
#[test]
fn baum_welch_rejects_a_struct_literal_model_without_states() {
  let mut hmm = GaussianHmm {
    initial: Array1::zeros(0),
    transitions: Array2::zeros((0, 0)),
    means: Array1::zeros(0),
    stds: Array1::zeros(0),
  };
  assert_eq!(
    hmm
      .baum_welch(array![0.1, 0.4, -0.2].view(), 5, 1e-6)
      .unwrap_err(),
    StatsError::InvalidParameter {
      name: "initial.len()",
      value: 0.0,
      constraint: "initial.len() >= 1"
    }
  );
}

#[test]
fn baum_welch_rejects_short_or_constant_observations() {
  let mut hmm = GaussianHmm::new(
    array![0.5, 0.5],
    array![[0.9, 0.1], [0.1, 0.9]],
    array![-0.5, 0.5],
    array![1.0, 1.0],
  );
  assert_eq!(
    hmm.baum_welch(array![0.3].view(), 5, 1e-6).unwrap_err(),
    StatsError::InsufficientData { needed: 2, got: 1 }
  );
  assert_eq!(
    hmm
      .baum_welch(Array1::from_elem(50, 0.3).view(), 5, 1e-6)
      .unwrap_err(),
    StatsError::DegenerateData("constant series")
  );
}

/// A point at `1e155` overflows the sample variance that Rabiner's floor is taken from; the `Err`
/// leaves the model as it was.
#[test]
fn a_sample_variance_that_overflows_is_a_numerical_error() {
  let mut hmm = GaussianHmm::new(
    array![0.5, 0.5],
    array![[0.9, 0.1], [0.1, 0.9]],
    array![0.0, 0.0],
    array![1.0, 1e155],
  );
  let before = hmm.clone();
  let mut observations = Array1::from_iter((0..40).map(|t| 0.3 * (t as f64 * 0.7).sin()));
  observations[20] = 1e155;
  assert_eq!(
    hmm.baum_welch(observations.view(), 5, 1e-8).unwrap_err(),
    StatsError::Numerical {
      stage: "sample variance"
    }
  );
  assert_eq!(hmm.initial, before.initial);
  assert_eq!(hmm.transitions, before.transitions);
  assert_eq!(hmm.means, before.means);
  assert_eq!(hmm.stds, before.stds);
}

/// One point `1.35e154` from 39 near 0 keeps the sample variance finite, but its squared distance
/// from the other state's mean overflows that state's variance.
#[test]
fn an_emission_variance_that_overflows_is_a_numerical_error() {
  let d = 1.35e154;
  let mut hmm = GaussianHmm::new(
    array![0.5, 0.5],
    array![[0.9, 0.1], [0.1, 0.9]],
    array![0.0, d],
    array![1.0, 1e153],
  );
  let before = hmm.clone();
  let observations = Array1::from_iter((0..39).map(|t| 0.3 * (t as f64).sin()).chain([d]));
  assert_eq!(
    hmm.baum_welch(observations.view(), 5, 1e-8).unwrap_err(),
    StatsError::Numerical {
      stage: "emission variance"
    }
  );
  assert_eq!(hmm.initial, before.initial);
  assert_eq!(hmm.transitions, before.transitions);
  assert_eq!(hmm.means, before.means);
  assert_eq!(hmm.stds, before.stds);
}
