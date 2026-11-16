use ndarray::array;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_distributions::DistributionSampler;
use stochastic_rs_distributions::SimdDistribution;
use stochastic_rs_distributions::normal::SimdNormal;

use super::*;

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
  let path = hmm.viterbi(obs.view());
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

#[test]
fn filter_step_survives_an_implausible_observation() {
  let init = array![0.5, 0.5];
  let trans = array![[0.9, 0.1], [0.2, 0.8]];
  let means = array![-1.0, 1.0];
  let stds = array![0.1, 0.1];
  let hmm = GaussianHmm::new(init, trans, means, stds);

  let next = hmm.filter_step(hmm.initial.view(), 1e9);

  assert!((next.sum() - 1.0).abs() < 1e-9, "must stay a distribution");
  assert!(next.iter().all(|p| p.is_finite()), "must not emit NaN");
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
  assert!(hmm.viterbi(empty.view()).is_empty());
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
