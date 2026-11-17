use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_stochastic::diffusion::cir::Cir;
use stochastic_rs_stochastic::diffusion::ou::Ou;
use stochastic_rs_stochastic::params::VarianceBoundary;

use super::*;
use crate::qmle::fit_ou_mle;
use crate::qmle::test_support::simulate_ou;

#[test]
fn mle_converges_at_initial_brownian_drift_optimum() {
  struct BrownianDrift {
    mu: f64,
  }

  impl DiffusionModel for BrownianDrift {
    fn num_params(&self) -> usize {
      1
    }

    fn params(&self) -> Array1<f64> {
      ndarray::array![self.mu]
    }

    fn set_params(&mut self, params: &[f64]) {
      self.mu = params[0];
    }

    fn param_names(&self) -> Vec<&str> {
      vec!["mu"]
    }

    fn param_bounds(&self) -> Vec<(f64, f64)> {
      vec![(-10.0, 10.0)]
    }

    fn drift(&self, _x: f64, _t: f64) -> f64 {
      self.mu
    }

    fn diffusion(&self, _x: f64, _t: f64) -> f64 {
      1.0
    }
  }

  let sample = ndarray::array![0.0, 1.0, 0.0];
  let mut model = BrownianDrift { mu: 0.0 };
  let result = fit_mle(&mut model, sample.view(), 1.0, DensityApprox::Euler, None).unwrap();

  assert!(result.converged, "The initial drift is already the MLE");
  assert_eq!(result.iterations, 0);
  assert_eq!(result.params, ndarray::array![0.0]);
  assert_eq!(model.mu, 0.0);
}

#[test]
fn bounded_mle_cost_rejects_negative_sigma_mirror_minimum() {
  let sample = Array1::from_vec(vec![0.1, 0.2, 0.15]);
  let mut model = Cir::new(
    1.0,
    0.1,
    3.0,
    3,
    None,
    None,
    VarianceBoundary::Floor,
    Deterministic::new(0),
  );
  let problem = MleProblem {
    model: Mutex::new(&mut model),
    sample: sample.view(),
    dt: 0.01,
    density: DensityApprox::Euler,
    lower: vec![1e-4, 1e-6, 1e-6],
    upper: vec![50.0, 10.0, 10.0],
  };
  let positive = problem.eval_nll(&[1.0, 0.1, 3.0]);
  let negative = problem.eval_nll(&[1.0, 0.1, -3.0]);
  let boundary = problem.eval_nll(&[1.0, 0.1, 1e-6]);

  assert_eq!(negative, boundary);
  assert!(negative > positive);
}

#[test]
fn bounded_mle_gradient_uses_the_feasible_boundary() {
  let sample = Array1::from_vec(vec![0.1, 0.2, 0.15]);
  let mut model = Cir::new(
    1.0,
    0.1,
    3.0,
    3,
    None,
    None,
    VarianceBoundary::Floor,
    Deterministic::new(0),
  );
  let problem = MleProblem {
    model: Mutex::new(&mut model),
    sample: sample.view(),
    dt: 0.01,
    density: DensityApprox::Euler,
    lower: vec![1e-4, 1e-6, 1e-6],
    upper: vec![50.0, 10.0, 10.0],
  };
  let outside = problem.gradient(&vec![1.0, 0.1, 13.0]).unwrap();
  let boundary = problem.gradient(&vec![1.0, 0.1, 10.0]).unwrap();

  // At this upper bound the likelihood improves toward smaller sigma.
  assert!(boundary[2] > 0.0);
  assert_eq!(outside, boundary);
}

/// A negative point has no exact CIR density at any parameter, so the objective is its penalty
/// and the log-likelihood of the fit not finite.
#[test]
fn a_path_without_a_density_is_penalised_and_its_fit_an_err() {
  let mut model = Cir::new(
    1.0,
    0.1,
    0.3,
    3,
    None,
    None,
    VarianceBoundary::Floor,
    Deterministic::new(0),
  );
  for (sample, penalised) in [
    (ndarray::array![0.1, 0.2, 0.15], false),
    (ndarray::array![0.1, -0.2, 0.15], true),
  ] {
    let problem = MleProblem {
      model: Mutex::new(&mut model),
      sample: sample.view(),
      dt: 0.01,
      density: DensityApprox::Exact,
      lower: vec![1e-4, 1e-6, 1e-6],
      upper: vec![50.0, 10.0, 10.0],
    };
    let nll = problem.eval_nll(&[1.0, 0.1, 0.3]);
    assert_eq!(nll == NO_LIKELIHOOD, penalised, "nll = {nll}");
    assert!(nll.is_finite());
    drop(problem);
    let fit = fit_mle(&mut model, sample.view(), 0.01, DensityApprox::Exact, None);
    assert_eq!(
      fit.map(|f| f.log_likelihood.is_finite()),
      if penalised {
        Err(StatsError::Numerical {
          stage: "log-likelihood",
        })
      } else {
        Ok(true)
      }
    );
  }
}

/// From a state of zero Euler's CIR variance is zero, where the exact law has its central χ²
/// density: the same path is an `Err` by Euler and a fit by the law.
#[test]
fn a_cir_path_from_zero_fits_by_the_law_and_not_by_euler() {
  let sample = ndarray::array![0.0, 0.02, 0.03, 0.025, 0.04, 0.035, 0.05, 0.04];
  let fit = |density| {
    let mut model = Cir::new(
      1.0,
      0.05,
      0.2,
      3,
      None,
      None,
      VarianceBoundary::Floor,
      Deterministic::new(0),
    );
    fit_mle(&mut model, sample.view(), 0.01, density, None).map(|f| f.log_likelihood.is_finite())
  };
  assert_eq!(fit(DensityApprox::Exact), Ok(true));
  assert_eq!(
    fit(DensityApprox::Euler),
    Err(StatsError::Numerical {
      stage: "log-likelihood"
    })
  );
}

/// At a `σ` far too small every transition lies where its normal density is zero: the objective
/// sums their log-densities, graded in `σ`, and keeps the penalty for a point with no likelihood.
#[test]
fn an_underflowing_density_keeps_its_log_in_the_objective() {
  let sample = ndarray::array![0.5, 0.52, 0.49, 0.51];
  let mut model = Ou::new(
    1.0,
    0.5,
    0.5,
    100,
    Some(0.5),
    Some(1.0),
    Deterministic::new(0),
  );
  model.set_params(&[1.0, 0.5, 1e-6]);
  assert_eq!(
    DensityApprox::Euler.density(&model, 0.5, 0.52, 0.0, 0.01),
    0.0
  );
  let problem = MleProblem {
    model: Mutex::new(&mut model),
    sample: sample.view(),
    dt: 0.01,
    density: DensityApprox::Euler,
    lower: vec![1e-4, -10.0, 1e-6],
    upper: vec![50.0, 10.0, 10.0],
  };
  let narrow = problem.eval_nll(&[1.0, 0.5, 1e-6]);
  let wider = problem.eval_nll(&[1.0, 0.5, 2e-6]);
  assert!(narrow > 1e10 && narrow < 1e12, "nll = {narrow}");
  assert!(wider < 0.3 * narrow, "{wider} against {narrow}");
}

/// On data near `1e9` the sum at `σ = 1e-6` is finite and past `1e30`: it ranks below a point with
/// no likelihood, and a wider `σ` below it.
#[test]
fn a_point_with_a_likelihood_ranks_below_one_without() {
  let sample = ndarray::array![1e9, 3e9, 2e9, 4e9];
  let mut model = Ou::new(
    1.0,
    0.5,
    0.5,
    100,
    Some(0.5),
    Some(1.0),
    Deterministic::new(0),
  );
  let problem = MleProblem {
    model: Mutex::new(&mut model),
    sample: sample.view(),
    dt: 0.01,
    density: DensityApprox::Euler,
    lower: vec![1e-4, -10.0, 1e-6],
    upper: vec![50.0, 10.0, 10.0],
  };
  let narrow = problem.eval_nll(&[1.0, 0.5, 1e-6]);
  let wider = problem.eval_nll(&[1.0, 0.5, 2e-6]);
  assert!(
    wider < narrow && narrow < NO_LIKELIHOOD,
    "{wider}, {narrow}"
  );
}

/// A far start on data near `1e12` sums to about `1e39`, which the search must read as it is.
#[test]
fn the_ranking_keeps_a_sum_below_its_knee_and_every_finite_one_below_the_penalty() {
  for nll in [-1e300, -3.5, 0.0, 39.38, 4.5e32, 1e39, 1e97] {
    assert_eq!(ranked(nll), nll);
  }
  let past = [1e97, 1.0000001e97, 1e100, 1e200, f64::MAX].map(ranked);
  assert!(past.windows(2).all(|w| w[0] < w[1]), "{past:?}");
  assert!(past[4] < NO_LIKELIHOOD);
  let slope = (ranked(1.0000001e97) - 1e97) / 1e90;
  assert!((slope - 1.0).abs() < 1e-6, "slope {slope}");
}

/// From `σ = 0.5` the search probes `σ` near its bound, where every density underflows; summed
/// in logs the objective leads it back, to the closed form's optimum.
#[test]
fn a_fit_through_underflowing_densities_reaches_the_closed_form() {
  let dt = 1.0 / 12.0;
  for seed in [1u64, 2, 3] {
    let path = simulate_ou(8.0, 0.03, 0.02, 0.03, dt, 1000, seed);
    let closed = fit_ou_mle(path.view(), dt).unwrap();
    let mut model = Ou::new(
      1.0,
      0.5,
      0.5,
      100,
      Some(1.0),
      Some(1.0),
      Deterministic::new(0),
    );
    let fit = fit_mle(&mut model, path.view(), dt, DensityApprox::Exact, None).unwrap();
    assert!(fit.converged, "seed {seed}");
    assert!(
      (fit.log_likelihood - closed.log_likelihood).abs() < 1e-6,
      "seed {seed}: {} for {}",
      fit.log_likelihood,
      closed.log_likelihood
    );
    assert!(
      (fit.params[0] / closed.theta - 1.0).abs() < 1e-4,
      "seed {seed}: θ̂ = {} for {}",
      fit.params[0],
      closed.theta
    );
  }
}

#[test]
fn mle_result_signals_non_convergence_without_error() {
  let best = vec![0.5];
  let (params, converged, iterations) =
    resolve_fit_outcome(best.clone(), TerminationReason::MaxIter, 200);

  assert_eq!(params, best);
  assert!(!converged, "MaxIter must not be reported as converged");
  assert_eq!(iterations, 200);
}

#[test]
fn mle_result_retains_best_point_on_solver_failure() {
  let best = vec![1.0, 2.0, 3.0];
  let (params, converged, iterations) =
    resolve_fit_outcome(best.clone(), TerminationReason::SolverFailed, 7);

  assert_eq!(params, best);
  assert!(!converged);
  assert_eq!(iterations, 7);
}

#[test]
fn mle_result_signals_genuine_convergence() {
  let fitted = vec![1.23, 4.56];
  let (params, converged, iterations) =
    resolve_fit_outcome(fitted.clone(), TerminationReason::SolverConverged, 17);

  assert_eq!(params, fitted);
  assert!(converged);
  assert_eq!(iterations, 17);
}
