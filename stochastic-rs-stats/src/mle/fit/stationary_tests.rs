use ndarray::Array1;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_stochastic::diffusion::ou::Ou;

use super::*;
use crate::qmle::fit_ou_mle;
use crate::qmle::test_support::cir;
use crate::qmle::test_support::simulate_ou;

/// The CIR law's own draws, each step on its own stream.
fn cir_path(law: [f64; 3], dt: f64, n: usize, seed: u64) -> Array1<f64> {
  let model = cir(law[0], law[1], law[2]);
  let mut path = Array1::zeros(n);
  path[0] = law[1];
  for t in 1..n {
    let stream = seed.wrapping_mul(1_000_003) ^ (t as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    path[t] = model.sample_transition(dt, path[t - 1], &Deterministic::new(stream));
  }
  path
}

fn fit_ou(path: &Array1<f64>, dt: f64, start: [f64; 3], density: DensityApprox) -> MleFit {
  let mut model = Ou::new(
    start[0],
    start[1],
    start[2],
    2,
    None,
    None,
    Deterministic::new(0),
  );
  fit_mle(&mut model, path.view(), dt, density, None).unwrap()
}

fn fit_cir(path: &Array1<f64>, dt: f64, start: [f64; 3], density: DensityApprox) -> MleFit {
  let mut model = cir(start[0], start[1], start[2]);
  fit_mle(&mut model, path.view(), dt, density, None).unwrap()
}

/// Where a search from (5, −1, 2) can stop on 100 daily points, ~350 below the law's fit, only the
/// verdict runs: the law's edge lies 3–5 % inside the first steps along θ and μ; σ's walk gains.
#[test]
fn an_ozaki_fit_that_stops_against_the_supports_edge_is_not_converged() {
  let dt = 1.0 / 252.0;
  let stops = [
    (0.3, 68, [5.464196259, -1.092867303, 1.978279161]),
    (2.0, 99, [5.504348451, -1.100874201, 1.977574921]),
  ];
  for (speed, seed, at) in stops {
    let path = simulate_ou(speed, 0.03, 0.02, 0.03, dt, 100, seed);
    let best = fit_ou(&path, dt, [speed, 0.03, 0.02], DensityApprox::Ozaki);
    let mut model = Ou::new(at[0], at[1], at[2], 2, None, None, Deterministic::new(0));
    let bounds = model.param_bounds();
    let (lower, upper) = bounds.into_iter().unzip::<_, _, Vec<_>, Vec<_>>();
    let ozaki = DensityApprox::Ozaki;
    let problem = MleProblem::new(&mut model, path.view(), dt, ozaki, &lower, &upper, &at);
    let short = best.log_likelihood + problem.eval_nll(&at);
    let (_, stop) = verdict(&problem, &at, &lower, &upper);
    assert!(
      (300.0..400.0).contains(&short),
      "speed {speed}: {short} below"
    );
    assert!(!stop.converged(), "speed {speed}: converged at {at:?}");
    let message = stop.message(&["theta", "mu", "sigma"]);
    assert_eq!(
      message, "stopped at the edge of the law's support",
      "speed {speed}"
    );
  }
}

/// What the walks beside `at` read on a CIR `path`, and the stop and message they give a search
/// that converged there.
fn cir_walk(path: &Array1<f64>, dt: f64, at: &[f64], density: DensityApprox) -> (Beside, String) {
  let mut model = cir(at[0], at[1], at[2]);
  let bounds = model.param_bounds();
  let (lower, upper): (Vec<f64>, Vec<f64>) = bounds.into_iter().unzip();
  let (beside, stop) = verdict(
    &MleProblem::new(&mut model, path.view(), dt, density, &lower, &upper, at),
    at,
    &lower,
    &upper,
  );
  (beside, stop.message(&model.param_names()))
}

/// From a level 20 times off, a CIR search ends where its platform's rounding takes it, at the
/// optimum or stalled at a speed of `1e-4`: either way the verdict is the walk's beside the fit.
#[test]
fn a_cir_fit_from_a_far_level_answers_as_the_walk_beside_it() {
  let dt = 1.0 / 252.0;
  for (speed, density) in [(0.5, DensityApprox::Ozaki), (8.0, DensityApprox::Elerian)] {
    let path = cir_path([speed, 1.0, 0.5], dt, 500, 2003);
    let fit = fit_cir(&path, dt, [1.0, 0.05, 0.2], density);
    let (beside, message) = cir_walk(&path, dt, fit.params.as_slice().unwrap(), density);
    assert_eq!(
      (fit.converged, &fit.message),
      (message == "converged", &message),
      "speed {speed}: a gain of {:e} beside {}",
      beside.gain,
      fit.params
    );
  }
}

/// The points such a search stalls at: the speed on its bound, where the level has no gradient,
/// and `σ` fitted. No search runs here; the level's walk gains `1.5e-3`, far above the bar.
#[test]
fn a_point_stalled_at_the_speeds_bound_is_not_stationary() {
  let dt = 1.0 / 252.0;
  let cases = [
    (0.5, DensityApprox::Ozaki, [1e-4, 0.0592, 0.5134]),
    (8.0, DensityApprox::Elerian, [1e-4, 0.0560, 0.5089]),
  ];
  for (speed, density, stalled) in cases {
    let path = cir_path([speed, 1.0, 0.5], dt, 500, 2003);
    let (beside, message) = cir_walk(&path, dt, &stalled, density);
    assert!(
      beside.along == 1 && (1e-4..1e-2).contains(&beside.gain),
      "speed {speed}: a gain of {:e} along parameter {}",
      beside.gain,
      beside.along
    );
    assert_eq!(message, "not stationary", "speed {speed}");
  }
}

/// From a `σ` of `1e-6` on data at `1e9` the staged search leaves `σ` `0.3 %` off, `0.01` short:
/// a step along it gains, so the search runs again from that step and ends at the closed form.
#[test]
fn a_fit_whose_sigma_a_search_could_not_move_is_searched_again_from_the_gain() {
  let dt = 1.0 / 252.0;
  let path = simulate_ou(2.0, 1.0, 0.3, 1.0, dt, 1000, 104).mapv(|v| 1e9 * v);
  let closed = fit_ou_mle(path.view(), dt).unwrap();
  let open = (f64::NEG_INFINITY, f64::INFINITY);
  let bounds = vec![(1e-4, 50.0), open, (1e-9, 1e11)];
  let mut model = Ou::new(1.0, 0.0, 1e-6, 2, None, None, Deterministic::new(0));
  let exact = DensityApprox::Exact;
  let fit = fit_mle(&mut model, path.view(), dt, exact, Some(bounds)).unwrap();
  let short = closed.log_likelihood - fit.log_likelihood;
  assert!(short.abs() < 1e-6, "{short} below");
  assert_eq!((fit.converged, &*fit.message), (true, "converged"));
}

/// 16 monthly points leave Ozaki's likelihood a second maximum, 1.2 below the one the fit from
/// the law reaches: no step beside it gains, and it stays converged.
#[test]
fn a_stationary_local_maximum_stays_converged() {
  let dt = 1.0 / 12.0;
  let path = simulate_ou(2.0, 0.03, 0.3, 0.03, dt, 16, 96);
  let best = fit_ou(&path, dt, [2.0, 0.03, 0.3], DensityApprox::Ozaki);
  let fit = fit_ou(&path, dt, [1.0, 0.5, 0.5], DensityApprox::Ozaki);
  let short = best.log_likelihood - fit.log_likelihood;
  assert!(
    (1.0..1.5).contains(&short),
    "{short} below at {}",
    fit.params
  );
  assert_eq!((fit.converged, &*fit.message), (true, "converged"));
  assert!(best.converged);
}

/// A million one-minute points sum to `5.5e6`, whose rounding is the size of the fixed bar: a
/// fit from the closed form, which ends within `1e-6` of it, is converged.
#[test]
fn a_fit_of_a_million_points_from_the_closed_form_is_converged() {
  let dt = 1.0 / (252.0 * 390.0);
  let path = simulate_ou(0.5, 1.0, 0.3, 1.0, dt, 1_000_000, 6);
  let closed = fit_ou_mle(path.view(), dt).unwrap();
  let start = [closed.theta, closed.mu, closed.sigma];
  let fit = fit_ou(&path, dt, start, DensityApprox::Exact);
  let short = closed.log_likelihood - fit.log_likelihood;
  assert!(short.abs() < 1e-6, "{short} below");
  assert_eq!((fit.converged, &*fit.message), (true, "converged"));
}

/// Beside the closed form no walk gains and none leaves the support; with `σ` one percent off
/// the walk along it gains what the fit is short of, `n δ²`.
#[test]
fn the_walk_beside_an_estimate_gains_what_one_parameter_is_short_of() {
  let dt = 1.0 / 252.0;
  let path = simulate_ou(2.0, 1.0, 0.3, 1.0, dt, 1000, 101);
  let closed = fit_ou_mle(path.view(), dt).unwrap();
  let (lower, upper) = ([1e-4, -10.0, 1e-6], [50.0, 10.0, 10.0]);
  let beside = |at: [f64; 3]| {
    let mut model = Ou::new(1.0, 0.5, 0.5, 2, None, None, Deterministic::new(0));
    let exact = DensityApprox::Exact;
    MleProblem::new(&mut model, path.view(), dt, exact, &lower, &upper, &at).beside(&at)
  };
  let optimum = beside([closed.theta, closed.mu, closed.sigma]);
  assert!(optimum.gain < 1e-8 && !optimum.walled, "{}", optimum.gain);
  let off = beside([closed.theta, closed.mu, 1.01 * closed.sigma]);
  assert!((0.08..0.11).contains(&off.gain), "{}", off.gain);
  assert_eq!(off.along, 2);
  let rounding = f64::EPSILON * 999.0_f64.sqrt() * (1.0 + closed.log_likelihood);
  assert!(
    (off.rounding / rounding - 1.0).abs() < 1e-3,
    "{}",
    off.rounding
  );
}

/// The stop's words are quant's, and a bound is named by its parameter.
#[test]
fn a_stop_is_worded_as_the_calibrators_word_theirs() {
  let names = ["theta", "mu", "sigma"];
  let said = |stop: Stop| stop.message(&names);
  assert_eq!(said(Stop::Converged), "converged");
  assert_eq!(said(Stop::NotStationary), "not stationary");
  assert_eq!(said(Stop::Bound(0)), "stopped at the bound of theta");
  assert_eq!(said(Stop::Edge), "stopped at the edge of the law's support");
  let budget = Stop::Search(SolverStop::IterationCap);
  assert_eq!(said(budget), "evaluation budget exhausted");
  let spent = Stop::Search(SolverStop::EvaluationBudget);
  assert_eq!(said(spent), "evaluation budget exhausted");
  let stuck = Stop::Search(SolverStop::NoProgress);
  assert_eq!(said(stuck), "no numerical progress");
  let failed = Stop::Search(SolverStop::Failed);
  assert_eq!(said(failed), "solver failed");
  let other = Stop::Search(SolverStop::Other);
  assert_eq!(said(other), "stopped without convergence");
  assert_eq!(said(Stop::Outside), "stopped without convergence");
}

/// A gain counts above `1e-7` and above the sum's rounding; the bound named is that of the
/// parameter whose walk gains, and a wall says `edge` only where none is.
#[test]
fn a_gain_counts_above_the_rounding_and_names_the_parameter_it_is_along() {
  let read = |gain: f64, rounding: f64, walled: bool, held: bool| {
    let beside = Beside {
      gain,
      along: 1,
      rounding,
      walled,
      best: Vec::new(),
    };
    beside.stop(held).message(&["theta", "mu", "sigma"])
  };
  assert_eq!(read(1e-7, 1e-9, true, true), "converged");
  assert_eq!(read(2e-7, 1e-9, false, false), "not stationary");
  assert_eq!(read(2e-7, 3e-7, false, false), "converged");
  assert_eq!(read(4e-7, 3e-7, false, true), "stopped at the bound of mu");
  let edge = "stopped at the edge of the law's support";
  assert_eq!(read(4e-7, 3e-7, true, false), edge);
}
