use stochastic_rs_distributions::RealExt;

use super::*;

struct Rosenbrock {
  x: Array1<f64>,
  calls: Cell<usize>,
  fail_at: Option<usize>,
  reject_outside_domain: bool,
  nonfinite_trials: Cell<usize>,
  accepted: Vec<Array1<f64>>,
}

impl Rosenbrock {
  fn new() -> Self {
    Self {
      x: Array1::from_vec(vec![-1.2, 1.0]),
      calls: Cell::new(0),
      fail_at: None,
      reject_outside_domain: false,
      nonfinite_trials: Cell::new(0),
      accepted: Vec::new(),
    }
  }

  fn cost(x: &Array1<f64>) -> f64 {
    50.0 * (x[1] - x[0].powi(2)).powi(2) + 0.5 * (1.0 - x[0]).powi(2)
  }
}

impl LeastSquaresProblem for Rosenbrock {
  fn set_params(&mut self, x: &Array1<f64>) {
    self.x.clone_from(x);
  }

  fn params(&self) -> Array1<f64> {
    self.x.clone()
  }

  fn residuals(&mut self) -> Option<Array1<f64>> {
    self.calls.set(self.calls.get() + 1);
    if self.fail_at == Some(self.calls.get()) {
      return None;
    }
    if self.reject_outside_domain && self.x[1] < -2.0 {
      self.nonfinite_trials.set(self.nonfinite_trials.get() + 1);
      return Some(Array1::from_elem(2, f64::NAN));
    }
    Some(Array1::from_vec(vec![
      10.0 * (self.x[1] - self.x[0].powi(2)),
      1.0 - self.x[0],
    ]))
  }

  fn jacobian(&self) -> Option<Array2<f64>> {
    Array2::from_shape_vec((2, 2), vec![-20.0 * self.x[0], 10.0, -1.0, 0.0]).ok()
  }

  fn accept(&mut self) {
    self.accepted.push(self.x.clone());
  }
}

#[test]
fn the_start_and_each_trial_that_lowers_the_cost_are_accepted() {
  for pivoted_qr in [false, true] {
    let mut problem = Rosenbrock::new();
    problem.reject_outside_domain = true;
    let (problem, report) = minimize(
      problem,
      LmOptions {
        pivoted_qr,
        ..LmOptions::default()
      },
    );
    let accepted = &problem.accepted;
    assert_eq!(accepted[0], Rosenbrock::new().x);
    assert!(problem.nonfinite_trials.get() > 0 && accepted.iter().all(|x| x[1] >= -2.0));
    assert!(accepted.len() < report.evaluations);
    assert!(
      accepted
        .windows(2)
        .all(|pair| Rosenbrock::cost(&pair[1]) < Rosenbrock::cost(&pair[0]))
    );
    assert_eq!(accepted.last(), Some(&problem.x));
  }
}

#[test]
fn rosenbrock_converges_with_consistent_evaluation_counts() {
  for pivoted_qr in [false, true] {
    let (problem, report) = minimize(
      Rosenbrock::new(),
      LmOptions {
        pivoted_qr,
        ..LmOptions::default()
      },
    );
    assert!(report.converged, "{report:?}");
    assert_eq!(report.message(), "converged");
    assert!(distance_to_optimum(&problem.x) < 1e-8);
    assert_eq!(report.evaluations, problem.calls.get());
  }
}

#[test]
fn the_report_builds_the_calibrated_fit() {
  use crate::traits::CalibrationResult;
  let (problem, report) = minimize(Rosenbrock::new(), LmOptions::default());
  let fit = report
    .calibrated(
      problem.x.to_vec(),
      &[1.0, 2.0],
      &[1.5, 2.0],
      &LossMetric::ALL,
      vec![],
    )
    .unwrap();
  assert!(fit.converged() && fit.history().is_empty());
  assert_eq!(fit.iterations(), Some(report.iterations));
  assert_eq!(fit.message(), Some("converged"));
  assert_eq!(fit.max_error(), Some(0.5));
  assert_eq!(fit.rmse(), 0.125_f64.sqrt());
}

#[test]
fn a_trail_keeps_the_last_staged_point_once_per_accept_and_numbers_it() {
  let options = CalibrationOptions::default().with_record_history(true);
  let (market, rows) = ([1.0, 2.0], Array1::from_vec(vec![0.5, -1.0]));
  let mut trail = Trail::new();
  trail.stage(&options, &market, &[0.5, 3.0], &0.1, &rows);
  trail.accept();
  trail.stage(&options, &market, &[9.0, 9.0], &0.2, &rows);
  trail.stage(&options, &market, &[1.0, 2.0], &0.3, &rows);
  trail.accept();
  trail.accept();
  let entries = trail.into_entries();
  let numbered = entries.iter().map(|e| (e.iteration, e.params));
  assert_eq!(numbered.collect::<Vec<_>>(), vec![(0, 0.1), (1, 0.3)]);
  assert!(entries.iter().all(|entry| entry.residuals == rows));
  assert_eq!(entries[0].loss.get(LossMetric::Mae), 0.75);
  assert_eq!(entries[1].loss.get(LossMetric::Rmse), 0.0);
}

#[test]
fn a_negated_trail_stores_the_rows_negated_and_an_unrecorded_one_nothing() {
  let rows = Array1::from_vec(vec![0.5, -0.0, f64::NAN]);
  let mut recorded = Trail::negated();
  let options = CalibrationOptions::default().with_record_history(true);
  recorded.stage(&options, &[1.0; 3], &[1.0; 3], &(), &rows);
  recorded.accept();
  let stored = &recorded.into_entries()[0].residuals;
  let bits = |v: &Array1<f64>| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
  assert_eq!(bits(stored), bits(&rows.mapv(|x| -x)));
  let mut unrecorded = Trail::negated();
  unrecorded.stage(&CalibrationOptions::default(), &[1.0], &[2.0], &(), &rows);
  unrecorded.accept();
  assert!(unrecorded.into_entries().is_empty());
}

#[test]
fn evaluation_budget_is_not_convergence() {
  let (problem, report) = minimize(
    Rosenbrock::new(),
    LmOptions {
      patience: 1,
      ..LmOptions::default()
    },
  );
  assert!(!report.converged, "{report:?}");
  assert_eq!(report.reason, TerminationReason::MaxCostEvals);
  assert_eq!(report.message(), "evaluation budget exhausted");
  assert_eq!(report.evaluations, 3);
  assert!(problem.x.iter().all(|v| v.is_finite()));
}

#[test]
fn an_iteration_cap_below_the_budget_stops_the_fit() {
  let options = |max_iterations| LmOptions {
    max_iterations,
    ..LmOptions::default()
  };
  let (free, free_report) = minimize(Rosenbrock::new(), options(None));
  let (capped, report) = minimize(Rosenbrock::new(), options(Some(2)));
  assert!(free_report.converged);
  assert!(!report.converged, "{report:?}");
  assert_eq!(report.reason, TerminationReason::MaxIter);
  assert!(report.evaluations < free_report.evaluations);
  assert!(distance_to_optimum(&capped.x) > distance_to_optimum(&free.x));
  let fit = report
    .calibrated(capped.x.to_vec(), &[1.0], &[1.0], &LossMetric::ALL, vec![])
    .unwrap();
  assert!(fit.iterations.unwrap() <= 2, "{:?}", fit.iterations);
  assert!(fit.iterations.unwrap() < report.evaluations);
}

#[test]
fn failed_trial_restores_the_last_accepted_parameters() {
  let mut problem = Rosenbrock::new();
  let initial = problem.params();
  problem.fail_at = Some(2);
  let (problem, report) = minimize(problem, LmOptions::default());
  assert!(!report.converged);
  assert_eq!(report.reason, TerminationReason::SolverFailed);
  assert_eq!(report.message(), "solver failed");
  assert_eq!(report.evaluations, 2);
  assert_eq!(problem.params(), initial);
}

#[test]
fn failed_initial_evaluation_is_reported_without_panicking() {
  let mut problem = Rosenbrock::new();
  problem.fail_at = Some(1);
  let (problem, report) = minimize(problem, LmOptions::default());
  assert!(!report.converged && !report.started);
  assert_eq!(report.evaluations, 1);
  let fit = report.calibrated(problem.x.to_vec(), &[1.0], &[1.0], &[], vec![]);
  assert!(
    matches!(fit, Err(QuantError::Optimizer { .. })),
    "a fit with no iterate is an error, got {fit:?}"
  );
}

#[test]
fn nonfinite_residual_is_not_convergence() {
  let mut problem = Rosenbrock::new();
  problem.x[0] = f64::NAN;
  let (_, report) = minimize(problem, LmOptions::default());
  assert!(!report.converged);
  assert_eq!(report.reason, TerminationReason::SolverFailed);
}

#[test]
fn nonfinite_trial_shrinks_the_step_and_recovers() {
  for pivoted_qr in [false, true] {
    let mut problem = Rosenbrock::new();
    problem.reject_outside_domain = true;
    let (problem, report) = minimize(
      problem,
      LmOptions {
        pivoted_qr,
        ..LmOptions::default()
      },
    );
    assert!(problem.nonfinite_trials.get() > 0);
    assert!(report.converged, "{report:?}");
    assert!(distance_to_optimum(&problem.x) < 1e-8);
  }
}

#[test]
fn a_pull_appends_its_rows_and_their_scaled_jacobian() {
  let objective = Objective {
    model: RefCell::new(Rosenbrock::new()),
    coordinates: RefCell::new(None),
    evaluations: Cell::new(0),
    norm: Cell::new(0.0),
    sizes: RefCell::new(Array1::zeros(2)),
    flat: Cell::new(false),
    regularization: Some(Regularization::new(vec![0.5, 2.0], vec![4.0, 9.0])),
  };
  let x = Array1::from_vec(vec![-1.2, 1.0]);
  let residuals = Residual::residual(&&objective, &x).unwrap();
  let rows = residuals.slice(ndarray::s![2..]).to_vec();
  assert_eq!(rows, vec![2.0 * (-1.2 - 0.5), 3.0 * (1.0 - 2.0)]);
  let jacobian = Jacobian::jacobian(&&objective, &x).unwrap();
  assert_eq!(jacobian.nrows(), 4);
  let pulled = jacobian.slice(ndarray::s![2.., ..]).to_owned();
  assert_eq!(pulled, ndarray::array![[2.0, 0.0], [0.0, 3.0]]);
}

#[test]
fn a_heavy_pull_holds_its_parameter_at_the_anchor() {
  let options = LmOptions {
    regularization: Some(Regularization::new(vec![0.5, 0.0], vec![1e8, 0.0])),
    ..LmOptions::default()
  };
  let (problem, report) = minimize(Rosenbrock::new(), options);
  assert!(report.converged, "{report:?}");
  assert!((problem.x[0] - 0.5).abs() < 1e-6, "{}", problem.x[0]);
  assert!((problem.x[1] - 0.25).abs() < 1e-4, "{}", problem.x[1]);
}

#[test]
fn a_zero_weight_pull_is_the_unpulled_fit_bit_for_bit() {
  let pulled = LmOptions {
    regularization: Some(Regularization::uniform(vec![0.5, 0.5], 0.0)),
    ..LmOptions::default()
  };
  let (plain, plain_report) = minimize(Rosenbrock::new(), LmOptions::default());
  let (zero, zero_report) = minimize(Rosenbrock::new(), pulled);
  let bits = |x: &Array1<f64>| x.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
  assert_eq!(bits(&plain.x), bits(&zero.x));
  assert_eq!(plain_report.evaluations, zero_report.evaluations);
}

fn distance_to_optimum(x: &Array1<f64>) -> f64 {
  let distance = x - &Array1::from_elem(2, 1.0);
  distance.dot(&distance).sqrt()
}

/// `∂θ/∂x` of a problem's natural parameters by central differences of step `h` around `x`.
pub(crate) fn natural_jacobian_by_differences<P: LeastSquaresProblem>(
  problem: &mut P,
  x: &Array1<f64>,
  h: f64,
) -> Array2<f64> {
  let mut at = |y: &Array1<f64>| {
    problem.set_params(y);
    Array1::from_vec(problem.natural_params())
  };
  let rows = at(x).len();
  let mut jacobian = Array2::zeros((rows, x.len()));
  for column in 0..x.len() {
    let (mut up, mut down) = (x.clone(), x.clone());
    up[column] += h;
    down[column] -= h;
    let derivative = (at(&up) - at(&down)) / (2.0 * h);
    jacobian.column_mut(column).assign(&derivative);
  }
  problem.set_params(x);
  jacobian
}

/// The largest entrywise `|a − b| / max(|a|, |b|)`, zero where both are zero; NaN propagates.
pub(crate) fn worst_relative_error(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
  let ratio = |(x, y): (&f64, &f64)| {
    let scale = x.abs().max(y.abs());
    if scale == 0.0 {
      0.0
    } else {
      (x - y).abs() / scale
    }
  };
  f64::max_or_nan_fold(a.iter().zip(b).map(ratio))
}

/// MINPACK's `chkder` as a test: the worst column's `‖J − D‖/‖D‖` at `x`, `D` the central
/// difference of the residuals over `step(x_j)`; two zero columns read 0.
pub(crate) fn jacobian_error<P: LeastSquaresProblem>(
  problem: &mut P,
  x: &Array1<f64>,
  step: impl Fn(usize, f64) -> f64,
) -> f64 {
  let mut at = |y: &Array1<f64>| {
    problem.set_params(y);
    problem.residuals().expect("residuals beside the point")
  };
  let columns = (0..x.len())
    .map(|column| {
      let h = step(column, x[column]);
      let (mut up, mut down) = (x.clone(), x.clone());
      up[column] += h;
      down[column] -= h;
      (at(&up) - at(&down)) / (2.0 * h)
    })
    .collect::<Vec<_>>();
  problem.set_params(x);
  let jacobian = problem.jacobian().expect("a Jacobian at the point");
  let norm = |v: &Array1<f64>| v.dot(v).sqrt();
  f64::max_or_nan_fold(columns.iter().enumerate().map(|(j, difference)| {
    let gap = norm(&(&jacobian.column(j) - difference));
    let scale = norm(difference);
    if scale == 0.0 { gap } else { gap / scale }
  }))
}

/// `base`, then per coordinate each bound and, where the box reflects, `base`'s mirror image
/// below the lower one; 1 % inside a bound, since a stencil that reaches it is one-sided.
pub(crate) fn box_points(
  base: &Array1<f64>,
  bounds: &[(f64, f64)],
  reflecting: &[bool],
) -> Vec<(String, Array1<f64>)> {
  let mut points = vec![("base".to_string(), base.clone())];
  for (j, &(lower, upper)) in bounds.iter().enumerate() {
    let inset = 0.01 * (upper - lower);
    let mirror = reflecting[j].then_some(("mirror", 2.0 * lower - base[j]));
    let sides = [("lower", lower + inset), ("upper", upper - inset)];
    for (side, value) in sides.into_iter().chain(mirror) {
      let mut point = base.clone();
      point[j] = value;
      points.push((format!("{side} {j}"), point));
    }
  }
  points
}

struct Flat;

impl LeastSquaresProblem for Flat {
  fn set_params(&mut self, _: &Array1<f64>) {}

  fn params(&self) -> Array1<f64> {
    Array1::from_vec(vec![1.0])
  }

  fn residuals(&mut self) -> Option<Array1<f64>> {
    Some(Array1::from_vec(vec![1.0, 2.0]))
  }

  fn jacobian(&self) -> Option<Array2<f64>> {
    Some(Array2::zeros((2, 1)))
  }
}

/// Moré's gradient test holds where the Jacobian is zero, with nothing fitted.
#[test]
fn residuals_that_no_parameter_moves_are_not_a_convergence() {
  for pivoted_qr in [false, true] {
    let options = LmOptions {
      pivoted_qr,
      ..LmOptions::default()
    };
    let (_, report) = minimize(Flat, options);
    assert!(!report.converged, "{report:?}");
    assert_eq!(report.message(), "flat objective");
  }
}

/// `r = [slope·(x − target), offset]` with the Jacobian's sign as given.
struct Line {
  x: f64,
  target: f64,
  offset: f64,
  jacobian_sign: f64,
}

impl LeastSquaresProblem for Line {
  fn set_params(&mut self, x: &Array1<f64>) {
    self.x = x[0];
  }

  fn params(&self) -> Array1<f64> {
    Array1::from_vec(vec![self.x])
  }

  fn residuals(&mut self) -> Option<Array1<f64>> {
    Some(Array1::from_vec(vec![self.x - self.target, self.offset]))
  }

  fn jacobian(&self) -> Option<Array2<f64>> {
    Some(Array2::from_shape_vec((2, 1), vec![self.jacobian_sign, 0.0]).unwrap())
  }
}

/// Every trial of a wrong-signed Jacobian goes uphill, the trust region collapses, and Moré's
/// (8.3) passes at the start.
#[test]
fn a_collapsed_trust_region_away_from_a_stationary_point_is_not_converged() {
  let line = Line {
    x: 1.0,
    target: 3.0,
    offset: 1.0,
    jacobian_sign: -1.0,
  };
  let (fitted, report) = minimize(line, LmOptions::default());
  assert!(!report.converged && report.stationarity > 0.5, "{report:?}");
  assert_eq!((fitted.x, report.message()), (1.0, "not stationary"));
  let fit = report
    .calibrated(fitted.x, &[3.0], &[1.0], &[], Vec::new())
    .unwrap();
  assert!(!fit.converged);
  assert_eq!(fit.message.as_deref(), Some("not stationary"));
  let exact = report.settled(3e-8, &[3.0], PRICER_ACCURACY);
  assert_eq!((exact.converged, exact.message()), (true, "converged"));
  let short = report.settled(4e-8, &[3.0], PRICER_ACCURACY);
  assert_eq!(
    (short.converged, short.message()),
    (false, "not stationary")
  );
  assert!(report.settled(4e-8, &[3.0], 2e-8).converged);
}

/// The minimum at `1e-12` is small against `ε‖r‖/‖J‖`, which is not a flat residual.
#[test]
fn a_minimum_near_the_origin_is_not_flat() {
  let line = Line {
    x: 1.0,
    target: 1e-12,
    offset: 1e5,
    jacobian_sign: 1.0,
  };
  let (fitted, report) = minimize(line, LmOptions::default());
  assert!(report.converged, "{report:?}");
  assert!((fitted.x - 1e-12).abs() < 1e-15, "{}", fitted.x);
}

#[derive(Clone, Copy, Debug)]
enum Map {
  /// Onto (−1, 1), as Heston's and CGMYSV's ρ.
  Logistic,
  Reflection,
  Clamp,
}

/// `r = [θ(x) − target, 0.05]`, θ the map's image of `x` with its bound at 0 or at ±1.
struct Mapped {
  x: f64,
  target: f64,
  map: Map,
}

impl Mapped {
  fn theta(&self) -> f64 {
    match self.map {
      Map::Logistic => -1.0 + 2.0 / (1.0 + (-self.x).exp()),
      Map::Reflection => self.x.abs(),
      Map::Clamp => self.x.max(0.0),
    }
  }
}

impl LeastSquaresProblem for Mapped {
  fn set_params(&mut self, x: &Array1<f64>) {
    self.x = x[0];
  }

  fn params(&self) -> Array1<f64> {
    Array1::from_vec(vec![self.x])
  }

  fn residuals(&mut self) -> Option<Array1<f64>> {
    Some(Array1::from_vec(vec![self.theta() - self.target, 0.05]))
  }

  fn jacobian(&self) -> Option<Array2<f64>> {
    let unit = 1.0 / (1.0 + (-self.x).exp());
    let slope = match self.map {
      Map::Logistic => 2.0 * unit * (1.0 - unit),
      Map::Reflection if self.x < 0.0 => -1.0,
      Map::Reflection | Map::Clamp => 1.0,
    };
    Some(Array2::from_shape_vec((2, 1), vec![slope, 0.0]).unwrap())
  }
}

/// A tolerance of 1e3 has basin report convergence after its first trial, kept only if it
/// lowers the cost: the first-order check then runs at a chosen point.
fn stopped(map: Map, x: f64, target: f64) -> (f64, LmReport) {
  let options = LmOptions {
    tolerance: Some(1e3),
    ..LmOptions::default()
  };
  let (fitted, report) = minimize(Mapped { x, target, map }, options);
  (fitted.x, report)
}

/// The whole Gauss–Newton step lands past the minimum at a higher cost; a quarter of it does not.
#[test]
fn a_stop_whose_full_step_overshoots_the_minimum_is_not_stationary() {
  for (map, x, target) in [(Map::Logistic, -2.94, -0.7), (Map::Reflection, 0.3, -0.5)] {
    let (end, report) = stopped(map, x, target);
    let verdict = (end, report.converged, report.message());
    assert_eq!(verdict, (x, false, "not stationary"), "{map:?}: {report:?}");
  }
}

#[test]
fn a_fit_resting_on_a_bound_with_its_minimum_outside_the_box_is_converged() {
  for map in [Map::Clamp, Map::Reflection] {
    let (end, report) = stopped(map, 0.0, -0.5);
    let verdict = (end, report.converged, report.message());
    assert_eq!(verdict, (0.0, true, "converged"), "{map:?}: {report:?}");
  }
}

/// basin's one step from `x = 14.5` ends past −700, where the logistic slope has underflowed.
#[test]
fn a_jacobian_that_is_flat_at_the_stop_is_a_flat_objective() {
  let (end, report) = stopped(Map::Logistic, 14.5, -0.7);
  assert!(end < -700.0 && !report.converged, "{end}: {report:?}");
  assert_eq!(
    (report.stationarity, report.message()),
    (0.0, "flat objective")
  );
}

#[test]
fn the_verdict_and_the_message_agree_at_every_stop() {
  let cases = [
    (Map::Logistic, -2.94, -0.7),
    (Map::Logistic, 14.5, -0.7),
    (Map::Logistic, 0.0, -0.7),
    (Map::Reflection, 0.3, -0.5),
    (Map::Clamp, 0.0, -0.5),
  ];
  for (map, x, target) in cases {
    let (_, report) = stopped(map, x, target);
    let named = report.message() == "converged";
    assert_eq!(report.converged, named, "{map:?} from {x}: {report:?}");
  }
}
