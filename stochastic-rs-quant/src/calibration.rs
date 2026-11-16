//! # Calibration
//!
//! $$
//! \hat\theta=\arg\min_\theta\sum_i w_i\left(P_i^{model}(\theta)-P_i^{mkt}\right)^2
//! $$
//!
use std::convert::Infallible;
use std::sync::OnceLock;

use basin::BasicSimplexState;
use basin::CostFunction;
use basin::Executor;
use basin::NelderMead;
use basin::SimplexState;
use basin::State;
use basin::StepOutcome;
use basin::TerminationReason;
use gauss_quad::GaussLegendre;
use ndarray::Array1;
use stochastic_rs_distributions::RealExt;

use crate::CalibrationLossScore;

/// Preserves the existing sample-standard-deviation stopping rule for simplex fits.
pub(crate) struct SimplexStandardDeviation {
  tolerance: f64,
}

impl SimplexStandardDeviation {
  pub(crate) fn new(tolerance: f64) -> Option<Self> {
    (tolerance.is_finite() && tolerance >= 0.0).then_some(Self { tolerance })
  }
}

impl SimplexStandardDeviation {
  fn check<S: SimplexState<Float = f64>>(&self, state: &S) -> Option<TerminationReason> {
    match sample_standard_deviation(state.costs()) {
      Some(sd) => (sd < self.tolerance).then_some(TerminationReason::SimplexTolerance),
      // Undefined spread must not silently disable the stopping rule.
      None => Some(TerminationReason::SolverFailed),
    }
  }
}

fn sample_standard_deviation(values: &[f64]) -> Option<f64> {
  if values.len() < 2 || values.iter().any(|value| !value.is_finite()) {
    return None;
  }
  // Scaling keeps finite costs from overflowing the mean or squared deviations.
  let scale = f64::max_or_nan_fold(values.iter().map(|value| value.abs()));
  if scale == 0.0 {
    return Some(0.0);
  }
  let mean = values.iter().map(|value| value / scale).sum::<f64>() / values.len() as f64;
  let squared_deviations = values
    .iter()
    .map(|value| (value / scale - mean).powi(2))
    .sum::<f64>();
  let sd = scale * (squared_deviations / (values.len() - 1) as f64).sqrt();
  sd.is_finite().then_some(sd)
}

pub(crate) struct SimplexRun {
  pub(crate) best: Vec<f64>,
  /// The simplex spread fell below its tolerance.
  pub(crate) converged: bool,
  pub(crate) iterations: u64,
  /// The best vertex at the start and after each iteration; empty unless recorded.
  pub(crate) trail: Vec<Vec<f64>>,
}

pub(crate) fn run_nelder_mead<P>(
  problem: P,
  simplex: Vec<Vec<f64>>,
  max_iters: u64,
  sd_tolerance: f64,
  record: bool,
) -> SimplexRun
where
  P: CostFunction<Param = Vec<f64>, Output = f64, Error = Infallible>,
{
  assert!(
    simplex.len() >= 2,
    "simplex must contain at least two vertices"
  );
  let Some(criterion) = SimplexStandardDeviation::new(sd_tolerance) else {
    let best = simplex[0].clone();
    let trail = if record {
      vec![best.clone()]
    } else {
      Vec::new()
    };
    return SimplexRun {
      best,
      converged: false,
      iterations: 0,
      trail,
    };
  };
  let state = BasicSimplexState::from_simplex(simplex);
  let mut stepper = Executor::new(problem, NelderMead::new(), state)
    .max_iter(max_iters)
    .stop_when(move |state| criterion.check(state))
    .into_stepper()
    .expect("calibration objective is infallible");
  let mut trail = Vec::new();
  let reason = loop {
    if record {
      trail.push(stepper.state().best_param().clone());
    }
    if let StepOutcome::Stopped(reason) =
      stepper.step().expect("calibration objective is infallible")
    {
      break reason;
    }
  };
  let state = stepper.state();
  SimplexRun {
    best: state.best_param().clone(),
    converged: reason == TerminationReason::SimplexTolerance,
    iterations: state.iter(),
    trail,
  }
}

/// A Nelder–Mead fit as a [`Calibrated`]; `price` maps a vertex to its parameters and its model
/// and market prices, for the best vertex and, when history is recorded, each one in the trail.
pub(crate) fn nelder_mead_fit<C, P>(
  cost: C,
  simplex: Vec<Vec<f64>>,
  options: &CalibrationOptions,
  default_max_iterations: usize,
  price: impl Fn(&[f64]) -> (P, Vec<f64>, Vec<f64>),
) -> Calibrated<P>
where
  C: CostFunction<Param = Vec<f64>, Output = f64, Error = Infallible>,
{
  let run = run_nelder_mead(
    cost,
    simplex,
    options.max_iterations.unwrap_or(default_max_iterations) as u64,
    options.tolerance.unwrap_or(1e-10),
    options.record_history,
  );
  let history = run
    .trail
    .iter()
    .enumerate()
    .map(|(iteration, vertex)| {
      let (params, market, model) = price(vertex);
      let residuals = Array1::from_iter(market.iter().zip(&model).map(|(q, m)| q - m));
      let loss = CalibrationLossScore::compute_selected(&market, &model, options.loss_metrics);
      CalibrationHistory::new(iteration, params, residuals, loss)
    })
    .collect();
  let (params, market, model) = price(&run.best);
  Calibrated::repriced(params, &market, &model, options.loss_metrics, run.converged)
    .with_iterations(run.iterations as usize)
    .with_history(history)
}

pub mod bsm;
mod calibrated;
pub mod cgmysv;
pub mod double_heston;
pub mod heston;
pub mod heston_slv;
pub mod heston_stoch_corr;
pub mod hkde;
pub mod hw_swaption;
#[doc(hidden)]
pub mod least_squares;
pub mod levy;
pub(crate) mod quote;
pub mod rbergomi;
pub mod regularization;
pub mod sabr;
pub mod sabr_caplet;
pub mod svj;
pub mod tree_swaption;

#[cfg(test)]
mod quadrature_tests;

pub use bsm::BsmCalibrationResult;
pub use bsm::BsmCalibrator;
pub use bsm::BsmParams;
pub use calibrated::Calibrated;
pub use calibrated::CalibrationHistory;
pub use calibrated::CalibrationOptions;
pub use cgmysv::CgmysvCalibrationResult;
pub use cgmysv::CgmysvCalibrator;
pub use double_heston::DoubleHestonCalibrationResult;
pub use double_heston::DoubleHestonCalibrator;
pub use double_heston::DoubleHestonParams;
pub use heston::HestonCalibrationResult;
pub use heston::HestonCalibrator;
pub use heston::HestonParams;
pub use heston_stoch_corr::HscmCalibrationResult;
pub use heston_stoch_corr::HscmParams;
pub use heston_stoch_corr::MarketOption;
pub use heston_stoch_corr::calibrate_hscm;
pub use hkde::HkdeCalibrationResult;
pub use hkde::HkdeCalibrator;
pub use hkde::HkdeParams;
pub use hw_swaption::HullWhiteParams;
pub use hw_swaption::HullWhiteSwaptionCalibrator;
pub use hw_swaption::SwaptionQuote;
pub use levy::LevyCalibrationResult;
pub use levy::LevyCalibrator;
pub use levy::LevyModelType;
pub use levy::LevyParams;
pub use levy::MarketSlice;
pub use regularization::Regularization;
pub use sabr::SabrCalibrator;
pub use sabr_caplet::SabrCapletCalibrator;
pub use sabr_caplet::SabrCapletParams;
pub use svj::SvjCalibrationResult;
pub use svj::SvjCalibrator;
pub use svj::SvjParams;
pub use tree_swaption::BlackKarasinskiParams;
pub use tree_swaption::BlackKarasinskiSwaptionCalibrator;
pub use tree_swaption::G2ppParams;
pub use tree_swaption::G2ppSwaptionCalibrator;

const GL_PANEL_WIDTH: f64 = 50.0;
const GL_MAX_PANELS: usize = 256;

fn gauss_legendre_64() -> &'static GaussLegendre {
  static GL64: OnceLock<GaussLegendre> = OnceLock::new();
  GL64.get_or_init(|| GaussLegendre::new(64.try_into().unwrap()))
}

fn compensated_add<T: RealExt>(sum: &mut T, correction: &mut T, value: T) {
  let adjusted = value - *correction;
  let next = *sum + adjusted;
  *correction = (next - *sum) - adjusted;
  *sum = next;
}

/// Integrate coupled characteristic-function terms over `[0, ∞)`.
///
/// Every fixed-width panel receives a fresh 64-point Gauss-Legendre rule, so
/// extending the effective upper bound also increases the node count without
/// reducing node density. Two consecutive panels must be negligible in every
/// component; this keeps price and gradient integrals on the same converged
/// domain and avoids stopping on a single oscillatory cancellation.
pub(crate) fn integrate_gl_to_convergence<T: RealExt, const N: usize, F>(
  integrand: F,
  tol: T,
) -> Option<[T; N]>
where
  F: Fn(f64) -> Option<[T; N]>,
{
  debug_assert!(N > 0);
  debug_assert!(tol.is_finite() && tol > T::zero());

  let quadrature = gauss_legendre_64();
  let half_width = 0.5 * GL_PANEL_WIDTH;
  let mut total = [T::zero(); N];
  let mut total_correction = [T::zero(); N];
  let mut negligible_streak = 0usize;

  for panel_index in 0..GL_MAX_PANELS {
    let midpoint = (panel_index as f64 + 0.5) * GL_PANEL_WIDTH;
    let mut panel = [T::zero(); N];
    let mut panel_correction = [T::zero(); N];

    for (node, weight) in quadrature.nodes().zip(quadrature.weights()) {
      let values = integrand(midpoint + half_width * *node)?;
      if values.iter().any(|value| !value.is_finite()) {
        return None;
      }

      for component in 0..N {
        compensated_add(
          &mut panel[component],
          &mut panel_correction[component],
          T::from_f64_fast(half_width * *weight) * values[component],
        );
      }
    }

    for component in 0..N {
      compensated_add(
        &mut total[component],
        &mut total_correction[component],
        panel[component],
      );
    }

    let negligible =
      (0..N).all(|component| panel[component].abs() <= tol * total[component].abs().max(T::one()));
    if negligible {
      negligible_streak += 1;
      if negligible_streak == 2 {
        return Some(total);
      }
    } else {
      negligible_streak = 0;
    }
  }

  None
}

/// Periodic linear extension mapping `x` into `[c, d]`.
///
/// Used to keep optimiser parameters in range without hard clipping.
pub(crate) fn periodic_map(x: f64, c: f64, d: f64) -> f64 {
  if c <= x && x <= d {
    x
  } else {
    let range = d - c;
    if range <= 0.0 {
      return c;
    }
    let n = ((x - c) / range).floor();
    let n_int = n as i64;
    if n_int % 2 == 0 {
      x - n * range
    } else {
      d + n * range - (x - c)
    }
  }
}

#[cfg(test)]
mod optimizer_tests {
  use std::convert::Infallible;

  use basin::CostFunction;

  use super::run_nelder_mead;
  use super::sample_standard_deviation;

  struct Sphere;

  impl CostFunction for Sphere {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = Infallible;

    fn cost(&self, x: &Self::Param) -> Result<Self::Output, Self::Error> {
      Ok(x.iter().map(|value| value.powi(2)).sum())
    }
  }

  #[test]
  fn simplex_spread_uses_sample_standard_deviation() {
    assert_eq!(sample_standard_deviation(&[1.0, 2.0, 3.0]), Some(1.0));
    assert_eq!(sample_standard_deviation(&[0.0, 0.0]), Some(0.0));
    assert_eq!(sample_standard_deviation(&[f64::MAX, f64::MAX]), Some(0.0));
  }

  #[test]
  fn undefined_simplex_spread_is_rejected() {
    for costs in [&[][..], &[1.0], &[0.0, f64::INFINITY], &[0.0, f64::NAN]] {
      assert_eq!(sample_standard_deviation(costs), None);
    }
  }

  #[test]
  #[should_panic(expected = "simplex must contain at least two vertices")]
  fn empty_simplex_has_an_explicit_precondition() {
    run_nelder_mead(Sphere, vec![], 100, 1e-10, false);
  }

  #[test]
  fn invalid_simplex_tolerance_returns_the_initial_point() {
    let initial = vec![2.0];
    for tolerance in [-1.0, f64::NAN, f64::INFINITY] {
      let simplex = vec![initial.clone(), vec![3.0]];
      let run = run_nelder_mead(Sphere, simplex, 100, tolerance, true);
      assert_eq!(run.best, initial);
      assert!(!run.converged);
      assert_eq!((run.iterations, run.trail), (0, vec![initial.clone()]));
    }
  }

  #[test]
  fn a_converged_run_reports_the_iterations_it_took() {
    let simplex = vec![vec![2.0], vec![3.0]];

    let run = run_nelder_mead(Sphere, simplex.clone(), 400, 1e-10, true);
    let capped = run_nelder_mead(Sphere, simplex, run.iterations - 1, 1e-10, false);

    assert!(run.converged && run.best[0].abs() < 1e-4 && run.iterations > 1);
    assert_eq!(capped.iterations, run.iterations - 1);
    assert!(capped.trail.is_empty());
    assert_eq!(run.trail.len() as u64, run.iterations + 1);
    assert_eq!(run.trail.last(), Some(&run.best));
  }

  #[test]
  fn iteration_limit_is_not_reported_as_convergence() {
    let simplex = vec![vec![2.0], vec![3.0]];

    let run = run_nelder_mead(Sphere, simplex, 0, 1e-10, false);

    assert!(!run.converged);
    assert_eq!(run.iterations, 0);
  }

  #[test]
  fn nonfinite_simplex_costs_stop_without_convergence() {
    struct Nonfinite;

    impl CostFunction for Nonfinite {
      type Param = Vec<f64>;
      type Output = f64;
      type Error = Infallible;

      fn cost(&self, x: &Self::Param) -> Result<f64, Infallible> {
        Ok(if x[0] > 0.0 { f64::INFINITY } else { 0.0 })
      }
    }

    let state = basin::BasicSimplexState::from_simplex(vec![vec![0.0], vec![1.0]]);
    let result = basin::Executor::new(Nonfinite, basin::NelderMead::new(), state)
      .max_iter(100)
      .stop_when(|state| {
        super::SimplexStandardDeviation::new(1e-10)
          .unwrap()
          .check(state)
      })
      .run()
      .unwrap();
    assert_eq!(result.reason, basin::TerminationReason::SolverFailed);
    assert_eq!(result.iter(), 0);
    assert_eq!(result.best_param(), &vec![0.0]);
  }
}
