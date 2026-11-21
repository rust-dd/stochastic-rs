use ndarray::Array1;

use super::LevyCalibrator;
use super::calibrator::LevyProblem;
use super::coordinates::coordinate_bounds;
use super::loss::default_params;
use super::loss::fourier_call_price;
use super::loss::param_bounds;
use super::types::LevyModelType;
use super::types::LevyParams;
use crate::OptionQuote;
use crate::OptionType;
use crate::calibration::least_squares::LeastSquaresProblem;
use crate::pricing::cf_quadrature::Walk;
use crate::traits::CalibrationResult;
use crate::traits::Calibrator;

const STRIKES: [f64; 9] = [80.0, 85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0, 120.0];
// Variance Gamma, `σ = 0.2, θ = −0.1, ν = 0.5`, at `S = 100, r = 0.05, q = 0, T = 1`.
const VG_REF: [f64; 9] = [
  25.056158, 20.941767, 17.091301, 13.572373, 10.453503, 7.795810, 5.640544, 3.991334, 2.793823,
];
const MODELS: [LevyModelType; 5] = [
  LevyModelType::VarianceGamma,
  LevyModelType::Nig,
  LevyModelType::Cgmy,
  LevyModelType::MertonJd,
  LevyModelType::Kou,
];

fn quotes() -> Vec<OptionQuote> {
  let rows = STRIKES.iter().zip(VG_REF);
  rows
    .map(|(&k, price)| OptionQuote::new(k, 1.0, price, OptionType::Call))
    .collect()
}

/// Each start's first step leaves the box on every side. A restart from a minimum stays there,
/// so a `converged` fit's own point is not improved tenfold.
#[test]
fn an_interior_start_is_not_converged_where_a_restart_improves_it_tenfold() {
  for (model, values) in [
    (LevyModelType::VarianceGamma, vec![1.2, 0.6, 0.5]),
    (LevyModelType::Cgmy, vec![2.0, 2.0, 2.0, 0.8]),
    (LevyModelType::Kou, vec![0.15, 3.0, 0.5, 1.2, 10.0]),
  ] {
    let from = |values: Vec<f64>| {
      let start = LevyParams {
        values,
        model_type: model,
      };
      let calibrator = LevyCalibrator::new(model, 100.0, 0.05, 0.0, quotes());
      calibrator.calibrate(Some(start)).unwrap()
    };
    let fit = from(values.clone());
    let again = from(fit.params.values.clone());
    assert!(
      !fit.converged || 10.0 * again.rmse() >= fit.rmse(),
      "{model:?} {values:?}: {:e} converged at {:?}, {:e} from it",
      fit.rmse(),
      fit.params.values,
      again.rmse()
    );
  }
}

/// Across each bound of each box, NIG's `β` at `α`'s ceiling included: a raw step that left its
/// coordinate unmoved would move no price.
#[test]
fn no_raw_step_across_a_bound_leaves_its_coordinate_where_it_was() {
  for model in MODELS {
    let calibrator = LevyCalibrator::new(model, 100.0, 0.05, 0.0, quotes());
    let mut problem = LevyProblem::new(&calibrator, default_params(model));
    let base = problem.params().to_vec();
    let mut at = |j: usize, raw: f64| {
      let mut x = Array1::from_vec(base.clone());
      x[j] = raw;
      problem.set_params(x.view());
      problem.params()[j]
    };
    for (j, (lo, hi)) in coordinate_bounds(model).into_iter().enumerate() {
      let width = hi - lo;
      let offsets = (-8..=8).map(|i| 2.5e-3 * width * f64::from(i));
      for raw in offsets.flat_map(|offset| [lo + offset, hi + offset]) {
        let (here, beside) = (at(j, raw), at(j, raw + 1e-4 * width));
        assert!(here != beside, "{model:?} {j} at {raw}: {here}");
        assert!(lo <= here && here <= hi, "{model:?} {j} at {raw}: {here}");
      }
    }
  }
}

/// At `α`'s ceiling the domain leaves `β` its whole box: each step across it moves the model's `β`.
#[test]
fn no_step_inside_its_box_leaves_the_nig_asymmetry_where_it_was() {
  let model = LevyModelType::Nig;
  let calibrator = LevyCalibrator::new(model, 100.0, 0.05, 0.0, quotes());
  let mut problem = LevyProblem::new(&calibrator, default_params(model));
  let mut beta = |raw: f64| {
    problem.set_params(ndarray::array![50.0, raw, 0.5].view());
    problem.params.values[1]
  };
  let (lo, hi) = param_bounds(model)[1];
  for raw in (0..=400).map(|i| lo + (hi - lo) * f64::from(i) / 400.0) {
    let (here, beside) = (beta(raw), beta(raw + 1e-3));
    assert!(here != beside, "{raw}: {here}");
  }
  assert_eq!((lo, hi), (-49.99, 48.99));
}

/// Nine one-year calls of `truth`, rounded to `1e-6` as a quote is: a fit to unrounded prices of
/// its own model ends in the pricer's noise, where a restart's ratio means nothing.
fn rounded_calls(model: LevyModelType, truth: &[f64]) -> Vec<OptionQuote> {
  let call =
    |k: f64| fourier_call_price(model, truth, 100.0, k, 0.05, 0.0, 1.0, &mut Walk::Adaptive);
  let rows = STRIKES.iter();
  rows
    .map(|&k| OptionQuote::new(k, 1.0, (call(k) * 1e6).round() / 1e6, OptionType::Call))
    .collect()
}

/// Each truth lies beside its domain's edge, `α = |β + 1|` and `1 − θν − σ²ν/2 = 0`, and each
/// start's fit runs onto it: a `converged` fit is improved tenfold neither from its own point nor
/// from one a tenth further inside.
#[test]
fn a_fit_on_its_domains_edge_is_not_converged_where_the_cost_falls_inside() {
  let nig_start = vec![47.68619895060727, 5.121723141418705, 2.8522282392243885];
  for (model, truth, start, coupled) in [
    (LevyModelType::Nig, vec![2.6, 1.5, 0.3], nig_start, (0, 1.1)),
    (
      LevyModelType::VarianceGamma,
      vec![0.5, 0.6, 1.36],
      vec![1.385, -0.496, 1.144],
      (2, 0.9),
    ),
  ] {
    let from = |values: Vec<f64>| {
      let start = LevyParams {
        values,
        model_type: model,
      };
      let calibrator = LevyCalibrator::new(model, 100.0, 0.05, 0.0, rounded_calls(model, &truth));
      calibrator.calibrate(Some(start)).unwrap()
    };
    let fit = from(start.clone());
    let mut inside = fit.params.values.clone();
    inside[coupled.0] *= coupled.1;
    let lowest = from(fit.params.values.clone())
      .rmse()
      .min(from(inside).rmse());
    assert!(
      !fit.converged || 10.0 * lowest >= fit.rmse(),
      "{model:?}: {:e} converged at {:?}, {lowest:e} from it",
      fit.rmse(),
      fit.params.values
    );
  }
}

/// Where its coupled condition binds, `α`'s floor at `β = 20` and `ν`'s cap at `θ + σ²/2 = 1`:
/// each step of a coordinate across its box moves the model, so its column is never zero there.
#[test]
fn no_step_across_its_box_leaves_a_coupled_parameter_where_it_was() {
  for (model, at, j) in [
    (LevyModelType::Nig, [0.0, 20.0, 0.5], 0),
    (LevyModelType::VarianceGamma, [1.0, 0.5, 0.0], 2),
  ] {
    let calibrator = LevyCalibrator::new(model, 100.0, 0.05, 0.0, quotes());
    let mut problem = LevyProblem::new(&calibrator, default_params(model));
    let (lo, hi) = coordinate_bounds(model)[j];
    let values = (0..=200).map(|i| {
      let mut x = Array1::from_vec(at.to_vec());
      x[j] = lo + (hi - lo) * f64::from(i) / 200.0;
      problem.set_params(x.view());
      problem.params.values[j]
    });
    let values = values.collect::<Vec<_>>();
    assert!(
      values.windows(2).all(|pair| pair[0] < pair[1]),
      "{model:?}: {values:?}"
    );
  }
}

/// A pull acts on the parameters: the tied one's row runs through its share and the edge that
/// `σ` and `θ`, or `β` on either side of −½, move; a reflected coordinate's slope is −1.
#[test]
fn the_natural_jacobian_matches_central_differences() {
  use crate::calibration::least_squares::checks::natural_jacobian_by_differences;
  use crate::calibration::least_squares::checks::worst_relative_error;
  for (model, x) in [
    (LevyModelType::VarianceGamma, ndarray::array![1.0, 0.5, 0.7]),
    (
      LevyModelType::VarianceGamma,
      ndarray::array![0.2, -0.1, 1.3],
    ),
    (LevyModelType::VarianceGamma, ndarray::array![2.4, 0.5, 0.7]),
    (LevyModelType::Nig, ndarray::array![0.3, 20.0, 0.5]),
    (LevyModelType::Nig, ndarray::array![1.3, -7.0, 0.5]),
    (
      LevyModelType::Kou,
      ndarray::array![0.15, 3.0, 1.2, 10.0, 10.0],
    ),
  ] {
    let calibrator = LevyCalibrator::new(model, 100.0, 0.05, 0.0, quotes());
    let mut problem = LevyProblem::new(&calibrator, default_params(model));
    problem.set_params(x.view());
    let analytic = problem.natural_jacobian();
    let numeric = natural_jacobian_by_differences(&mut problem, x.view(), 1e-6);
    let worst = worst_relative_error(analytic.view(), numeric.view());
    assert!(worst < 1e-6, "{model:?} {x}: {worst:e}");
  }
}
