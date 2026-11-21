use ndarray::Array1;

use super::LevyCalibrator;
use super::calibrator::LevyProblem;
use super::loss::default_params;
use super::loss::param_bounds;
use super::types::LevyModelType;
use super::types::LevyParams;
use crate::OptionQuote;
use crate::OptionType;
use crate::calibration::least_squares::LeastSquaresProblem;
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
    let mut base = default_params(model);
    if model == LevyModelType::Nig {
      base[0] = 50.0;
    }
    let mut problem = LevyProblem::new(&calibrator, base.clone());
    let mut at = |j: usize, raw: f64| {
      let mut x = Array1::from_vec(base.clone());
      x[j] = raw;
      problem.set_params(x.view());
      problem.params()[j]
    };
    for (j, (lo, hi)) in param_bounds(model).into_iter().enumerate() {
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
