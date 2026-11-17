use ndarray::Array1;
use ndarray::Array2;

use super::calibrator::HestonCalibrator;
use super::params::HestonJacobianMethod;
use super::params::feller_sigma_cap;
use super::transform::apply_chain_rule;
use super::transform::canonicalize;
use super::transform::from_optimizer_coordinates;
use super::transform::to_optimizer_coordinates;
use crate::calibration::least_squares::LeastSquaresProblem;
use crate::calibration::least_squares::Trail;
use crate::pricing::heston::HestonParams;

/// One fit: the optimiser's point, the Feller row's weight, the quotes' prices and the trail.
pub(super) struct HestonProblem<'a> {
  calibrator: &'a HestonCalibrator,
  point: HestonParams,
  /// Weight of the row on σ's overshoot past the Feller cap; set only inside a projected fit.
  pub(super) feller_row_weight: Option<f64>,
  pub(super) market: Vec<f64>,
  pub(super) trail: Trail<HestonParams>,
}

impl<'a> HestonProblem<'a> {
  pub(super) fn new(calibrator: &'a HestonCalibrator, start: HestonParams) -> Self {
    Self {
      calibrator,
      point: start,
      feller_row_weight: None,
      market: calibrator.quotes.iter().map(|quote| quote.price).collect(),
      trail: Trail::new(),
    }
  }

  /// The canonical point the optimiser's coordinates stand for, before the Feller projection.
  pub(super) fn free_params(&self) -> HestonParams {
    canonicalize(&self.point.projected())
  }

  /// The point every residual is evaluated at; projected last, so no rounding of the coordinates
  /// undoes Feller.
  pub(super) fn effective_params(&self) -> HestonParams {
    self.calibrator.feller_projected(self.free_params())
  }
}

impl LeastSquaresProblem for HestonProblem<'_> {
  fn set_params(&mut self, params: &Array1<f64>) {
    self.point = from_optimizer_coordinates(params);
  }

  fn params(&self) -> Array1<f64> {
    to_optimizer_coordinates(&self.effective_params())
  }

  fn residuals(&mut self) -> Option<Array1<f64>> {
    let calibrator = self.calibrator;
    let free = self.free_params();
    let params_eff = calibrator.feller_projected(free);
    let c_model = calibrator.compute_model_prices_for(&params_eff);
    let weighted_residuals = calibrator.weighted_residuals(&c_model);
    let model = c_model.as_slice().expect("model prices are a fresh array");
    let options = &calibrator.options;
    self.trail.stage(
      options,
      &self.market,
      model,
      &params_eff,
      &weighted_residuals,
    );

    let residuals = match &options.regularization {
      Some(reg) if reg.is_active() => {
        reg.augment_residuals(weighted_residuals, &physical(&params_eff))
      }
      _ => weighted_residuals,
    };
    Some(match self.feller_row_weight {
      // Zero wherever the iterate is feasible; past the cap it gives the optimiser a gradient back.
      Some(weight) => Array1::from_iter(
        residuals
          .iter()
          .copied()
          .chain(std::iter::once(weight * (free.sigma - params_eff.sigma))),
      ),
      None => residuals,
    })
  }

  fn jacobian(&self) -> Option<Array2<f64>> {
    let calibrator = self.calibrator;
    let free = self.free_params();
    let p = calibrator.feller_projected(free);
    let optimizer_coordinates = to_optimizer_coordinates(&free);
    let capped = calibrator.feller_projection && p.sigma < free.sigma;
    let to_optimizer = |physical: Array2<f64>| {
      let physical = if capped {
        through_feller_cap(physical, &free)
      } else {
        physical
      };
      apply_chain_rule(physical, &optimizer_coordinates)
    };
    let jacobian = match calibrator.jacobian_method {
      HestonJacobianMethod::NumericFiniteDiff => calibrator.numeric_optimizer_jacobian(&free),
      HestonJacobianMethod::CuiAnalytic => {
        match calibrator.compute_model_prices_and_residual_jacobian_cui(&p) {
          Some((_, jac)) => to_optimizer(jac),
          None => calibrator.numeric_optimizer_jacobian(&free),
        }
      }
    };
    let jacobian = match &calibrator.options.regularization {
      // The penalty rows are diagonal in the physical parameters; the same
      // chain rule as the price rows maps them into the optimiser coordinates.
      Some(reg) if reg.is_active() => {
        reg.augment_jacobian(jacobian, to_optimizer(reg.jacobian_rows()))
      }
      _ => jacobian,
    };
    Some(match self.feller_row_weight {
      // w·σ(x) differentiates in the free coordinates, −w·σ(P(x)) through the projection as the price rows do.
      Some(weight) => {
        let overshoot = apply_chain_rule(sigma_row(weight), &optimizer_coordinates)
          + to_optimizer(sigma_row(-weight));
        Array2::from_shape_fn((jacobian.nrows() + 1, jacobian.ncols()), |(i, j)| {
          if i < jacobian.nrows() {
            jacobian[(i, j)]
          } else {
            overshoot[(0, j)]
          }
        })
      }
      None => jacobian,
    })
  }

  fn accept(&mut self) {
    self.trail.accept();
  }
}

fn sigma_row(weight: f64) -> Array2<f64> {
  Array2::from_shape_fn((1, 5), |(_, j)| if j == 3 { weight } else { 0.0 })
}

/// Where the Feller projection caps σ at √(2κθ), σ's column moves into κ's and θ's by ∂σ/∂κ = σ/(2κ) and
/// ∂σ/∂θ = σ/(2θ), and the coordinate σ no longer moves the residuals.
fn through_feller_cap(mut jacobian: Array2<f64>, free: &HestonParams) -> Array2<f64> {
  let cap = feller_sigma_cap(free.kappa, free.theta);
  let sigma_column = jacobian.column(3).to_owned();
  jacobian
    .column_mut(1)
    .scaled_add(cap / (2.0 * free.kappa), &sigma_column);
  jacobian
    .column_mut(2)
    .scaled_add(cap / (2.0 * free.theta), &sigma_column);
  jacobian.column_mut(3).fill(0.0);
  jacobian
}

/// Physical parameters in the regularisation order `(v0, κ, θ, σ, ρ)`.
fn physical(params: &HestonParams) -> [f64; 5] {
  [
    params.v0,
    params.kappa,
    params.theta,
    params.sigma,
    params.rho,
  ]
}
