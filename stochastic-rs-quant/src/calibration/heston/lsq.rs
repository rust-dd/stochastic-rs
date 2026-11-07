use ndarray::Array1;
use ndarray::Array2;

use super::calibrator::HestonCalibrator;
use super::params::HestonJacobianMethod;
use super::params::HestonParams;
use super::params::feller_sigma_cap;
use super::transform::apply_chain_rule;
use super::transform::from_optimizer_coordinates;
use super::transform::to_optimizer_coordinates;
use crate::CalibrationLossScore;
use crate::calibration::CalibrationHistory;
use crate::calibration::least_squares::LeastSquaresProblem;
use crate::pricing::heston::HestonPricer;

impl LeastSquaresProblem for HestonCalibrator {
  fn set_params(&mut self, params: &Array1<f64>) {
    self.params = Some(from_optimizer_coordinates(params));
  }

  fn params(&self) -> Array1<f64> {
    to_optimizer_coordinates(&self.effective_params())
  }

  fn residuals(&self) -> Option<Array1<f64>> {
    let free = self.free_params();
    let params_eff = self.feller_projected(free.clone());
    let c_model = self.compute_model_prices_for(&params_eff);
    let weighted_residuals = (&self.c_market - &c_model) * &self.residual_weights;

    if self.record_history {
      self
        .calibration_history
        .borrow_mut()
        .push(CalibrationHistory {
          residuals: weighted_residuals.clone(),
          call_put: self
            .c_market
            .iter()
            .enumerate()
            .map(|(i, _)| {
              let pricer = HestonPricer::new(
                params_eff.v0,
                params_eff.rho,
                params_eff.kappa,
                params_eff.theta,
                params_eff.sigma,
                Some(0.0),
              );
              pricer.call_put(
                self.s[i],
                self.k[i],
                self.r,
                self.q.unwrap_or(0.0),
                self.flat_t[i],
              )
            })
            .collect::<Vec<(f64, f64)>>()
            .into(),
          params: params_eff.clone(),
          loss_scores: CalibrationLossScore::compute_selected(
            self.c_market.as_standard_layout().as_slice().unwrap(),
            c_model.as_slice().unwrap(),
            self.loss_metrics,
          ),
        });
    }

    let residuals = match &self.regularization {
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
    let free = self.free_params();
    let p = self.feller_projected(free.clone());
    let optimizer_coordinates = to_optimizer_coordinates(&free);
    let capped = self.feller_projection && p.sigma < free.sigma;
    let to_optimizer = |physical: Array2<f64>| {
      let physical = if capped {
        through_feller_cap(physical, &free)
      } else {
        physical
      };
      apply_chain_rule(physical, &optimizer_coordinates)
    };
    let jacobian = match self.jacobian_method {
      HestonJacobianMethod::NumericFiniteDiff => self.numeric_optimizer_jacobian(&free),
      HestonJacobianMethod::CuiAnalytic => {
        match self.compute_model_prices_and_residual_jacobian_cui(&p) {
          Some((_, jac)) => to_optimizer(jac),
          None => self.numeric_optimizer_jacobian(&free),
        }
      }
    };
    let jacobian = match &self.regularization {
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
