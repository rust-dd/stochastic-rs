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
    let params_eff = self.effective_params();
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

    match &self.regularization {
      Some(reg) if reg.is_active() => {
        Some(reg.augment_residuals(weighted_residuals, &physical(&params_eff)))
      }
      _ => Some(weighted_residuals),
    }
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
    Some(match &self.regularization {
      // The penalty rows are diagonal in the physical parameters; the same
      // chain rule as the price rows maps them into the optimiser coordinates.
      Some(reg) if reg.is_active() => {
        reg.augment_jacobian(jacobian, to_optimizer(reg.jacobian_rows()))
      }
      _ => jacobian,
    })
  }
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
