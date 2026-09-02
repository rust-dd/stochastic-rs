use levenberg_marquardt::LeastSquaresProblem;
use nalgebra::DMatrix;
use nalgebra::DVector;
use nalgebra::Dyn;
use nalgebra::Owned;

use super::calibrator::HestonCalibrator;
use super::params::HestonJacobianMethod;
use super::params::HestonParams;
use super::transform::apply_chain_rule;
use super::transform::from_optimizer_coordinates;
use super::transform::to_optimizer_coordinates;
use crate::CalibrationLossScore;
use crate::calibration::CalibrationHistory;
use crate::pricing::heston::HestonPricer;

impl LeastSquaresProblem<f64, Dyn, Dyn> for HestonCalibrator {
  type JacobianStorage = Owned<f64, Dyn, Dyn>;
  type ParameterStorage = Owned<f64, Dyn>;
  type ResidualStorage = Owned<f64, Dyn>;

  fn set_params(&mut self, params: &DVector<f64>) {
    self.params = Some(from_optimizer_coordinates(params));
  }

  fn params(&self) -> DVector<f64> {
    to_optimizer_coordinates(&self.effective_params())
  }

  fn residuals(&self) -> Option<DVector<f64>> {
    let params_eff = self.effective_params();
    let c_model = self.compute_model_prices_for(&params_eff);
    let weighted_residuals =
      (self.c_market.clone() - c_model.clone()).component_mul(&self.residual_weights);

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
            self.c_market.as_slice(),
            c_model.as_slice(),
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

  fn jacobian(&self) -> Option<DMatrix<f64>> {
    let p = self.effective_params();
    let optimizer_coordinates = to_optimizer_coordinates(&p);
    let jacobian = match self.jacobian_method {
      HestonJacobianMethod::NumericFiniteDiff => self.numeric_optimizer_jacobian(&p),
      HestonJacobianMethod::CuiAnalytic => {
        match self.compute_model_prices_and_residual_jacobian_cui(&p) {
          Some((_, jac)) => apply_chain_rule(jac, &optimizer_coordinates),
          None => self.numeric_optimizer_jacobian(&p),
        }
      }
    };
    Some(match &self.regularization {
      // The penalty rows are diagonal in the physical parameters; the same
      // chain rule as the price rows maps them into the optimiser coordinates.
      Some(reg) if reg.is_active() => reg.augment_jacobian(
        jacobian,
        apply_chain_rule(reg.jacobian_rows(), &optimizer_coordinates),
      ),
      _ => jacobian,
    })
  }
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
