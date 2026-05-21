use levenberg_marquardt::LeastSquaresProblem;
use nalgebra::DMatrix;
use nalgebra::DVector;
use nalgebra::Dyn;
use nalgebra::Owned;

use super::calibrator::HestonCalibrator;
use super::params::{HestonJacobianMethod, HestonParams};
use crate::CalibrationLossScore;
use crate::calibration::CalibrationHistory;
use crate::pricing::heston::HestonPricer;
use crate::traits::PricerExt;

impl LeastSquaresProblem<f64, Dyn, Dyn> for HestonCalibrator {
  type JacobianStorage = Owned<f64, Dyn, Dyn>;
  type ParameterStorage = Owned<f64, Dyn>;
  type ResidualStorage = Owned<f64, Dyn>;

  fn set_params(&mut self, params: &DVector<f64>) {
    let p = HestonParams::from(params.clone()).projected();
    self.params = Some(p);
  }

  fn params(&self) -> DVector<f64> {
    self.effective_params().into()
  }

  fn residuals(&self) -> Option<DVector<f64>> {
    let params_eff = self.effective_params();
    let c_model = self.compute_model_prices_for(&params_eff);

    if self.record_history {
      self
        .calibration_history
        .borrow_mut()
        .push(CalibrationHistory {
          residuals: self.c_market.clone() - c_model.clone(),
          call_put: self
            .c_market
            .iter()
            .enumerate()
            .map(|(i, _)| {
              let pricer = HestonPricer::new(
                self.s[i],
                params_eff.v0,
                self.k[i],
                self.r,
                self.q,
                params_eff.rho,
                params_eff.kappa,
                params_eff.theta,
                params_eff.sigma,
                Some(0.0),
                Some(self.flat_t[i]),
                None,
                None,
              );
              pricer.calculate_call_put()
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

    Some(self.c_market.clone() - c_model)
  }

  fn jacobian(&self) -> Option<DMatrix<f64>> {
    let p = self.effective_params();
    match self.jacobian_method {
      HestonJacobianMethod::NumericFiniteDiff => Some(self.numeric_jacobian(&p)),
      HestonJacobianMethod::CuiAnalytic => {
        if let Some((_, jac)) = self.compute_model_prices_and_residual_jacobian_cui(&p) {
          Some(jac)
        } else {
          Some(self.numeric_jacobian(&p))
        }
      }
    }
  }
}
