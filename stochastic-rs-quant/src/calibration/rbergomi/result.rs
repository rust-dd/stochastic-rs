use super::params::RBergomiParams;

#[derive(Clone, Debug)]
pub struct RBergomiCalibrationResult {
  pub initial_params: RBergomiParams,
  pub calibrated_params: RBergomiParams,
  pub initial_loss: f64,
  pub final_loss: f64,
  pub maturity_losses: Vec<(f64, f64)>,
  pub iterations: usize,
  pub converged: bool,
}

impl RBergomiCalibrationResult {
  /// Convert to an [`RBergomiPricer`] for pricing / vol surface generation.
  pub fn to_model(&self) -> crate::pricing::rbergomi::RBergomiPricer {
    crate::pricing::rbergomi::RBergomiPricer::new(self.calibrated_params.clone())
  }
}

impl crate::traits::ToModel for RBergomiCalibrationResult {
  type Model = crate::pricing::rbergomi::RBergomiPricer;
  fn to_model(&self, _r: f64, _q: f64) -> Self::Model {
    RBergomiCalibrationResult::to_model(self)
  }
}

impl crate::traits::CalibrationResult for RBergomiCalibrationResult {
  type Params = RBergomiParams;
  fn rmse(&self) -> f64 {
    // `final_loss` is a Wasserstein-1 distance averaged over maturities; we
    // expose its square root so the magnitude is comparable to per-quote RMSE
    // values from price-fitting calibrators.
    self.final_loss.abs().sqrt()
  }
  fn converged(&self) -> bool {
    self.converged
  }
  fn params(&self) -> Self::Params {
    self.calibrated_params.clone()
  }
}
