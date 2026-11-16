use crate::CalibrationLossScore;
use crate::LossMetric;
use crate::calibration::heston::HestonCalibrationResult;
use crate::pricing::slv::HestonSlvParams;
use crate::pricing::slv::HestonSlvPricer;
use crate::pricing::slv::LeverageSurface;

/// The calibrated object: the Heston-SLV parameters and the leverage surface
/// fitted to the vanilla surface. Both are needed to specify the model — the
/// parameters alone are a Heston model with a mixing fraction, and the
/// leverage is what makes it reprice the market — which is why the
/// [`CalibrationResult::Params`](crate::traits::CalibrationResult::Params)
/// of an SLV calibration carries the two together.
#[derive(Clone, Debug)]
pub struct HestonSlvFit {
  pub params: HestonSlvParams,
  pub leverage: LeverageSurface,
}

/// Calibration result for the Heston SLV model; the leverage is anchored to the `(r, q)` its cloud
/// moved under, so [`ToModel::to_model`](crate::traits::ToModel::to_model) builds the pricer there.
#[derive(Clone, Debug)]
pub struct HestonSlvCalibrationResult {
  /// The parameters and the leverage surface.
  pub fit: HestonSlvFit,
  /// The Heston fit that supplied the parameters, `None` when they were
  /// pinned on the calibrator.
  pub heston: Option<HestonCalibrationResult>,
  /// Repricing of the input calls by the calibration cloud, every metric of
  /// [`LossMetric::ALL`].
  pub loss: CalibrationLossScore,
  /// The worst absolute repricing error over the input grid.
  pub max_error: f64,
  /// The Heston fit converged (or was pinned) and the leverage is finite
  /// everywhere.
  pub converged: bool,
  /// The spot the cloud started from.
  pub s: f64,
  /// The rates the leverage is anchored to.
  pub r: f64,
  pub q: f64,
}

impl HestonSlvCalibrationResult {
  /// The Heston-SLV parameters, without the leverage.
  pub fn slv_params(&self) -> HestonSlvParams {
    self.fit.params
  }

  /// The calibrated leverage surface.
  pub fn leverage(&self) -> &LeverageSurface {
    &self.fit.leverage
  }
}

impl crate::traits::CalibrationResult for HestonSlvCalibrationResult {
  type Params = HestonSlvFit;

  fn rmse(&self) -> f64 {
    self.loss.get(LossMetric::Rmse)
  }

  fn converged(&self) -> bool {
    self.converged
  }

  fn params(&self) -> Self::Params {
    self.fit.clone()
  }

  fn loss_score(&self) -> Option<&CalibrationLossScore> {
    Some(&self.loss)
  }

  fn max_error(&self) -> Option<f64> {
    Some(self.max_error)
  }
}

impl crate::traits::ToModel for HestonSlvCalibrationResult {
  type Model = HestonSlvPricer;

  /// The Monte Carlo pricer of the calibrated model, at the calibration rates.
  fn to_model(&self) -> Self::Model {
    HestonSlvPricer::new(self.fit.params, self.fit.leverage.clone(), self.r, self.q)
  }
}
