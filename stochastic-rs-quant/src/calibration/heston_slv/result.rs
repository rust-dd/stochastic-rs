use super::calibrator::RATE_MATCH_TOL;
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

/// Calibration result for the Heston SLV model.
///
/// The leverage is anchored to `(r, q)` — the cloud it was read off moved
/// under the drift $r - q$ — so the result records the pair and
/// [`to_model`](Self::to_model) builds the pricer at it.
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

  /// The Monte Carlo pricer of the calibrated model, anchored to the
  /// calibration rates.
  ///
  /// # Panics
  ///
  /// When `(r, q)` is not the pair the leverage was calibrated at: the
  /// pricer would reject every query at those rates anyway, and a surface
  /// re-anchored to rates it was not fitted under would silently stop
  /// reproducing the market. Recalibrate at the new rates instead.
  pub fn to_model(&self, r: f64, q: f64) -> HestonSlvPricer {
    assert!(
      (r - self.r).abs() <= RATE_MATCH_TOL && (q - self.q).abs() <= RATE_MATCH_TOL,
      "HestonSlvCalibrationResult::to_model: the leverage surface was calibrated at r={}, q={} \
       but a pricer at r={r}, q={q} was asked for; L(S,t) is rate-dependent, so recalibrate at the \
       new rates",
      self.r,
      self.q
    );
    HestonSlvPricer::new(self.fit.params, self.fit.leverage.clone(), self.r, self.q)
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

  fn max_error(&self) -> f64 {
    self.max_error
  }
}

impl crate::traits::ToModel for HestonSlvCalibrationResult {
  type Model = HestonSlvPricer;

  fn to_model(&self, r: f64, q: f64) -> Self::Model {
    HestonSlvCalibrationResult::to_model(self, r, q)
  }
}
