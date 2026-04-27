//! Calibration traits — `Calibrator`, `CalibrationResult`, `ToModel`,
//! `ToShortRateModel`.

use super::pricing::ModelPricer;
use crate::CalibrationLossScore;

/// Trait for calibration results that can produce a [`ModelPricer`].
///
/// Every calibration result implementing this trait can be fed directly into
/// the vol-surface pipeline via [`build_surface_from_calibration`]. Uses an
/// associated type so calling code stays in static dispatch — no `&dyn` or
/// `Box<dyn>` in the user-facing surface.
///
/// Fourier-based models (Heston, Bates, Lévy) embed `r` and `q` in their
/// characteristic function. Non-Fourier models (Sabr, HSCM) receive `r`/`q`
/// at pricing time and may ignore them here.
///
/// [`build_surface_from_calibration`]: crate::vol_surface::pipeline::build_surface_from_calibration
pub trait ToModel {
  /// Concrete pricer type produced by this calibration result.
  type Model: ModelPricer;
  fn to_model(&self, r: f64, q: f64) -> Self::Model;
}

/// Bridge from a short-rate calibration result to a concrete tree / lattice model.
///
/// Parallel to [`ToModel`], but for the rates pipeline. Short-rate models
/// (Hull-White, Black-Karasinski, G2++) consume an initial yield curve and a
/// drift offset (`theta`), not spot/strike, so they cannot implement
/// [`ModelPricer`] directly. Instead, calibrators (e.g. swaption / cap /
/// floor calibrators) implement this trait to produce a lattice model that
/// the [`crate::lattice`] instruments and bond / swaption pricers consume.
pub trait ToShortRateModel {
  /// Concrete short-rate model produced by this calibration result.
  type Model;
  /// Build the model from the calibration result.
  ///
  /// `initial_rate` is the time-0 short rate observed from the curve;
  /// `theta` is the drift function offset (often derived from the curve and
  /// passed in by the caller).
  fn to_short_rate_model(&self, initial_rate: f64, theta: f64) -> Self::Model;
}

/// Common interface for calibrators across the quant library.
///
/// Each calibrator declares an [`InitialGuess`](Calibrator::InitialGuess) type
/// (use `()` if no initial guess is supported) and an [`Output`](Calibrator::Output)
/// type that implements [`CalibrationResult`]. The trait gives a single entry
/// point — `calibrate(initial)` — that returns rich diagnostic information
/// (loss / convergence) regardless of the underlying optimiser.
pub trait Calibrator {
  /// Optional initial-guess type.
  type InitialGuess;
  /// Calibration output. Must implement [`CalibrationResult`].
  type Output: CalibrationResult;

  /// Run calibration. Pass `None` to let the calibrator infer an initial guess.
  fn calibrate(&self, initial: Option<Self::InitialGuess>) -> Self::Output;
}

/// Common interface for calibration results.
///
/// Calibrators expose either a structured [`CalibrationLossScore`] (when LM /
/// trust-region pipelines compute the full metric set) or just a scalar
/// `rmse` / `mae` (when the calibrator runs Nelder-Mead or similar). This
/// trait unifies both: `rmse()` is always available, [`loss_score()`](Self::loss_score)
/// returns `Some(&CalibrationLossScore)` when the richer breakdown is computed.
pub trait CalibrationResult {
  /// Root-mean-square pricing error on the calibration grid.
  fn rmse(&self) -> f64;

  /// Whether the underlying optimiser reported convergence.
  fn converged(&self) -> bool;

  /// Structured loss-metric breakdown when available. `None` for
  /// calibrators that only track scalar `rmse`/`mae`.
  fn loss_score(&self) -> Option<&CalibrationLossScore> {
    None
  }
}
