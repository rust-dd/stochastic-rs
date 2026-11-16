//! The result every calibrator returns, its optional iterate trail and the knobs they share.

use ndarray::Array1;
use stochastic_rs_distributions::RealExt;

use crate::CalibrationLossScore;
use crate::LossMetric;
use crate::calibration::Regularization;
use crate::traits::CalibrationResult;

/// What every calibrator returns: the parameters, the fit, and the trail when history was recorded.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct Calibrated<P, D = ()> {
  pub params: P,
  pub loss: CalibrationLossScore,
  pub converged: bool,
  pub iterations: Option<usize>,
  pub message: Option<String>,
  /// Worst absolute repricing error over the quotes; `None` when the calibrator has no quote grid.
  pub max_error: Option<f64>,
  pub history: Vec<CalibrationHistory<P>>,
  /// Model-specific output beside the parameters; `()` for most calibrators.
  pub diagnostics: D,
}

impl<P> Calibrated<P> {
  pub fn new(params: P, loss: CalibrationLossScore, converged: bool) -> Self {
    Self {
      params,
      loss,
      converged,
      iterations: None,
      message: None,
      max_error: None,
      history: Vec::new(),
      diagnostics: (),
    }
  }

  pub(crate) fn repriced(
    params: P,
    market: &[f64],
    model: &[f64],
    metrics: &[LossMetric],
    converged: bool,
  ) -> Self {
    let loss = CalibrationLossScore::compute_selected(market, model, metrics);
    let max_error = f64::max_or_nan_fold(market.iter().zip(model).map(|(m, q)| (m - q).abs()));
    Self::new(params, loss, converged).with_max_error(max_error)
  }
}

impl<P, D> Calibrated<P, D> {
  pub fn with_iterations(mut self, n: usize) -> Self {
    self.iterations = Some(n);
    self
  }

  pub fn with_message(mut self, message: impl Into<String>) -> Self {
    self.message = Some(message.into());
    self
  }

  pub fn with_max_error(mut self, max_error: f64) -> Self {
    self.max_error = Some(max_error);
    self
  }

  pub fn with_history(mut self, history: Vec<CalibrationHistory<P>>) -> Self {
    self.history = history;
    self
  }

  pub fn with_diagnostics<E>(self, diagnostics: E) -> Calibrated<P, E> {
    Calibrated {
      params: self.params,
      loss: self.loss,
      converged: self.converged,
      iterations: self.iterations,
      message: self.message,
      max_error: self.max_error,
      history: self.history,
      diagnostics,
    }
  }
}

impl<P: Clone, D> CalibrationResult for Calibrated<P, D> {
  type Params = P;

  fn rmse(&self) -> f64 {
    self.loss.get(LossMetric::Rmse)
  }

  fn converged(&self) -> bool {
    self.converged
  }

  fn params(&self) -> P {
    self.params.clone()
  }

  fn loss_score(&self) -> Option<&CalibrationLossScore> {
    Some(&self.loss)
  }

  fn iterations(&self) -> Option<usize> {
    self.iterations
  }

  fn message(&self) -> Option<&str> {
    self.message.as_deref()
  }

  fn max_error(&self) -> Option<f64> {
    self.max_error
  }

  fn history(&self) -> &[CalibrationHistory<P>] {
    &self.history
  }
}

/// One optimiser iterate: the parameters it was evaluated at, its residual vector and its loss.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct CalibrationHistory<P> {
  pub iteration: usize,
  pub params: P,
  pub residuals: Array1<f64>,
  pub loss: CalibrationLossScore,
}

impl<P> CalibrationHistory<P> {
  pub fn new(
    iteration: usize,
    params: P,
    residuals: Array1<f64>,
    loss: CalibrationLossScore,
  ) -> Self {
    Self {
      iteration,
      params,
      residuals,
      loss,
    }
  }

  /// One metric across the trail; NaN where an entry did not compute it.
  pub fn metric(history: &[Self], metric: LossMetric) -> Vec<f64> {
    history.iter().map(|entry| entry.loss.get(metric)).collect()
  }
}

/// The knobs every calibrator shares; a model's own stay on its calibrator as `with_*` builders.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct CalibrationOptions {
  /// Iteration cap; `None` keeps each optimiser's own, for Levenberg–Marquardt its budget.
  pub max_iterations: Option<usize>,
  /// `None` keeps the optimiser's own: Levenberg–Marquardt 30 ε relative, Nelder–Mead 1e-10 spread.
  pub tolerance: Option<f64>,
  /// Levenberg–Marquardt budget: `patience · (n + 1)` evaluations for `n` parameters; default 100.
  pub patience: usize,
  /// Record the optimiser's iterates into `Calibrated::history`.
  pub record_history: bool,
  /// Metrics computed for the result and each history entry; every `LossMetric` by default.
  pub loss_metrics: &'static [LossMetric],
  /// Tikhonov pull in the calibrator's natural parameter order.
  pub regularization: Option<Regularization>,
}

impl Default for CalibrationOptions {
  fn default() -> Self {
    Self {
      max_iterations: None,
      tolerance: None,
      patience: 100,
      record_history: false,
      loss_metrics: &LossMetric::ALL,
      regularization: None,
    }
  }
}

impl CalibrationOptions {
  /// Panics unless `n > 0`.
  pub fn with_max_iterations(mut self, n: usize) -> Self {
    assert!(n > 0, "n must satisfy `n > 0`, got n = {n:?}");
    self.max_iterations = Some(n);
    self
  }

  /// Panics unless `tolerance` is finite and positive.
  pub fn with_tolerance(mut self, tolerance: f64) -> Self {
    assert!(
      tolerance.is_finite() && tolerance > 0.0,
      "tolerance must satisfy `tolerance.is_finite() && tolerance > 0.0`, got tolerance = {tolerance:?}"
    );
    self.tolerance = Some(tolerance);
    self
  }

  /// Panics unless `patience > 0`.
  pub fn with_patience(mut self, patience: usize) -> Self {
    assert!(
      patience > 0,
      "patience must satisfy `patience > 0`, got patience = {patience:?}"
    );
    self.patience = patience;
    self
  }

  pub fn with_record_history(mut self, record: bool) -> Self {
    self.record_history = record;
    self
  }

  pub fn with_loss_metrics(mut self, metrics: &'static [LossMetric]) -> Self {
    self.loss_metrics = metrics;
    self
  }

  pub fn with_regularization(mut self, regularization: Regularization) -> Self {
    self.regularization = Some(regularization);
    self
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn repriced_takes_the_worst_error_and_keeps_a_nan() {
    let fit = Calibrated::repriced(
      (),
      &[1.0, 2.0, 3.0],
      &[1.5, 2.0, 1.0],
      &[LossMetric::Mae],
      true,
    );
    assert_eq!(fit.max_error, Some(2.0));
    assert_eq!(fit.loss.try_get(LossMetric::Mae), Some(2.5 / 3.0));
    assert_eq!(fit.loss.try_get(LossMetric::Rmse), None);
    let nan = Calibrated::repriced((), &[1.0, 2.0], &[f64::NAN, 2.0], &[], true);
    assert!(nan.max_error.unwrap().is_nan());
  }

  #[test]
  fn metric_reads_one_loss_across_the_trail() {
    let entry = |i: usize, model: f64| {
      CalibrationHistory::new(
        i,
        (),
        Array1::from_elem(1, 1.0 - model),
        CalibrationLossScore::compute(&[1.0], &[model]),
      )
    };
    let trail = [entry(0, 3.0), entry(1, 1.5)];
    assert_eq!(
      CalibrationHistory::metric(&trail, LossMetric::Mae),
      vec![2.0, 0.5]
    );
  }

  #[test]
  #[should_panic(expected = "tolerance must satisfy `tolerance.is_finite() && tolerance > 0.0`")]
  fn a_zero_tolerance_panics() {
    let _ = CalibrationOptions::default().with_tolerance(0.0);
  }

  #[test]
  #[should_panic(expected = "n must satisfy `n > 0`, got n = 0")]
  fn a_zero_iteration_cap_panics() {
    let _ = CalibrationOptions::default().with_max_iterations(0);
  }

  #[test]
  #[should_panic(expected = "patience must satisfy `patience > 0`, got patience = 0")]
  fn a_zero_patience_panics() {
    let _ = CalibrationOptions::default().with_patience(0);
  }
}
