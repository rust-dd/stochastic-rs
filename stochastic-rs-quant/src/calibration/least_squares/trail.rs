use ndarray::ArrayView1;

use crate::CalibrationLossScore;
use crate::calibration::CalibrationHistory;
use crate::calibration::CalibrationOptions;

/// One fit's trail: `residuals` stages the point it evaluated, and `accept` keeps it.
pub(crate) struct Trail<P> {
  entries: Vec<CalibrationHistory<P>>,
  pending: Option<CalibrationHistory<P>>,
  negated: bool,
}

impl<P: Clone> Trail<P> {
  /// The trail of a fit whose rows are market − model.
  pub(crate) fn new() -> Self {
    Self {
      entries: Vec::new(),
      pending: None,
      negated: false,
    }
  }

  /// The trail of a fit whose rows are model − market, which an entry stores negated.
  pub(crate) fn negated() -> Self {
    Self {
      negated: true,
      ..Self::new()
    }
  }

  /// `residuals` are the fit's rows at `params`; a no-op unless `options.record_history`.
  pub(crate) fn stage(
    &mut self,
    options: &CalibrationOptions,
    market: &[f64],
    model: &[f64],
    params: &P,
    residuals: ArrayView1<f64>,
  ) {
    if !options.record_history {
      return;
    }
    let residuals = if self.negated {
      -&residuals
    } else {
      residuals.to_owned()
    };
    let loss = CalibrationLossScore::compute_selected(market, model, options.loss_metrics);
    let entry = CalibrationHistory::new(self.entries.len(), params.clone(), residuals, loss);
    self.pending = Some(entry);
  }

  pub(crate) fn accept(&mut self) {
    self.entries.extend(self.pending.take());
  }

  pub(crate) fn into_entries(self) -> Vec<CalibrationHistory<P>> {
    self.entries
  }
}
