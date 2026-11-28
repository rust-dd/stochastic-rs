//! A simplex fit as a [`Calibrated`], on the Nelder–Mead driver of `stochastic_rs_stats::optim`.

use std::convert::Infallible;

use basin::CostFunction;
use ndarray::Array1;
use stochastic_rs_stats::optim::simplex::run_nelder_mead;

use crate::CalibrationLossScore;
use crate::QuantError;
use crate::calibration::Calibrated;
use crate::calibration::CalibrationHistory;
use crate::calibration::CalibrationOptions;
use crate::calibration::Knob;

/// The least step of a start simplex along a rate or a volatility parameter: a multiple of a
/// start on its floor would leave the cost flat along that axis.
pub(crate) const LEAST_START_STEP: f64 = 1e-4;

/// A start vertex `factor` times `x` along its axis, or `LEAST_START_STEP` past it.
pub(crate) fn start_vertex(x: f64, factor: f64) -> f64 {
  let step = x * (factor - 1.0);
  x + if step.abs() < LEAST_START_STEP {
    LEAST_START_STEP
  } else {
    step
  }
}

/// A Nelder–Mead fit as a [`Calibrated`]; `price` maps a vertex to its parameters and its model
/// and market prices, for the best vertex and, when history is recorded, each one in the trail.
pub(crate) fn nelder_mead_fit<C, P>(
  cost: C,
  simplex: Vec<Vec<f64>>,
  options: &CalibrationOptions,
  default_max_iterations: usize,
  price: impl Fn(&[f64]) -> (P, Vec<f64>, Vec<f64>),
) -> Result<Calibrated<P>, QuantError>
where
  C: CostFunction<Param = Vec<f64>, Output = f64, Error = Infallible>,
{
  options.check(&[Knob::Patience])?;
  let run = run_nelder_mead(
    cost,
    simplex,
    options.max_iterations.unwrap_or(default_max_iterations) as u64,
    options.tolerance.unwrap_or(1e-10),
    options.record_history,
  );
  let history = run
    .trail
    .iter()
    .enumerate()
    .map(|(iteration, vertex)| {
      let (params, market, model) = price(vertex);
      let residuals = Array1::from_iter(market.iter().zip(&model).map(|(q, m)| q - m));
      let loss = CalibrationLossScore::compute_selected(&market, &model, options.loss_metrics);
      CalibrationHistory::new(iteration, params, residuals, loss)
    })
    .collect();
  let (params, market, model) = price(&run.best);
  let fit = Calibrated::repriced(params, &market, &model, options.loss_metrics, run.converged);
  Ok(
    fit
      .with_iterations(run.iterations as usize)
      .with_message(run.stop.message())
      .with_history(history),
  )
}

#[cfg(test)]
mod tests;
