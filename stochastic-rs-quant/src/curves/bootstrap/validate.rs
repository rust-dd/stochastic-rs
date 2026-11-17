//! The per-quote checks a bootstrap runs before any solve, in the caller's order.

use crate::QuantError;
use crate::curves::types::BootstrapInstrument;
use crate::traits::RealExt;

/// `InvalidQuote { index }` unless `instrument` can give one pillar after `t = 0`.
pub(super) fn check<T: RealExt>(
  index: usize,
  instrument: &BootstrapInstrument<T>,
  dual_curve: bool,
) -> Result<(), QuantError> {
  let bad = |reason: String| Err(QuantError::InvalidQuote { index, reason });
  if let BootstrapInstrument::SwapWithSchedule { fixed_times, .. } = instrument
    && fixed_times.is_empty()
  {
    return bad("fixed_times must satisfy `!fixed_times.is_empty()`".into());
  }
  let maturity = instrument.maturity();
  if !maturity.is_finite() {
    return bad(format!(
      "maturity must satisfy `maturity.is_finite()`, got maturity = {maturity:?}"
    ));
  }
  if maturity <= T::zero() {
    return bad(format!(
      "maturity must satisfy `maturity > T::zero()`, got maturity = {maturity:?}"
    ));
  }
  let (rate, name) = match instrument {
    BootstrapInstrument::Future { price, .. } => (*price, "price"),
    BootstrapInstrument::Deposit { rate, .. }
    | BootstrapInstrument::Fra { rate, .. }
    | BootstrapInstrument::Swap { rate, .. }
    | BootstrapInstrument::SwapWithSchedule { rate, .. } => (*rate, "rate"),
  };
  if !rate.is_finite() {
    return bad(format!(
      "{name} must satisfy `{name}.is_finite()`, got {name} = {rate:?}"
    ));
  }
  match instrument {
    BootstrapInstrument::Deposit { .. } => Ok(()),
    BootstrapInstrument::Fra { start, end, .. } => starts_before(*start, *end).or_else(bad),
    BootstrapInstrument::Future {
      start, end, sigma, ..
    } => {
      if !sigma.is_finite() {
        return bad(format!(
          "sigma must satisfy `sigma.is_finite()`, got sigma = {sigma:?}"
        ));
      }
      starts_before(*start, *end).or_else(bad)
    }
    BootstrapInstrument::Swap {
      maturity,
      frequency,
      ..
    } => {
      if dual_curve {
        return bad(
          "a swap against an exogenous discount curve must be a `SwapWithSchedule`".into(),
        );
      }
      match whole_payments(*maturity, frequency.periods_per_year()) {
        Some(_) => Ok(()),
        None => bad(format!(
          "maturity must be within 0.1 of a payment from a whole number of them, got maturity = {maturity:?}, frequency = {frequency}"
        )),
      }
    }
    BootstrapInstrument::SwapWithSchedule {
      fixed_times,
      float_times,
      ..
    } => {
      increasing("fixed_times", fixed_times).or_else(bad)?;
      if !dual_curve {
        return Ok(());
      }
      increasing("float_times", float_times).or_else(bad)?;
      let float_end = float_times.last().copied().unwrap_or_else(T::nan);
      if (float_end - maturity).abs() <= T::from_f64_fast(1e-8) {
        Ok(())
      } else {
        bad(format!(
          "float_times must end at the maturity {maturity:?}, got {float_end:?}"
        ))
      }
    }
  }
}

/// The payment count of a swap whose `maturity · per_year` is within 0.1 of a whole number ≥ 1.
pub(super) fn whole_payments<T: RealExt>(maturity: T, per_year: u32) -> Option<usize> {
  let raw = maturity * T::from_f64_fast(f64::from(per_year));
  let rounded = raw.round();
  ((raw - rounded).abs() < T::from_f64_fast(0.1) && rounded >= T::one())
    .then(|| rounded.to_f64().map_or(0, |n| n as usize))
}

fn starts_before<T: RealExt>(start: T, end: T) -> Result<(), String> {
  if start >= T::zero() && start < end {
    Ok(())
  } else {
    Err(format!(
      "start must satisfy `start >= T::zero() && start < end`, got start = {start:?}, end = {end:?}"
    ))
  }
}

fn increasing<T: RealExt>(name: &str, times: &[T]) -> Result<(), String> {
  let positive = times.first().is_some_and(|t| *t > T::zero());
  if positive && times.windows(2).all(|w| w[0] < w[1]) {
    Ok(())
  } else {
    Err(format!(
      "{name} must be positive and strictly increasing, got {name} = {times:?}"
    ))
  }
}
