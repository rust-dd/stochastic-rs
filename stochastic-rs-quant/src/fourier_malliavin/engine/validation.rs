//! Input and frequency-window validation for the Fourier-Malliavin engine.

use super::FmVol;
use super::helpers::default_bias_corrected_m;
use crate::QuantError;
use crate::traits::RealExt;

pub(super) fn validate_irregular_inputs<T: RealExt>(
  prices: &[T],
  times: &[T],
  period: T,
) -> Result<(usize, T, T), QuantError> {
  validate_period(period)?;
  validate_prices(prices)?;
  if prices.len() != times.len() {
    return Err(QuantError::DimensionMismatch {
      what: "observation times, one per price",
      expected: prices.len(),
      got: times.len(),
    });
  }
  if let Some((i, time)) = times.iter().enumerate().find(|(_, t)| !t.is_finite()) {
    return Err(QuantError::invalid(
      "observation times",
      format!("time {i} must be finite, got {time:?}"),
    ));
  }

  let mut mesh = T::zero();
  for (i, pair) in times.windows(2).enumerate() {
    let gap = pair[1] - pair[0];
    if gap <= T::zero() {
      return Err(QuantError::invalid(
        "observation times",
        format!(
          "must be strictly increasing, got time {i} = {:?} before {:?}",
          pair[0], pair[1]
        ),
      ));
    }
    mesh = mesh.max(gap);
  }

  let span = times[times.len() - 1] - times[0];
  let tolerance = T::from_f64_fast(64.0) * T::machine_epsilon() * span.abs().max(period.abs());
  if (span - period).abs() > tolerance {
    return Err(QuantError::invalid(
      "observation times",
      format!("must span the supplied period {period:?}, got a span of {span:?}"),
    ));
  }

  Ok((prices.len() - 1, mesh, times[0]))
}

pub(super) fn validate_uniform_inputs<T: RealExt>(
  prices: &[T],
  period: T,
) -> Result<usize, QuantError> {
  validate_period(period)?;
  validate_prices(prices)?;
  Ok(prices.len() - 1)
}

fn validate_prices<T: RealExt>(prices: &[T]) -> Result<(), QuantError> {
  if prices.len() < 2 {
    return Err(QuantError::InsufficientData {
      what: "prices for the Fourier-Malliavin engine",
      needed: 2,
      got: prices.len(),
    });
  }
  if let Some((i, price)) = prices.iter().enumerate().find(|(_, p)| !p.is_finite()) {
    return Err(QuantError::invalid(
      "price observations",
      format!("price {i} must be finite, got {price:?}"),
    ));
  }
  Ok(())
}

pub(super) fn validate_frequency_bounds(
  n: usize,
  n_freq: usize,
  max_freq: usize,
) -> Result<(), QuantError> {
  if n_freq == 0 {
    return Err(QuantError::invalid("n_freq", "must be positive, got 0"));
  }
  if max_freq < n_freq {
    return Err(QuantError::invalid(
      "max_freq",
      format!("must be at least n_freq = {n_freq}, got {max_freq}"),
    ));
  }
  if max_freq >= n {
    return Err(QuantError::invalid(
      "max_freq",
      format!("must be smaller than the increment count n = {n}, got {max_freq}"),
    ));
  }
  Ok(())
}

pub(super) fn window_overflow(window: &str) -> QuantError {
  QuantError::Numerical {
    what: "frequency window",
    reason: format!("{window} overflows usize"),
  }
}

pub(super) fn default_max_frequency<T: RealExt>(
  n: usize,
  n_freq: usize,
  mesh: T,
) -> Result<usize, QuantError> {
  let m_standard = (n_freq as f64).sqrt().floor() as usize;
  let l_standard = (n_freq as f64).powf(0.25).floor() as usize;
  let m_raw = (n_freq as f64).powf(0.4).floor() as usize;
  let l_raw = (n_freq as f64).powf(0.2).floor() as usize;
  let m_corrected = default_bias_corrected_m(mesh);
  let standard_offset = m_standard
    .checked_add(l_standard)
    .ok_or_else(|| window_overflow("default M + L"))?;
  let raw_offset = m_raw
    .checked_add(l_raw)
    .ok_or_else(|| window_overflow("default raw M + L"))?;
  let corrected_offset = m_corrected
    .checked_add(l_standard)
    .ok_or_else(|| window_overflow("default corrected M + L"))?;
  let offset = standard_offset.max(raw_offset).max(corrected_offset);
  let max_freq = n_freq
    .checked_add(offset)
    .ok_or_else(|| window_overflow("default N + M + L"))?;
  if max_freq >= n {
    return Err(QuantError::InsufficientData {
      what: "prices for the default frequency windows",
      needed: max_freq + 2,
      got: n + 1,
    });
  }
  Ok(max_freq)
}

fn validate_period<T: RealExt>(period: T) -> Result<(), QuantError> {
  if !period.is_finite() || period <= T::zero() {
    return Err(QuantError::invalid(
      "period",
      format!("must be positive and finite, got {period:?}"),
    ));
  }
  Ok(())
}

impl<T: RealExt> FmVol<T> {
  pub(super) fn validate_stored_frequency(&self, required: usize) -> Result<(), QuantError> {
    if required > self.max_freq {
      return Err(QuantError::invalid(
        "frequency window",
        format!(
          "requires max_freq >= {required}, but the engine stores max_freq = {}",
          self.max_freq
        ),
      ));
    }
    Ok(())
  }

  pub(super) fn validate_m_window(&self, m_freq: usize) -> Result<(), QuantError> {
    if m_freq == 0 {
      return Err(QuantError::invalid("M", "must be positive, got 0"));
    }
    let required = self
      .n_freq
      .checked_add(m_freq)
      .ok_or_else(|| window_overflow("N + M"))?;
    self.validate_stored_frequency(required)
  }

  pub(super) fn validate_ml_window(&self, m_freq: usize, l_freq: usize) -> Result<(), QuantError> {
    self.validate_m_window(m_freq)?;
    if l_freq == 0 {
      return Err(QuantError::invalid("L", "must be positive, got 0"));
    }
    let required = self
      .n_freq
      .checked_add(m_freq)
      .and_then(|value| value.checked_add(l_freq))
      .ok_or_else(|| window_overflow("N + M + L"))?;
    self.validate_stored_frequency(required)
  }

  pub(super) fn validate_leverage_window(
    &self,
    m_freq: usize,
    l_freq: usize,
  ) -> Result<(), QuantError> {
    self.validate_m_window(m_freq)?;
    if l_freq == 0 {
      return Err(QuantError::invalid("L", "must be positive, got 0"));
    }
    let required = m_freq
      .checked_add(l_freq)
      .ok_or_else(|| window_overflow("M + L"))?;
    self.validate_stored_frequency(required)
  }

  pub(super) fn validate_tau(&self, tau: &[T]) -> Result<(), QuantError> {
    if let Some((i, time)) = tau.iter().enumerate().find(|(_, t)| !t.is_finite()) {
      return Err(QuantError::invalid(
        "evaluation times",
        format!("time {i} must be finite, got {time:?}"),
      ));
    }
    Ok(())
  }

  pub(super) fn validate_compatible_period(&self, other: &Self) -> Result<(), QuantError> {
    let period_tolerance =
      T::from_f64_fast(64.0) * T::machine_epsilon() * self.period.abs().max(other.period.abs());
    if (self.period - other.period).abs() > period_tolerance {
      return Err(QuantError::invalid(
        "Fourier-Malliavin engine pair",
        format!(
          "periods must be equal, got {:?} and {:?}",
          self.period, other.period
        ),
      ));
    }
    let origin_scale = self
      .origin
      .abs()
      .max(other.origin.abs())
      .max(self.period.abs())
      .max(other.period.abs());
    let origin_tolerance = T::from_f64_fast(64.0) * T::machine_epsilon() * origin_scale;
    if (self.origin - other.origin).abs() > origin_tolerance {
      return Err(QuantError::invalid(
        "Fourier-Malliavin engine pair",
        format!(
          "time origins must be equal, got {:?} and {:?}",
          self.origin, other.origin
        ),
      ));
    }
    Ok(())
  }
}
