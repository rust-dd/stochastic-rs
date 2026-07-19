//! Input and frequency-window validation for the Fourier-Malliavin engine.

use super::FMVol;
use super::helpers::default_bias_corrected_m;
use crate::traits::FloatExt;

pub(super) fn validate_irregular_inputs<T: FloatExt>(
  prices: &[T],
  times: &[T],
  period: T,
) -> anyhow::Result<(usize, T, T)> {
  validate_period(period)?;
  if prices.len() < 2 {
    anyhow::bail!(
      "at least 2 price observations are required, got {}",
      prices.len()
    );
  }
  if prices.len() != times.len() {
    anyhow::bail!(
      "prices.len()={} must equal times.len()={}",
      prices.len(),
      times.len()
    );
  }
  if prices.iter().any(|price| !price.is_finite()) {
    anyhow::bail!("all price observations must be finite");
  }
  if times.iter().any(|time| !time.is_finite()) {
    anyhow::bail!("all observation times must be finite");
  }

  let mut mesh = T::zero();
  for pair in times.windows(2) {
    let gap = pair[1] - pair[0];
    if gap <= T::zero() {
      anyhow::bail!("observation times must be strictly increasing");
    }
    mesh = mesh.max(gap);
  }

  let span = times[times.len() - 1] - times[0];
  let tolerance = T::from_f64_fast(64.0) * T::epsilon() * span.abs().max(period.abs());
  if (span - period).abs() > tolerance {
    anyhow::bail!("observation times must span the supplied period");
  }

  Ok((prices.len() - 1, mesh, times[0]))
}

pub(super) fn validate_uniform_inputs<T: FloatExt>(
  prices: &[T],
  period: T,
) -> anyhow::Result<usize> {
  validate_period(period)?;
  if prices.len() < 2 {
    anyhow::bail!(
      "at least 2 price observations are required, got {}",
      prices.len()
    );
  }
  if prices.iter().any(|price| !price.is_finite()) {
    anyhow::bail!("all price observations must be finite");
  }
  Ok(prices.len() - 1)
}

pub(super) fn validate_frequency_bounds(
  n: usize,
  n_freq: usize,
  max_freq: usize,
) -> anyhow::Result<()> {
  if n_freq == 0 {
    anyhow::bail!("n_freq must be positive");
  }
  if max_freq < n_freq {
    anyhow::bail!("max_freq={max_freq} must be at least n_freq={n_freq}");
  }
  if max_freq >= n {
    anyhow::bail!("max_freq={max_freq} must be smaller than the increment count n={n}");
  }
  Ok(())
}

pub(super) fn default_max_frequency<T: FloatExt>(
  n: usize,
  n_freq: usize,
  mesh: T,
) -> anyhow::Result<usize> {
  let m_standard = (n_freq as f64).sqrt().floor() as usize;
  let l_standard = (n_freq as f64).powf(0.25).floor() as usize;
  let m_raw = (n_freq as f64).powf(0.4).floor() as usize;
  let l_raw = (n_freq as f64).powf(0.2).floor() as usize;
  let m_corrected = default_bias_corrected_m(mesh);
  let standard_offset = m_standard
    .checked_add(l_standard)
    .ok_or_else(|| anyhow::anyhow!("default M + L overflows usize"))?;
  let raw_offset = m_raw
    .checked_add(l_raw)
    .ok_or_else(|| anyhow::anyhow!("default raw M + L overflows usize"))?;
  let corrected_offset = m_corrected
    .checked_add(l_standard)
    .ok_or_else(|| anyhow::anyhow!("default corrected M + L overflows usize"))?;
  let offset = standard_offset.max(raw_offset).max(corrected_offset);
  n_freq
    .checked_add(offset)
    .filter(|max_freq| *max_freq < n)
    .ok_or_else(|| {
      anyhow::anyhow!(
        "the default windows require a frequency below n={n}; use try_with_freq with explicit windows"
      )
    })
}

fn validate_period<T: FloatExt>(period: T) -> anyhow::Result<()> {
  if !period.is_finite() || period <= T::zero() {
    anyhow::bail!("period must be positive and finite");
  }
  Ok(())
}

impl<T: FloatExt> FMVol<T> {
  pub(super) fn validate_stored_frequency(&self, required: usize) -> anyhow::Result<()> {
    if required > self.max_freq {
      anyhow::bail!(
        "window requires max_freq >= {required}, but max_freq={}",
        self.max_freq
      );
    }
    Ok(())
  }

  pub(super) fn validate_m_window(&self, m_freq: usize) -> anyhow::Result<()> {
    if m_freq == 0 {
      anyhow::bail!("M must be positive");
    }
    let required = self
      .n_freq
      .checked_add(m_freq)
      .ok_or_else(|| anyhow::anyhow!("N + M overflows usize"))?;
    self.validate_stored_frequency(required)
  }

  pub(super) fn validate_ml_window(&self, m_freq: usize, l_freq: usize) -> anyhow::Result<()> {
    self.validate_m_window(m_freq)?;
    if l_freq == 0 {
      anyhow::bail!("L must be positive");
    }
    let required = self
      .n_freq
      .checked_add(m_freq)
      .and_then(|value| value.checked_add(l_freq))
      .ok_or_else(|| anyhow::anyhow!("N + M + L overflows usize"))?;
    self.validate_stored_frequency(required)
  }

  pub(super) fn validate_leverage_window(
    &self,
    m_freq: usize,
    l_freq: usize,
  ) -> anyhow::Result<()> {
    self.validate_m_window(m_freq)?;
    if l_freq == 0 {
      anyhow::bail!("L must be positive");
    }
    let required = m_freq
      .checked_add(l_freq)
      .ok_or_else(|| anyhow::anyhow!("M + L overflows usize"))?;
    self.validate_stored_frequency(required)
  }

  pub(super) fn validate_tau(&self, tau: &[T]) -> anyhow::Result<()> {
    if tau.iter().any(|time| !time.is_finite()) {
      anyhow::bail!("all evaluation times must be finite");
    }
    Ok(())
  }

  pub(super) fn validate_compatible_period(&self, other: &Self) -> anyhow::Result<()> {
    let period_tolerance =
      T::from_f64_fast(64.0) * T::epsilon() * self.period.abs().max(other.period.abs());
    if (self.period - other.period).abs() > period_tolerance {
      anyhow::bail!("Fourier-Malliavin engines must have equal periods");
    }
    let origin_scale = self
      .origin
      .abs()
      .max(other.origin.abs())
      .max(self.period.abs())
      .max(other.period.abs());
    let origin_tolerance = T::from_f64_fast(64.0) * T::epsilon() * origin_scale;
    if (self.origin - other.origin).abs() > origin_tolerance {
      anyhow::bail!("Fourier-Malliavin engines must have equal time origins");
    }
    Ok(())
  }
}
