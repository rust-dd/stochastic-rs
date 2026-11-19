//! Spectral analysis utilities (periodogram and FFT-based spectrum search).
//!
//! A constant signal has a periodogram too: its power sits at 0 Hz, which detrending removes.

use std::f64::consts::PI;

use ndarray::Array1;
use ndarray::ArrayView1;
use ndrustfft::FftHandler;
use ndrustfft::ndfft;
use num_complex::Complex64;

use crate::StatsError;
use crate::input::as_slice_or_copy;
use crate::input::validate;

/// Detrending strategy used before spectral estimation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum DetrendMethod {
  /// No detrending.
  None,
  /// Remove sample mean.
  Mean,
  /// Remove best-fit linear trend.
  Linear,
}

/// Window function used before FFT.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum WindowFunction {
  /// w\[n\] = 1
  Rectangular,
  /// w\[n\] = 0.5 - 0.5 cos(2πn/(N-1))
  Hann,
  /// w\[n\] = 0.54 - 0.46 cos(2πn/(N-1))
  Hamming,
  /// w\[n\] = 0.42 - 0.5 cos(2πn/(N-1)) + 0.08 cos(4πn/(N-1))
  Blackman,
}

/// Spectrum scaling mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpectrumScaling {
  /// Power spectrum.
  Power,
  /// Power spectral density.
  Density,
}

/// Configuration for periodogram estimation.
#[derive(Debug, Clone, Copy)]
pub struct PeriodogramConfig {
  /// Sampling frequency in Hz.
  pub sampling_rate: f64,
  /// FFT size. If `None`, the next power of two is used.
  pub nfft: Option<usize>,
  /// Detrending strategy.
  pub detrend: DetrendMethod,
  /// Taper applied to the detrended series before the FFT.
  pub window: WindowFunction,
  /// Whether to return one-sided spectrum (for real signals).
  pub onesided: bool,
  /// Power spectrum or power spectral density.
  pub scaling: SpectrumScaling,
}

impl Default for PeriodogramConfig {
  fn default() -> Self {
    Self {
      sampling_rate: 1.0,
      nfft: None,
      detrend: DetrendMethod::Mean,
      window: WindowFunction::Hann,
      onesided: true,
      scaling: SpectrumScaling::Density,
    }
  }
}

/// Periodogram output.
#[derive(Debug, Clone)]
pub struct PeriodogramResult {
  /// Frequency bins in Hz.
  pub frequencies: Vec<f64>,
  /// Spectrum values at `frequencies`.
  pub spectrum: Vec<f64>,
  /// Frequency resolution in Hz.
  pub resolution_hz: f64,
  /// FFT size used.
  pub nfft: usize,
  /// Original sample size.
  pub nobs: usize,
}

/// Single spectral peak.
#[derive(Debug, Clone, Copy)]
pub struct SpectrumPeak {
  /// Bin index in the periodogram output.
  pub index: usize,
  /// Peak frequency in Hz.
  pub frequency_hz: f64,
  /// Peak period in seconds (if frequency > 0).
  pub period_s: Option<f64>,
  /// Peak power/spectral density value.
  pub power: f64,
}

/// Configuration for FFT-based peak search.
#[derive(Debug, Clone, Copy)]
pub struct SpectrumSearchConfig {
  /// Periodogram configuration used internally.
  pub periodogram: PeriodogramConfig,
  /// Number of peaks to return.
  pub top_k: usize,
  /// Minimum frequency in Hz (inclusive).
  pub min_frequency_hz: Option<f64>,
  /// Maximum frequency in Hz (inclusive).
  pub max_frequency_hz: Option<f64>,
  /// Minimum separation between selected peaks (in bins).
  pub min_separation_bins: usize,
  /// Whether to ignore the DC bin (0 Hz).
  pub exclude_dc: bool,
}

impl Default for SpectrumSearchConfig {
  fn default() -> Self {
    Self {
      periodogram: PeriodogramConfig::default(),
      top_k: 3,
      min_frequency_hz: None,
      max_frequency_hz: None,
      min_separation_bins: 2,
      exclude_dc: true,
    }
  }
}

fn detrend_in_place(data: &mut [f64], method: DetrendMethod) {
  match method {
    DetrendMethod::None => {}
    DetrendMethod::Mean => {
      let mean = data.iter().sum::<f64>() / data.len() as f64;
      for x in data {
        *x -= mean;
      }
    }
    DetrendMethod::Linear => {
      let n = data.len();
      let n_f = n as f64;
      let mean_x = (n_f - 1.0) * 0.5;
      let mean_y = data.iter().sum::<f64>() / n_f;

      let mut num = 0.0;
      let mut den = 0.0;
      for (i, &y) in data.iter().enumerate() {
        let x = i as f64;
        num += (x - mean_x) * (y - mean_y);
        den += (x - mean_x) * (x - mean_x);
      }
      let slope = if den > 0.0 { num / den } else { 0.0 };
      let intercept = mean_y - slope * mean_x;

      for (i, y) in data.iter_mut().enumerate() {
        let trend = intercept + slope * i as f64;
        *y -= trend;
      }
    }
  }
}

fn window_coeff(window: WindowFunction, i: usize, n: usize) -> f64 {
  if n <= 1 {
    return 1.0;
  }
  let a = 2.0 * PI * i as f64 / (n as f64 - 1.0);
  match window {
    WindowFunction::Rectangular => 1.0,
    WindowFunction::Hann => 0.5 - 0.5 * a.cos(),
    WindowFunction::Hamming => 0.54 - 0.46 * a.cos(),
    WindowFunction::Blackman => 0.42 - 0.5 * a.cos() + 0.08 * (2.0 * a).cos(),
  }
}

fn apply_window_in_place(data: &mut [f64], window: WindowFunction) -> f64 {
  let n = data.len();
  let mut sum_w2 = 0.0;
  for (i, x) in data.iter_mut().enumerate() {
    let w = window_coeff(window, i, n);
    *x *= w;
    sum_w2 += w * w;
  }
  sum_w2
}

/// FFT periodogram of `signal`; `Err` below 4 points, on a non-finite one or an overflowing
/// spectrum; panics unless `sampling_rate > 0` is finite and a given `nfft >= signal.len()`.
pub fn periodogram_fft(
  signal: ArrayView1<f64>,
  cfg: PeriodogramConfig,
) -> Result<PeriodogramResult, StatsError> {
  assert!(
    cfg.sampling_rate.is_finite() && cfg.sampling_rate > 0.0,
    "sampling_rate must satisfy `sampling_rate.is_finite() && sampling_rate > 0.0`, got sampling_rate = {:?}",
    cfg.sampling_rate
  );
  let n = signal.len();
  if let Some(nfft) = cfg.nfft {
    assert!(
      nfft >= n,
      "nfft must satisfy `nfft >= signal.len()`, got nfft = {nfft:?}, signal.len() = {n:?}"
    );
  }
  let signal = as_slice_or_copy(signal);
  validate(&signal, 4)?;

  let nfft = cfg.nfft.unwrap_or_else(|| n.next_power_of_two());

  let mut x = signal.to_vec();
  detrend_in_place(&mut x, cfg.detrend);
  let sum_w2 = apply_window_in_place(&mut x, cfg.window);

  let mut input = Array1::<Complex64>::zeros(nfft);
  for i in 0..n {
    input[i] = Complex64::new(x[i], 0.0);
  }

  let mut fft_out = Array1::<Complex64>::zeros(nfft);
  let fft = FftHandler::<f64>::new(nfft);
  ndfft(&input, &mut fft_out, &fft, 0);

  let resolution_hz = cfg.sampling_rate / nfft as f64;
  let bins = if cfg.onesided { nfft / 2 + 1 } else { nfft };

  let mut frequencies = Vec::with_capacity(bins);
  let mut spectrum = Vec::with_capacity(bins);

  let base_scale = n as f64 * sum_w2;
  for k in 0..bins {
    let mut p = fft_out[k].norm_sqr() / base_scale;

    if cfg.onesided && k > 0 && !(nfft.is_multiple_of(2) && k == nfft / 2) {
      p *= 2.0;
    }

    if matches!(cfg.scaling, SpectrumScaling::Density) {
      p /= cfg.sampling_rate;
    }

    if !p.is_finite() {
      return Err(StatsError::Numerical {
        stage: "periodogram",
      });
    }
    frequencies.push(k as f64 * resolution_hz);
    spectrum.push(p);
  }

  Ok(PeriodogramResult {
    frequencies,
    spectrum,
    resolution_hz,
    nfft,
    nobs: n,
  })
}

/// Up to `top_k` periodogram peaks by descending power; `Err` as [`periodogram_fft`], and panics
/// as it or unless `top_k >= 1`.
pub fn fft_spectrum_search(
  signal: ArrayView1<f64>,
  cfg: SpectrumSearchConfig,
) -> Result<Vec<SpectrumPeak>, StatsError> {
  assert!(
    cfg.top_k >= 1,
    "top_k must satisfy `top_k >= 1`, got top_k = {:?}",
    cfg.top_k
  );

  let pg = periodogram_fft(signal, cfg.periodogram)?;
  let n = pg.spectrum.len();
  if n == 0 {
    return Ok(Vec::new());
  }

  let min_f = cfg.min_frequency_hz.unwrap_or(0.0);
  let max_f = cfg.max_frequency_hz.unwrap_or(f64::INFINITY);

  let mut candidates = Vec::new();
  for i in 0..n {
    let f = pg.frequencies[i];
    if f < min_f || f > max_f {
      continue;
    }
    if cfg.exclude_dc && i == 0 {
      continue;
    }

    let p = pg.spectrum[i];
    let left = if i > 0 {
      pg.spectrum[i - 1]
    } else {
      f64::NEG_INFINITY
    };
    let right = if i + 1 < n {
      pg.spectrum[i + 1]
    } else {
      f64::NEG_INFINITY
    };

    if p >= left && p >= right {
      candidates.push((i, f, p));
    }
  }

  if candidates.is_empty() {
    if let Some((i, f, p)) = pg
      .frequencies
      .iter()
      .zip(pg.spectrum.iter())
      .enumerate()
      .filter(|(i, (f, _))| (!cfg.exclude_dc || *i != 0) && **f >= min_f && **f <= max_f)
      .max_by(|a, b| a.1.1.total_cmp(b.1.1))
      .map(|(i, (f, p))| (i, *f, *p))
    {
      return Ok(vec![SpectrumPeak {
        index: i,
        frequency_hz: f,
        period_s: if f > 0.0 { Some(1.0 / f) } else { None },
        power: p,
      }]);
    }
    return Ok(Vec::new());
  }

  candidates.sort_by(|a, b| b.2.total_cmp(&a.2));

  let min_sep = cfg.min_separation_bins.max(1);
  let mut peaks = Vec::with_capacity(cfg.top_k);
  for (idx, f, p) in candidates {
    if peaks
      .iter()
      .all(|peak: &SpectrumPeak| peak.index.abs_diff(idx) >= min_sep)
    {
      peaks.push(SpectrumPeak {
        index: idx,
        frequency_hz: f,
        period_s: if f > 0.0 { Some(1.0 / f) } else { None },
        power: p,
      });
      if peaks.len() >= cfg.top_k {
        break;
      }
    }
  }

  Ok(peaks)
}

/// The strongest periodogram peak, `None` when the search band holds none; `Err` and panics as
/// [`periodogram_fft`].
pub fn dominant_frequency_fft(
  signal: ArrayView1<f64>,
  mut cfg: SpectrumSearchConfig,
) -> Result<Option<SpectrumPeak>, StatsError> {
  cfg.top_k = 1;
  Ok(fft_spectrum_search(signal, cfg)?.into_iter().next())
}

#[cfg(test)]
mod tests;
