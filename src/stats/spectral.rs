//! Spectral analysis utilities (periodogram and FFT-based spectrum search).

use std::f64::consts::PI;

use ndarray::Array1;
use ndrustfft::FftHandler;
use ndrustfft::ndfft;
use num_complex::Complex64;

/// Detrending strategy used before spectral estimation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
pub enum WindowFunction {
  /// w[n] = 1
  Rectangular,
  /// w[n] = 0.5 - 0.5 cos(2πn/(N-1))
  Hann,
  /// w[n] = 0.54 - 0.46 cos(2πn/(N-1))
  Hamming,
  /// w[n] = 0.42 - 0.5 cos(2πn/(N-1)) + 0.08 cos(4πn/(N-1))
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
  /// Window function.
  pub window: WindowFunction,
  /// Whether to return one-sided spectrum (for real signals).
  pub onesided: bool,
  /// Spectrum scaling.
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
  pub sample_size: usize,
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
  sum_w2.max(1e-12)
}

/// Compute periodogram using FFT.
///
/// # Panics
/// Panics on invalid configuration, non-finite signal values, or too short signal.
pub fn periodogram_fft(signal: &[f64], cfg: PeriodogramConfig) -> PeriodogramResult {
  assert!(
    signal.len() >= 4,
    "signal must contain at least 4 observations"
  );
  assert!(
    signal.iter().all(|v| v.is_finite()),
    "signal must contain only finite values"
  );
  assert!(
    cfg.sampling_rate > 0.0 && cfg.sampling_rate.is_finite(),
    "sampling_rate must be positive and finite"
  );

  let n = signal.len();
  let mut nfft = cfg.nfft.unwrap_or_else(|| n.next_power_of_two());
  if nfft < n {
    nfft = n;
  }

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

  let base_scale = (n as f64 * sum_w2).max(1e-12);
  for k in 0..bins {
    let mut p = fft_out[k].norm_sqr() / base_scale;

    if cfg.onesided && k > 0 && !(nfft % 2 == 0 && k == nfft / 2) {
      p *= 2.0;
    }

    if matches!(cfg.scaling, SpectrumScaling::Density) {
      p /= cfg.sampling_rate;
    }

    frequencies.push(k as f64 * resolution_hz);
    spectrum.push(p.max(0.0));
  }

  PeriodogramResult {
    frequencies,
    spectrum,
    resolution_hz,
    nfft,
    sample_size: n,
  }
}

/// FFT-based spectrum peak search over the periodogram.
///
/// Returns up to `top_k` peaks sorted by descending power.
///
/// # Panics
/// Panics on invalid configuration.
pub fn fft_spectrum_search(signal: &[f64], cfg: SpectrumSearchConfig) -> Vec<SpectrumPeak> {
  assert!(cfg.top_k > 0, "top_k must be positive");

  let pg = periodogram_fft(signal, cfg.periodogram);
  let n = pg.spectrum.len();
  if n == 0 {
    return Vec::new();
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
      return vec![SpectrumPeak {
        index: i,
        frequency_hz: f,
        period_s: if f > 0.0 { Some(1.0 / f) } else { None },
        power: p,
      }];
    }
    return Vec::new();
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

  peaks
}

/// Convenience function returning the strongest frequency peak.
pub fn dominant_frequency_fft(
  signal: &[f64],
  mut cfg: SpectrumSearchConfig,
) -> Option<SpectrumPeak> {
  cfg.top_k = 1;
  fft_spectrum_search(signal, cfg).into_iter().next()
}

#[cfg(test)]
mod tests {
  use super::DetrendMethod;
  use super::PeriodogramConfig;
  use super::SpectrumScaling;
  use super::SpectrumSearchConfig;
  use super::WindowFunction;
  use super::dominant_frequency_fft;
  use super::fft_spectrum_search;
  use super::periodogram_fft;
  use crate::distributions::normal::SimdNormal;

  fn sine_wave(freq_hz: f64, sampling_rate: f64, n: usize, amplitude: f64, phase: f64) -> Vec<f64> {
    (0..n)
      .map(|i| {
        let t = i as f64 / sampling_rate;
        amplitude * (2.0 * std::f64::consts::PI * freq_hz * t + phase).sin()
      })
      .collect()
  }

  #[test]
  fn periodogram_detects_single_tone_frequency() {
    let fs = 256.0;
    let n = 2048;
    let f0 = 17.5;
    let x = sine_wave(f0, fs, n, 1.0, 0.3);

    let cfg = PeriodogramConfig {
      sampling_rate: fs,
      nfft: Some(n),
      detrend: DetrendMethod::None,
      window: WindowFunction::Rectangular,
      onesided: true,
      scaling: SpectrumScaling::Power,
    };
    let pg = periodogram_fft(&x, cfg);

    let (idx, _) = pg
      .spectrum
      .iter()
      .enumerate()
      .skip(1)
      .max_by(|a, b| a.1.total_cmp(b.1))
      .expect("periodogram must contain a peak");

    let f_est = pg.frequencies[idx];
    assert!(
      (f_est - f0).abs() <= pg.resolution_hz,
      "single-tone frequency mismatch: est={f_est}, true={f0}, res={}",
      pg.resolution_hz
    );
  }

  #[test]
  fn fft_spectrum_search_finds_two_tones() {
    let fs = 512.0;
    let n = 4096;
    let f1 = 12.0;
    let f2 = 70.0;

    let mut x = sine_wave(f1, fs, n, 1.0, 0.0);
    let x2 = sine_wave(f2, fs, n, 0.6, 0.4);
    for (a, b) in x.iter_mut().zip(x2.iter()) {
      *a += *b;
    }

    let cfg = SpectrumSearchConfig {
      periodogram: PeriodogramConfig {
        sampling_rate: fs,
        nfft: Some(n),
        detrend: DetrendMethod::Mean,
        window: WindowFunction::Hann,
        onesided: true,
        scaling: SpectrumScaling::Power,
      },
      top_k: 2,
      min_frequency_hz: Some(1.0),
      max_frequency_hz: Some(120.0),
      min_separation_bins: 6,
      exclude_dc: true,
    };

    let peaks = fft_spectrum_search(&x, cfg);
    assert_eq!(peaks.len(), 2, "expected 2 peaks, got {peaks:?}");

    let mut freqs = peaks.iter().map(|p| p.frequency_hz).collect::<Vec<_>>();
    freqs.sort_by(f64::total_cmp);

    let tol = fs / n as f64 * 2.0;
    assert!(
      (freqs[0] - f1).abs() <= tol,
      "first peak mismatch: {freqs:?}"
    );
    assert!(
      (freqs[1] - f2).abs() <= tol,
      "second peak mismatch: {freqs:?}"
    );
  }

  #[test]
  fn dominant_frequency_on_simd_normal_noise_is_finite() {
    let fs = 100.0;
    let n = 2048;

    let dist = SimdNormal::<f64>::new(0.0, 1.0);
    let mut rng = rand::rng();
    let mut x = vec![0.0; n];
    dist.fill_slice(&mut rng, &mut x);

    let cfg = SpectrumSearchConfig {
      periodogram: PeriodogramConfig {
        sampling_rate: fs,
        nfft: Some(2048),
        detrend: DetrendMethod::Mean,
        window: WindowFunction::Hann,
        onesided: true,
        scaling: SpectrumScaling::Density,
      },
      top_k: 1,
      min_frequency_hz: Some(0.0),
      max_frequency_hz: Some(fs / 2.0),
      min_separation_bins: 1,
      exclude_dc: false,
    };

    let peak = dominant_frequency_fft(&x, cfg).expect("expected at least one peak");
    assert!(peak.frequency_hz.is_finite());
    assert!(peak.power.is_finite() && peak.power >= 0.0);
    assert!((0.0..=fs / 2.0).contains(&peak.frequency_hz));
  }

  #[test]
  fn sunspot_series_has_about_eleven_year_cycle() {
    let raw = include_str!("data/sunspot_yearly_silso_v2.txt");
    let signal: Vec<f64> = raw
      .lines()
      .filter_map(|line| {
        let mut it = line.split_whitespace();
        let _year = it.next()?;
        let sn = it.next()?.parse::<f64>().ok()?;
        Some(sn)
      })
      .collect();

    assert!(
      signal.len() > 200,
      "unexpectedly short sunspot fixture: n={}",
      signal.len()
    );

    let cfg = SpectrumSearchConfig {
      periodogram: PeriodogramConfig {
        sampling_rate: 1.0, // annual data => cycles/year
        nfft: None,
        detrend: DetrendMethod::Mean,
        window: WindowFunction::Hann,
        onesided: true,
        scaling: SpectrumScaling::Power,
      },
      top_k: 6,
      min_frequency_hz: Some(1.0 / 30.0),
      max_frequency_hz: Some(1.0 / 5.0),
      min_separation_bins: 2,
      exclude_dc: true,
    };

    let peaks = fft_spectrum_search(&signal, cfg);
    assert!(
      !peaks.is_empty(),
      "no spectral peaks found on sunspot series"
    );

    let has_eleven_year_peak = peaks.iter().any(|p| {
      let period = 1.0 / p.frequency_hz;
      (period - 11.0).abs() <= 1.0
    });

    assert!(
      has_eleven_year_peak,
      "expected a peak near 11 years, got peaks={peaks:?}"
    );
  }
}
