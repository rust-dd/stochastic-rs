use ndarray::ArrayView1;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_distributions::DistributionSampler;
use stochastic_rs_distributions::SimdDistribution;
use stochastic_rs_distributions::normal::SimdNormal;

use super::DetrendMethod;
use super::PeriodogramConfig;
use super::SpectrumScaling;
use super::SpectrumSearchConfig;
use super::WindowFunction;
use super::apply_window_in_place;
use super::dominant_frequency_fft;
use super::fft_spectrum_search;
use super::periodogram_fft;
use crate::StatsError;

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
  let pg = periodogram_fft(ArrayView1::from(&x), cfg).unwrap();

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

  let peaks = fft_spectrum_search(ArrayView1::from(&x), cfg).unwrap();
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

  let mut dist = SimdNormal::<f64>::new(0.0, 1.0).seeded(&Deterministic::new(0x5ec7));
  let mut x = vec![0.0; n];
  dist.fill_slice(&mut x);

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

  let peak = dominant_frequency_fft(ArrayView1::from(&x), cfg)
    .unwrap()
    .expect("expected at least one peak");
  assert!(peak.frequency_hz.is_finite());
  assert!(peak.power.is_finite() && peak.power >= 0.0);
  assert!((0.0..=fs / 2.0).contains(&peak.frequency_hz));
}

#[test]
fn sunspot_series_has_about_eleven_year_cycle() {
  let raw = include_str!("../data/sunspot_yearly_silso_v2.txt");
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

  let peaks = fft_spectrum_search(ArrayView1::from(&signal), cfg).unwrap();
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

/// These finite points overflow their mean and NaN the taper: an `Err`, not a zero spectrum.
#[test]
fn a_spectrum_that_overflows_is_a_numerical_error() {
  let x = [1e308, 1e308, 1e308, 5e307];
  assert_eq!(
    periodogram_fft(ArrayView1::from(&x), PeriodogramConfig::default()).unwrap_err(),
    StatsError::Numerical {
      stage: "periodogram"
    }
  );
}

/// The four-point minimum leaves every taper positive energy, so the scale needs no floor.
#[test]
fn every_window_keeps_positive_energy_at_the_minimum_length() {
  for window in [
    WindowFunction::Rectangular,
    WindowFunction::Hann,
    WindowFunction::Hamming,
    WindowFunction::Blackman,
  ] {
    let energy = apply_window_in_place(&mut [1.0; 4], window);
    assert!(energy > 0.79, "{window:?}: {energy}");
  }
}

#[test]
#[should_panic(
  expected = "nfft must satisfy `nfft >= signal.len()`, got nfft = 8, signal.len() = 10"
)]
fn an_fft_shorter_than_the_signal_panics() {
  let cfg = PeriodogramConfig {
    nfft: Some(8),
    ..PeriodogramConfig::default()
  };
  let _ = periodogram_fft(ndarray::Array1::<f64>::linspace(0.0, 1.0, 10).view(), cfg);
}

/// The sample size is `nobs`, as everywhere in the crate: the signal's length, not `nfft`'s.
#[test]
fn the_periodogram_counts_its_points_as_nobs() {
  let x = ndarray::Array1::<f64>::linspace(0.0, 1.0, 10);
  let p = periodogram_fft(x.view(), PeriodogramConfig::default()).unwrap();
  assert_eq!((p.nobs, p.nfft), (10, 16));
}
