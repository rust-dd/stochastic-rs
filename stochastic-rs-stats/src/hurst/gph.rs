//! Geweke-Porter-Hudak (1983) log-periodogram regression.
//!
//! Procedure:
//! 1. (Optional) First-difference the input series so the regression
//!    operates on an FGN-like (zero-mean stationary) input.
//! 2. Compute the periodogram `I(λⱼ) = |Σₜ Z_t e^{-iλⱼt}|² / (2πn)`
//!    at the Fourier frequencies `λⱼ = 2πj/n`.
//! 3. Take the first `m = ⌊n^α⌋` frequencies (default `α = 0.6`).
//! 4. Regress `log I(λⱼ) = c - d · log(4 sin²(λⱼ/2)) + εⱼ`; the
//!    coefficient `d` is the fractional integration parameter.
//! 5. Map to Hurst: `H = d + ½`.
//!
//! Asymptotic standard error: `SE(d̂) = π / √(24m)` (Hurvich-Deo).
//!
//! Reference: Geweke J. & Porter-Hudak S. (1983) — *The estimation and
//! application of long memory time series models*, J. Time Series
//! Analysis 4(4), 221–238.

use std::f64::consts::PI;

use ndarray::ArrayView1;

use super::HurstDiagnostic;
use super::HurstError;
use super::HurstEstimator;
use super::HurstResult;
use super::to_f64_vec;
use super::weighted_linreg;
use crate::traits::FloatExt;

/// GPH log-periodogram Hurst estimator.
#[derive(Clone, Debug)]
pub struct Gph {
  /// Bandwidth exponent for `m = ⌊n^α⌋`; typical `α ∈ [0.5, 0.8]`.
  pub bandwidth_alpha: f64,
  /// When `true`, first-difference the input before estimation (use
  /// when the input is an FBM-like walk; leave `false` when it is
  /// already FGN-like noise).
  pub take_differences: bool,
}

impl Default for Gph {
  fn default() -> Self {
    Self {
      bandwidth_alpha: 0.6,
      take_differences: true,
    }
  }
}

impl Gph {
  #[must_use]
  pub fn new() -> Self {
    Self::default()
  }
}

impl<T: FloatExt> HurstEstimator<T> for Gph {
  fn estimate(&self, x: ArrayView1<T>) -> Result<HurstResult<T>, HurstError> {
    if !(0.1..1.0).contains(&self.bandwidth_alpha) {
      return Err(HurstError::InvalidParameter(
        "bandwidth_alpha",
        self.bandwidth_alpha,
      ));
    }
    let n_obs = x.len();
    if n_obs < 64 {
      return Err(HurstError::TooFewObservations {
        got: n_obs,
        required: 64,
      });
    }

    let xs = to_f64_vec::<T>(x);
    let series: Vec<f64> = if self.take_differences {
      xs.windows(2).map(|w| w[1] - w[0]).collect()
    } else {
      xs
    };
    let n = series.len();
    if n < 32 {
      return Err(HurstError::TooFewObservations { got: n, required: 32 });
    }
    let mean = series.iter().sum::<f64>() / n as f64;
    let demeaned: Vec<f64> = series.iter().map(|v| *v - mean).collect();

    let m = ((n as f64).powf(self.bandwidth_alpha).floor() as usize).max(8).min(n / 2);
    if m < 4 {
      return Err(HurstError::NotEnoughScales);
    }

    let mut log_freq = Vec::with_capacity(m);
    let mut log_periodogram = Vec::with_capacity(m);
    for j in 1..=m {
      let lambda = 2.0 * PI * (j as f64) / (n as f64);
      let mut cr = 0.0;
      let mut ci = 0.0;
      for (t, &v) in demeaned.iter().enumerate() {
        let phase = lambda * (t as f64);
        cr += v * phase.cos();
        ci += v * phase.sin();
      }
      let i_lambda = (cr * cr + ci * ci) / (2.0 * PI * n as f64);
      let sin_half = (lambda * 0.5).sin();
      let regressor = 4.0 * sin_half * sin_half;
      if !(i_lambda > 0.0 && regressor > 0.0 && i_lambda.is_finite()) {
        continue;
      }
      log_freq.push(regressor.ln());
      log_periodogram.push(i_lambda.ln());
    }
    if log_freq.len() < 4 {
      return Err(HurstError::NotEnoughScales);
    }

    let m_used = log_freq.len();
    let (slope, intercept, r_squared) =
      weighted_linreg(&log_freq, &log_periodogram, None).ok_or(HurstError::RegressionFailed)?;
    if !slope.is_finite() {
      return Err(HurstError::RegressionFailed);
    }
    let d = -slope;
    let h = d + 0.5;
    let std_err = PI / (24.0 * m_used as f64).sqrt();

    Ok(HurstResult {
      hurst: T::from_f64_fast(h),
      std_err: Some(T::from_f64_fast(std_err)),
      n_obs,
      diagnostic: HurstDiagnostic::LogLogRegression {
        slope: T::from_f64_fast(slope),
        intercept: T::from_f64_fast(intercept),
        r_squared: T::from_f64_fast(r_squared),
        log_scales: log_freq.into_iter().map(T::from_f64_fast).collect(),
        log_stats: log_periodogram.into_iter().map(T::from_f64_fast).collect(),
      },
    })
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use stochastic_rs_core::simd_rng::Unseeded;
  use stochastic_rs_stochastic::process::fbm::Fbm;

  use crate::traits::ProcessExt;

  #[test]
  fn gph_matches_known_h_on_fbm() {
    let h = 0.7_f64;
    let m_paths = 24;
    let mut acc = 0.0;
    for _ in 0..m_paths {
      let fbm = Fbm::new(h, 8192, Some(1.0), Unseeded);
      let path = fbm.sample();
      let r = Gph::default().estimate(path.view()).expect("gph");
      acc += r.hurst;
    }
    let h_est = acc / m_paths as f64;
    assert!(
      (h_est - h).abs() < 0.08,
      "GPH H={h_est:.3}, expected {h:.3}"
    );
  }

  #[test]
  fn gph_returns_finite_std_err() {
    let fbm = Fbm::new(0.5_f64, 4096, Some(1.0), Unseeded);
    let path = fbm.sample();
    let r = Gph::default().estimate(path.view()).expect("gph");
    let se = r.std_err.expect("std_err present");
    assert!(se > 0.0 && se < 1.0, "SE={se}");
  }

  #[test]
  fn gph_rejects_too_short() {
    let path = ndarray::Array1::<f64>::zeros(32);
    let r = Gph::default().estimate(path.view());
    assert!(matches!(r, Err(HurstError::TooFewObservations { .. })));
  }
}
