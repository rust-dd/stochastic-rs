//! Rescaled-range Hurst estimator with Anis-Lloyd (1976) bias
//! correction.
//!
//! Procedure (Weron 2002, §3.1):
//! 1. For each log-spaced window size `n_w`, split the (zero-mean,
//!    demeaned) path into `floor(N / n_w)` non-overlapping chunks.
//! 2. In each chunk compute the rescaled range
//!    `R/S = (max Y_k - min Y_k) / s` where `Y_k = Σᵢ≤k (X_i - <X>)`
//!    and `s` is the (biased) chunk standard deviation.
//! 3. Average across chunks.  When `anis_lloyd = true` replace
//!    `(R/S)_obs` by `(R/S)_obs - E[R/S]_iid + √(π·n_w/2)`.
//! 4. Linear regression of `log(R/S)_corrected` on `log n_w`; slope =
//!    Hurst exponent.
//!
//! Closed-form `E[R/S]_iid`:
//!
//! - `n_w ≤ 340`:
//!   `((n-½)/n) · Γ((n-1)/2)/(√π · Γ(n/2)) · Σᵢ₌₁ⁿ⁻¹ √((n-i)/i)`
//! - `n_w > 340`:
//!   `((n-½)/n) · √(2/(nπ)) · Σᵢ₌₁ⁿ⁻¹ √((n-i)/i)`
//!
//! References:
//! - Hurst H. E. (1951) — *Long-term storage capacity of reservoirs*,
//!   Trans. ASCE 116, 770–799.
//! - Anis A. A. & Lloyd E. H. (1976) — *The expected value of the
//!   adjusted rescaled Hurst range of independent normal summands*,
//!   Biometrika 63(1), 111–116.
//! - Weron R. (2002) — *Estimating long-range dependence: finite sample
//!   properties and confidence intervals*, Physica A 312, 285–299.

use ndarray::ArrayView1;
use stochastic_rs_distributions::special::ln_gamma;

use super::HurstDiagnostic;
use super::HurstError;
use super::HurstEstimator;
use super::HurstResult;
use super::log_spaced_windows;
use super::to_f64_vec;
use super::weighted_linreg;
use crate::traits::FloatExt;

/// Rescaled-range Hurst estimator.
#[derive(Clone, Debug)]
pub struct RescaledRange {
  /// Apply the Anis-Lloyd (1976) bias correction.
  pub anis_lloyd: bool,
  /// First-difference the input before estimation.  Default `true`
  /// because R/S is defined on stationary increments (FGN-like): when
  /// the caller hands in an FBM-like walk we must difference first.
  /// Pass `false` when the input is already a stationary returns /
  /// noise series.
  pub take_differences: bool,
  /// Smallest window in the log-log regression.
  pub min_window: usize,
  /// Largest window in the log-log regression (`None` → `N / 4`).
  pub max_window: Option<usize>,
  /// Number of log-spaced windows.
  pub n_windows: usize,
}

impl Default for RescaledRange {
  fn default() -> Self {
    Self {
      anis_lloyd: true,
      take_differences: true,
      min_window: 16,
      max_window: None,
      n_windows: 30,
    }
  }
}

impl RescaledRange {
  #[must_use]
  pub fn new() -> Self {
    Self::default()
  }
}

impl<T: FloatExt> HurstEstimator<T> for RescaledRange {
  fn estimate(&self, x: ArrayView1<T>) -> Result<HurstResult<T>, HurstError> {
    let n = x.len();
    let min_required = self.min_window.max(8) * 2;
    if n < min_required {
      return Err(HurstError::TooFewObservations {
        got: n,
        required: min_required,
      });
    }
    if self.min_window < 4 {
      return Err(HurstError::InvalidParameter(
        "min_window",
        self.min_window as f64,
      ));
    }
    if self.n_windows < 4 {
      return Err(HurstError::InvalidParameter(
        "n_windows",
        self.n_windows as f64,
      ));
    }

    let max_window = self.max_window.unwrap_or(n / 4).min(n / 2);
    if max_window <= self.min_window {
      return Err(HurstError::InvalidParameter(
        "max_window",
        max_window as f64,
      ));
    }
    let windows = log_spaced_windows(self.min_window, max_window, self.n_windows);
    if windows.len() < 2 {
      return Err(HurstError::NotEnoughScales);
    }

    let xs_raw = to_f64_vec::<T>(x);
    let xs: Vec<f64> = if self.take_differences {
      xs_raw.windows(2).map(|w| w[1] - w[0]).collect()
    } else {
      xs_raw
    };
    let n_series = xs.len();
    let mut log_n = Vec::with_capacity(windows.len());
    let mut log_rs = Vec::with_capacity(windows.len());

    for &nw in &windows {
      let n_chunks = n_series / nw;
      if n_chunks == 0 {
        continue;
      }
      let mut rs_acc = 0.0;
      let mut rs_count = 0usize;
      for c in 0..n_chunks {
        let chunk = &xs[c * nw..(c + 1) * nw];
        if let Some(rs) = chunk_rs(chunk) {
          rs_acc += rs;
          rs_count += 1;
        }
      }
      if rs_count == 0 {
        continue;
      }
      let mut rs_mean = rs_acc / rs_count as f64;

      if self.anis_lloyd {
        let expected = expected_rs_iid(nw);
        let asymptotic = (std::f64::consts::PI * nw as f64 * 0.5).sqrt();
        rs_mean = rs_mean - expected + asymptotic;
        if rs_mean <= 0.0 {
          continue;
        }
      }
      log_n.push((nw as f64).ln());
      log_rs.push(rs_mean.ln());
    }

    if log_n.len() < 2 {
      return Err(HurstError::NotEnoughScales);
    }

    let (slope, intercept, r_squared) =
      weighted_linreg(&log_n, &log_rs, None).ok_or(HurstError::RegressionFailed)?;
    if !slope.is_finite() {
      return Err(HurstError::RegressionFailed);
    }

    Ok(HurstResult {
      hurst: T::from_f64_fast(slope),
      std_err: None,
      n_obs: n,
      diagnostic: HurstDiagnostic::LogLogRegression {
        slope: T::from_f64_fast(slope),
        intercept: T::from_f64_fast(intercept),
        r_squared: T::from_f64_fast(r_squared),
        log_scales: log_n.into_iter().map(T::from_f64_fast).collect(),
        log_stats: log_rs.into_iter().map(T::from_f64_fast).collect(),
      },
    })
  }
}

/// Single-chunk biased rescaled range.
fn chunk_rs(chunk: &[f64]) -> Option<f64> {
  let n = chunk.len();
  if n < 4 {
    return None;
  }
  let n_f = n as f64;
  let mean = chunk.iter().sum::<f64>() / n_f;
  let mut var = 0.0;
  for &v in chunk {
    var += (v - mean).powi(2);
  }
  var /= n_f;
  let std = var.sqrt();
  if !(std > 0.0 && std.is_finite()) {
    return None;
  }
  let mut acc = 0.0;
  let mut max = f64::NEG_INFINITY;
  let mut min = f64::INFINITY;
  for &v in chunk {
    acc += v - mean;
    if acc > max {
      max = acc;
    }
    if acc < min {
      min = acc;
    }
  }
  let r = max - min;
  if !(r > 0.0 && r.is_finite()) {
    return None;
  }
  Some(r / std)
}

/// Anis-Lloyd (1976) closed-form expectation `E[R/S | iid normal]`.
pub(crate) fn expected_rs_iid(n: usize) -> f64 {
  let pi = std::f64::consts::PI;
  let n_f = n as f64;
  let sum: f64 = (1..n).map(|i| ((n - i) as f64 / i as f64).sqrt()).sum();
  let prefactor = if n <= 340 {
    let ln_num = ln_gamma(0.5 * (n_f - 1.0));
    let ln_den = 0.5 * pi.ln() + ln_gamma(0.5 * n_f);
    (n_f - 0.5) / n_f * (ln_num - ln_den).exp()
  } else {
    (n_f - 0.5) / n_f * (2.0 / (n_f * pi)).sqrt()
  };
  prefactor * sum
}

#[cfg(test)]
mod tests {
  use ndarray::Array1;
  use rand::SeedableRng;
  use rand::rngs::StdRng;
  use rand_distr::Distribution;
  use rand_distr::StandardNormal;
  use stochastic_rs_core::simd_rng::Unseeded;
  use stochastic_rs_stochastic::process::fbm::Fbm;

  use super::*;
  use crate::traits::ProcessExt;

  #[test]
  fn expected_rs_table_match() {
    let r10 = expected_rs_iid(10);
    let r100 = expected_rs_iid(100);
    let r1000 = expected_rs_iid(1000);
    assert!(r10 > 1.5 && r10 < 4.5, "E[R/S](10) = {r10}");
    assert!(r100 > 9.0 && r100 < 13.5, "E[R/S](100) = {r100}");
    assert!(r1000 > 35.0 && r1000 < 42.0, "E[R/S](1000) = {r1000}");
  }

  #[test]
  fn anis_lloyd_corrects_iid_bias() {
    let mut rng = StdRng::seed_from_u64(42);
    let n = 2048_usize;
    let x: Vec<f64> = (0..n).map(|_| StandardNormal.sample(&mut rng)).collect();
    let view = Array1::from_vec(x);

    // iid Gaussian is stationary noise — pass take_differences=false so
    // we don't antipersistent-ify it via first-diff
    let naive = RescaledRange {
      anis_lloyd: false,
      take_differences: false,
      ..Default::default()
    }
    .estimate(view.view())
    .expect("naive RS");
    let corrected = RescaledRange {
      anis_lloyd: true,
      take_differences: false,
      ..Default::default()
    }
    .estimate(view.view())
    .expect("corrected RS");

    // for iid the true H = 0.5; Anis-Lloyd should pull the estimate close
    assert!(
      (corrected.hurst - 0.5).abs() < 0.08,
      "AL corrected H={:.3} too far from 0.5 (naive={:.3})",
      corrected.hurst,
      naive.hurst,
    );
  }

  #[test]
  fn rs_matches_known_h_on_fbm() {
    let h = 0.7_f64;
    let m = 16;
    let mut acc = 0.0;
    for _ in 0..m {
      let fbm = Fbm::new(h, 8192, Some(1.0), Unseeded);
      let path = fbm.sample();
      let r = RescaledRange::default()
        .estimate(path.view())
        .expect("rs estimator");
      acc += r.hurst;
    }
    let h_est = acc / m as f64;
    assert!(
      (h_est - h).abs() < 0.08,
      "R/S H={h_est:.3}, expected {h:.3}"
    );
  }
}
