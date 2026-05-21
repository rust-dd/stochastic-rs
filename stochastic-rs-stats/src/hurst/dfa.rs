//! Detrended Fluctuation Analysis (Peng et al. 1994).
//!
//! Procedure:
//! 1. Integrate the (demeaned) input series: `Y(i) = Σ_{k≤i}(X_k - <X>)`.
//! 2. For each log-spaced window size `s ∈ [min, max]`, slide
//!    `N_s ≈ N/s` (optionally overlapping) segments of length `s`.
//! 3. In each segment fit a polynomial of degree `order` and compute
//!    the residual variance; aggregate across segments to obtain
//!    `F²(s) = (1/N_s) Σᵥ (1/s) Σᵢ (Y_v(i) - p_v(i))²`.
//! 4. Log-log regression `log F(s) ~ α · log s` over the chosen
//!    window range; `α` is the DFA exponent.
//! 5. For a series that has *not* been pre-integrated by the user
//!    (`assume_integrated = false`): `H = α - 1` when `α > 1`, else
//!    `H = α`.  For an FGN-like noise input (`assume_integrated =
//!    true`): `H = α` directly.
//!
//! Reference: Peng C.-K., Buldyrev S. V., Havlin S., Simons M.,
//! Stanley H. E., Goldberger A. L. (1994) — *Mosaic organization of
//! DNA nucleotides*, Phys. Rev. E 49(2), 1685–1689,
//! DOI: 10.1103/PhysRevE.49.1685.
//!
//! Subsequent refinement: Kantelhardt J. W. et al. (2001) — *Detecting
//! long-range correlations with detrended fluctuation analysis*,
//! Physica A 295, 441–454.

use ndarray::ArrayView1;

use super::HurstDiagnostic;
use super::HurstError;
use super::HurstEstimator;
use super::HurstResult;
use super::log_spaced_windows;
use super::to_f64_vec;
use super::weighted_linreg;
use crate::traits::FloatExt;

/// Detrended fluctuation analysis Hurst estimator.
#[derive(Clone, Debug)]
pub struct Dfa {
  /// Polynomial detrending order (1 = linear, 2 = quadratic, …).
  pub order: usize,
  /// Smallest window in the log-log regression (default 8).
  pub min_window: usize,
  /// Largest window in the log-log regression (`None` → `N / 4`).
  pub max_window: Option<usize>,
  /// Number of log-spaced windows.
  pub n_windows: usize,
  /// Segment overlap, in `[0, 1)`.  `0.0` = no overlap (Peng 1994),
  /// `0.5` = 50% overlap (common refinement).
  pub overlap_pct: f64,
  /// When `true` the caller has already integrated their FGN-like
  /// noise into an FBM-like walk and `H = α` directly; when `false`
  /// (default) the estimator integrates internally and applies
  /// `H = α - 1` for `α > 1`.
  pub assume_integrated: bool,
}

impl Default for Dfa {
  fn default() -> Self {
    Self {
      order: 1,
      min_window: 8,
      max_window: None,
      n_windows: 24,
      overlap_pct: 0.0,
      assume_integrated: false,
    }
  }
}

impl Dfa {
  #[must_use]
  pub fn new(order: usize) -> Self {
    Self {
      order,
      ..Self::default()
    }
  }
}

impl<T: FloatExt> HurstEstimator<T> for Dfa {
  fn estimate(&self, x: ArrayView1<T>) -> Result<HurstResult<T>, HurstError> {
    let n = x.len();
    if self.order < 1 {
      return Err(HurstError::InvalidParameter("order", self.order as f64));
    }
    if !(0.0..1.0).contains(&self.overlap_pct) {
      return Err(HurstError::InvalidParameter(
        "overlap_pct",
        self.overlap_pct,
      ));
    }
    if self.min_window < self.order + 2 {
      return Err(HurstError::InvalidParameter(
        "min_window",
        self.min_window as f64,
      ));
    }
    let min_required = (self.min_window * 4).max(self.order + 4);
    if n < min_required {
      return Err(HurstError::TooFewObservations {
        got: n,
        required: min_required,
      });
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

    let xs = to_f64_vec::<T>(x);
    let mean: f64 = xs.iter().sum::<f64>() / xs.len() as f64;

    // Integrate (or skip integration when the caller already did so).
    let profile: Vec<f64> = if self.assume_integrated {
      xs.clone()
    } else {
      let mut p = Vec::with_capacity(xs.len());
      let mut acc = 0.0_f64;
      for &v in &xs {
        acc += v - mean;
        p.push(acc);
      }
      p
    };

    let stride_for = |s: usize| -> usize {
      if self.overlap_pct <= 0.0 {
        s
      } else {
        ((s as f64) * (1.0 - self.overlap_pct)).round().max(1.0) as usize
      }
    };

    let mut log_s = Vec::with_capacity(windows.len());
    let mut log_f = Vec::with_capacity(windows.len());

    for &s in &windows {
      let stride = stride_for(s);
      if profile.len() < s {
        continue;
      }
      let n_segments = (profile.len() - s) / stride + 1;
      if n_segments == 0 {
        continue;
      }

      let mut f2_acc = 0.0_f64;
      let mut count = 0usize;
      for seg_idx in 0..n_segments {
        let start = seg_idx * stride;
        let segment = &profile[start..start + s];
        if let Some(rss) = detrend_rss(segment, self.order) {
          f2_acc += rss / s as f64;
          count += 1;
        }
      }
      if count == 0 {
        continue;
      }
      let f_s = (f2_acc / count as f64).sqrt();
      if !(f_s > 0.0 && f_s.is_finite()) {
        continue;
      }
      log_s.push((s as f64).ln());
      log_f.push(f_s.ln());
    }

    if log_s.len() < 2 {
      return Err(HurstError::NotEnoughScales);
    }

    let (alpha, intercept, r_squared) =
      weighted_linreg(&log_s, &log_f, None).ok_or(HurstError::RegressionFailed)?;
    if !alpha.is_finite() {
      return Err(HurstError::RegressionFailed);
    }

    let h = if self.assume_integrated {
      alpha
    } else if alpha > 1.0 {
      alpha - 1.0
    } else {
      alpha
    };

    Ok(HurstResult {
      hurst: T::from_f64_fast(h),
      std_err: None,
      n_obs: n,
      diagnostic: HurstDiagnostic::LogLogRegression {
        slope: T::from_f64_fast(alpha),
        intercept: T::from_f64_fast(intercept),
        r_squared: T::from_f64_fast(r_squared),
        log_scales: log_s.into_iter().map(T::from_f64_fast).collect(),
        log_stats: log_f.into_iter().map(T::from_f64_fast).collect(),
      },
    })
  }
}

/// Residual sum of squares after polynomial detrending of `segment`
/// by a polynomial of `order`.  Uses normal equations (Vandermonde
/// least squares).
fn detrend_rss(segment: &[f64], order: usize) -> Option<f64> {
  let n = segment.len();
  if n < order + 2 {
    return None;
  }
  let m = order + 1;

  // Build normal equations A·c = b where A_{jk} = Σ t^{j+k}, b_j = Σ y · t^j.
  let mut a = vec![0.0_f64; m * m];
  let mut b = vec![0.0_f64; m];
  for i in 0..n {
    let t = i as f64;
    let mut pow_jk = 1.0_f64;
    let mut pow_row = vec![0.0_f64; 2 * m - 1];
    for entry in pow_row.iter_mut() {
      *entry = pow_jk;
      pow_jk *= t;
    }
    for j in 0..m {
      for k in 0..m {
        a[j * m + k] += pow_row[j + k];
      }
      b[j] += segment[i] * pow_row[j];
    }
  }
  let coeffs = solve_symmetric(&mut a, &mut b, m)?;

  let mut rss = 0.0_f64;
  for (i, y) in segment.iter().enumerate() {
    let t = i as f64;
    let mut fit = 0.0_f64;
    let mut t_pow = 1.0_f64;
    for c in &coeffs {
      fit += *c * t_pow;
      t_pow *= t;
    }
    let r = *y - fit;
    rss += r * r;
  }
  if rss.is_finite() { Some(rss) } else { None }
}

/// Gauss-Jordan solve of `A·x = b` with `A` symmetric positive-definite
/// (or at least non-singular).  Returns the solution in place of `b`.
fn solve_symmetric(a: &mut [f64], b: &mut [f64], n: usize) -> Option<Vec<f64>> {
  for i in 0..n {
    let mut pivot_row = i;
    let mut pivot_val = a[i * n + i].abs();
    for k in (i + 1)..n {
      let v = a[k * n + i].abs();
      if v > pivot_val {
        pivot_val = v;
        pivot_row = k;
      }
    }
    if pivot_val < 1e-12 {
      return None;
    }
    if pivot_row != i {
      for j in 0..n {
        a.swap(i * n + j, pivot_row * n + j);
      }
      b.swap(i, pivot_row);
    }
    let pivot = a[i * n + i];
    for j in 0..n {
      a[i * n + j] /= pivot;
    }
    b[i] /= pivot;
    for k in 0..n {
      if k == i {
        continue;
      }
      let factor = a[k * n + i];
      if factor == 0.0 {
        continue;
      }
      for j in 0..n {
        a[k * n + j] -= factor * a[i * n + j];
      }
      b[k] -= factor * b[i];
    }
  }
  Some(b.to_vec())
}

#[cfg(test)]
mod tests {
  use stochastic_rs_core::simd_rng::Unseeded;
  use stochastic_rs_stochastic::process::fbm::Fbm;

  use super::*;
  use crate::traits::ProcessExt;

  #[test]
  fn dfa_matches_known_h_on_fbm() {
    let h = 0.7_f64;
    let m = 16;
    let mut acc = 0.0;
    for _ in 0..m {
      let fbm = Fbm::new(h, 8192, Some(1.0), Unseeded);
      let path = fbm.sample();
      let r = Dfa {
        assume_integrated: true,
        ..Default::default()
      }
      .estimate(path.view())
      .expect("dfa");
      acc += r.hurst;
    }
    let h_est = acc / m as f64;
    assert!(
      (h_est - h).abs() < 0.08,
      "DFA-1 H={h_est:.3}, expected {h:.3}"
    );
  }

  #[test]
  fn dfa_quadratic_detrending_works() {
    let h = 0.3_f64;
    let m = 16;
    let mut acc = 0.0;
    for _ in 0..m {
      let fbm = Fbm::new(h, 8192, Some(1.0), Unseeded);
      let path = fbm.sample();
      let r = Dfa {
        order: 2,
        assume_integrated: true,
        ..Default::default()
      }
      .estimate(path.view())
      .expect("dfa-2");
      acc += r.hurst;
    }
    let h_est = acc / m as f64;
    assert!(
      (h_est - h).abs() < 0.10,
      "DFA-2 H={h_est:.3}, expected {h:.3}"
    );
  }

  #[test]
  fn detrend_constant_segment_zero_rss() {
    let seg = vec![3.0_f64; 16];
    let rss = detrend_rss(&seg, 1).unwrap();
    assert!(
      rss.abs() < 1e-10,
      "constant detrend should leave 0 rss, got {rss}"
    );
  }

  #[test]
  fn detrend_linear_segment_zero_rss() {
    let seg: Vec<f64> = (0..16).map(|i| 2.0 + 0.5 * i as f64).collect();
    let rss = detrend_rss(&seg, 1).unwrap();
    assert!(
      rss.abs() < 1e-10,
      "linear detrend should leave 0 rss, got {rss}"
    );
  }
}
