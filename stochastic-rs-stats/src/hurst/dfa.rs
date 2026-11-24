//! Detrended fluctuation analysis (Peng et al. 1994): the log-log slope of the detrended profile.
//!
//! $$
//! Y(i) = \sum_{k \le i} (X_k - \bar X),\qquad
//! F^2(s) = \frac{1}{N_s}\sum_{v=1}^{N_s}\frac1s\sum_{i=1}^{s}\bigl(Y_v(i) - p_v(i)\bigr)^2,\qquad
//! \log F(s) \sim \alpha \log s
//! $$
//!
//! `p_v` is segment `v`'s degree-`order` fit; how `α` maps to `H` is on `assume_integrated`.
//!
//! Reference: Peng et al. (1994), Phys. Rev. E 49(2), 1685–1689, DOI: 10.1103/PhysRevE.49.1685
//! Reference: Kantelhardt et al. (2001), Physica A 295, 441–454, DOI: 10.1016/S0378-4371(01)00144-3

use ndarray::ArrayView1;

use super::HurstDiagnostic;
use super::HurstEstimator;
use super::HurstFit;
use super::log_spaced_windows;
use super::to_f64_vec;
use super::weighted_linreg;
use crate::StatsError;
use crate::input::reject_constant;
use crate::stationarity::common::about_mean;
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
  /// `true`: the input is already an integrated, FBM-like walk and `H = α`; `false` (default): the
  /// estimator integrates it, and `H = α - 1` for `α > 1`, else `H = α`.
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
  fn estimate(&self, x: ArrayView1<T>) -> Result<HurstFit<T>, StatsError> {
    let n = x.len();
    if self.order < 1 {
      return Err(StatsError::InvalidParameter {
        name: "order",
        value: self.order as f64,
        constraint: "order >= 1",
      });
    }
    if !(0.0..1.0).contains(&self.overlap_pct) {
      return Err(StatsError::InvalidParameter {
        name: "overlap_pct",
        value: self.overlap_pct,
        constraint: "0.0 <= overlap_pct < 1.0",
      });
    }
    if self.min_window < self.order + 2 {
      return Err(StatsError::InvalidParameter {
        name: "min_window",
        value: self.min_window as f64,
        constraint: "min_window >= order + 2",
      });
    }
    // Unset, the largest window is `n / 4`, which must exceed `min_window`.
    let windows = self
      .min_window
      .saturating_add(usize::from(self.max_window.is_none()));
    let min_required = windows.saturating_mul(4).max(self.order.saturating_add(4));
    if n < min_required {
      return Err(StatsError::InsufficientData {
        needed: min_required,
        got: n,
      });
    }
    if let Some(max_window) = self.max_window
      && max_window <= self.min_window
    {
      return Err(StatsError::InvalidParameter {
        name: "max_window",
        value: max_window as f64,
        constraint: "max_window > min_window",
      });
    }

    let max_window = self.max_window.unwrap_or(n / 4).min(n / 2);
    let windows = log_spaced_windows(self.min_window, max_window, self.n_windows);
    if windows.len() < 2 {
      return Err(StatsError::DegenerateData("fewer than two valid scales"));
    }

    let xs = to_f64_vec::<T>(x);
    reject_constant(&xs, "constant series")?;
    let (xs, units) = about_mean(xs);
    let in_units = units[0].ln() + units[1].ln();
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
      return Err(StatsError::DegenerateData("fewer than two valid scales"));
    }

    let (alpha, intercept, r_squared) =
      weighted_linreg(&log_s, &log_f, None).ok_or(StatsError::Singular("log-log regression"))?;
    if !alpha.is_finite() {
      return Err(StatsError::Singular("log-log regression"));
    }

    let h = if self.assume_integrated {
      alpha
    } else if alpha > 1.0 {
      alpha - 1.0
    } else {
      alpha
    };

    Ok(HurstFit {
      hurst: T::from_f64_fast(h),
      std_err: None,
      nobs: n,
      diagnostic: HurstDiagnostic::LogLogRegression {
        slope: T::from_f64_fast(alpha),
        intercept: T::from_f64_fast(intercept - in_units),
        r_squared: T::from_f64_fast(r_squared),
        log_scales: log_s.into_iter().map(T::from_f64_fast).collect(),
        log_stats: log_f
          .iter()
          .map(|f| T::from_f64_fast(f - in_units))
          .collect(),
      },
    })
  }
}

/// Residual sum of squares of `segment` about its least-squares polynomial of degree `order`, by
/// the Vandermonde normal equations.
fn detrend_rss(segment: &[f64], order: usize) -> Option<f64> {
  let n = segment.len();
  if n < order + 2 {
    return None;
  }
  let m = order + 1;

  // Build normal equations A·c = b where A_{jk} = Σ t^{j+k}, b_j = Σ y · t^j.
  let mut a = vec![0.0_f64; m * m];
  let mut b = vec![0.0_f64; m];
  for (i, &y) in segment.iter().enumerate() {
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
      b[j] += y * pow_row[j];
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
  use ndarray::Array1;
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

  #[test]
  fn the_window_range_folds_into_the_length_check() {
    let x = Array1::<f64>::linspace(0.0, 1.0, 34);
    assert_eq!(
      HurstEstimator::<f64>::estimate(&Dfa::default(), x.view()).unwrap_err(),
      StatsError::InsufficientData {
        needed: 36,
        got: 34
      }
    );
    let at_min = Dfa {
      max_window: Some(8),
      ..Dfa::default()
    };
    let long = Array1::<f64>::linspace(0.0, 1.0, 200);
    assert!(matches!(
      HurstEstimator::<f64>::estimate(&at_min, long.view()),
      Err(StatsError::InvalidParameter {
        name: "max_window",
        ..
      })
    ));
  }
}
