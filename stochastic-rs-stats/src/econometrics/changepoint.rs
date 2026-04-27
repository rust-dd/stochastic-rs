//! Changepoint detection — CUSUM (Page 1954) and PELT (Killick et al. 2012).
//!
//! - **CUSUM** detects mean shifts via cumulative sums of standardised
//!   deviations and signals when one of two recursive sums exceeds a
//!   threshold.
//! - **PELT** finds the global optimum of a penalised likelihood
//!   objective with at most linear cost in the sample size.
//!
//! Reference: Page, "Continuous Inspection Schemes", Biometrika, 41(1/2),
//! 100-115 (1954). DOI: 10.2307/2333009
//!
//! Reference: Killick, Fearnhead, Eckley, "Optimal Detection of Changepoints
//! with a Linear Computational Cost", Journal of the American Statistical
//! Association, 107(500), 1590-1598 (2012).
//! DOI: 10.1080/01621459.2012.737745

use ndarray::Array1;
use ndarray::ArrayView1;

use crate::traits::FloatExt;

/// Result of a CUSUM control-chart pass.
#[derive(Debug, Clone)]
pub struct CusumResult {
  /// Upper-side CUSUM statistic $S_t^+$.
  pub upper: Array1<f64>,
  /// Lower-side CUSUM statistic $S_t^-$.
  pub lower: Array1<f64>,
  /// Indices `t` at which $S_t^+ > h$ or $S_t^- < -h$.
  pub alarms: Vec<usize>,
}

/// One-sided / two-sided CUSUM with reference value `k` (half the smallest
/// shift to detect, in standard-deviation units) and threshold `h`.
///
/// `series` is standardised in-place: pass already-standardised residuals or a
/// raw series whose mean and standard deviation will be subtracted.
pub fn cusum<T: FloatExt>(series: ArrayView1<T>, k: f64, h: f64) -> CusumResult {
  let n = series.len();
  if n == 0 {
    return CusumResult {
      upper: Array1::zeros(0),
      lower: Array1::zeros(0),
      alarms: Vec::new(),
    };
  }
  let mut x = Array1::<f64>::zeros(n);
  for i in 0..n {
    x[i] = series[i].to_f64().unwrap();
  }
  let mean = x.iter().sum::<f64>() / n as f64;
  let var = x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n as f64 - 1.0).max(1.0);
  let sd = var.sqrt().max(1e-12);
  let mut upper = Array1::<f64>::zeros(n);
  let mut lower = Array1::<f64>::zeros(n);
  let mut alarms = Vec::new();
  let mut s_plus = 0.0;
  let mut s_minus = 0.0;
  for i in 0..n {
    let z = (x[i] - mean) / sd;
    s_plus = (s_plus + z - k).max(0.0);
    s_minus = (s_minus + z + k).min(0.0);
    upper[i] = s_plus;
    lower[i] = s_minus;
    if s_plus > h || s_minus < -h {
      alarms.push(i);
    }
  }
  CusumResult {
    upper,
    lower,
    alarms,
  }
}

/// Result of a PELT pass.
#[derive(Debug, Clone)]
pub struct PeltResult {
  /// Sorted changepoint indices (exclusive end of each segment, except the
  /// last which is `n`).
  pub changepoints: Vec<usize>,
  /// Total cost at the optimum.
  pub cost: f64,
}

/// PELT (Pruned Exact Linear Time) changepoint detection on a univariate
/// series with the squared-error mean-shift cost
/// $\mathcal C(y_{a:b}) = \sum_{t=a}^{b-1}(y_t - \bar y_{a:b})^2$
/// and per-changepoint penalty `penalty`.
pub fn pelt<T: FloatExt>(series: ArrayView1<T>, penalty: f64, min_size: usize) -> PeltResult {
  let n = series.len();
  if n == 0 {
    return PeltResult {
      changepoints: Vec::new(),
      cost: 0.0,
    };
  }
  let min_size = min_size.max(1);
  let mut prefix = vec![0.0_f64; n + 1];
  let mut prefix_sq = vec![0.0_f64; n + 1];
  for i in 0..n {
    let v = series[i].to_f64().unwrap();
    prefix[i + 1] = prefix[i] + v;
    prefix_sq[i + 1] = prefix_sq[i] + v * v;
  }
  let segment_cost = |a: usize, b: usize| -> f64 {
    if b <= a {
      return 0.0;
    }
    let len = (b - a) as f64;
    let s = prefix[b] - prefix[a];
    let s2 = prefix_sq[b] - prefix_sq[a];
    s2 - s * s / len
  };
  let mut f = vec![f64::INFINITY; n + 1];
  let mut prev = vec![0_usize; n + 1];
  f[0] = -penalty;
  let mut candidates: Vec<usize> = vec![0];
  for end in min_size..=n {
    let mut best = f64::INFINITY;
    let mut arg = 0usize;
    for &start in &candidates {
      if end < start + min_size {
        continue;
      }
      let cost = f[start] + segment_cost(start, end) + penalty;
      if cost < best {
        best = cost;
        arg = start;
      }
    }
    f[end] = best;
    prev[end] = arg;
    let mut next_candidates = Vec::with_capacity(candidates.len() + 1);
    for &start in &candidates {
      if end < start + min_size {
        next_candidates.push(start);
        continue;
      }
      if f[start] + segment_cost(start, end) <= f[end] {
        next_candidates.push(start);
      }
    }
    next_candidates.push(end);
    candidates = next_candidates;
  }
  let mut cps = Vec::new();
  let mut current = n;
  while current > 0 {
    let p = prev[current];
    if p > 0 {
      cps.push(p);
    }
    if p == current {
      break;
    }
    current = p;
  }
  cps.reverse();
  PeltResult {
    changepoints: cps,
    cost: f[n],
  }
}

#[cfg(test)]
mod tests {
  use ndarray::Array1;
  use stochastic_rs_distributions::normal::SimdNormal;

  use super::*;

  #[test]
  fn cusum_few_alarms_under_pure_noise() {
    let dist = SimdNormal::<f64>::with_seed(0.0, 1.0, 5);
    let mut buf = vec![0.0_f64; 1_000];
    dist.fill_slice_fast(&mut buf);
    let s = Array1::from(buf);
    let res = cusum(s.view(), 0.5, 8.0);
    assert!(res.alarms.is_empty());
  }

  #[test]
  fn cusum_detects_mean_shift() {
    let dist = SimdNormal::<f64>::with_seed(0.0, 1.0, 7);
    let mut buf = vec![0.0_f64; 500];
    dist.fill_slice_fast(&mut buf);
    for v in buf.iter_mut().take(500).skip(250) {
      *v += 5.0;
    }
    let s = Array1::from(buf);
    let res = cusum(s.view(), 0.5, 4.0);
    assert!(!res.alarms.is_empty());
  }

  #[test]
  fn pelt_no_changepoints_for_constant_series() {
    let s = Array1::<f64>::from_elem(100, 1.5);
    let res = pelt(s.view(), 5.0, 5);
    assert!(res.changepoints.is_empty());
  }

  #[test]
  fn pelt_finds_changepoint_at_known_break() {
    let mut buf = Array1::<f64>::zeros(200);
    for i in 0..200 {
      buf[i] = if i < 100 { 0.0 } else { 5.0 };
    }
    let res = pelt(buf.view(), 1.0, 5);
    assert!(!res.changepoints.is_empty());
    assert!(
      res
        .changepoints
        .iter()
        .any(|&cp| (cp as i64 - 100).abs() <= 5)
    );
  }
}
