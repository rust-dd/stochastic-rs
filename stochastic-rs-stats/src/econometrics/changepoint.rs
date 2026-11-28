//! Changepoint detection: Page's two-sided CUSUM for a mean shift, and PELT.
//!
//! PELT finds the global optimum at a cost at most linear in the sample; a constant has no change.
//!
//! Reference: Page (1954), Biometrika 41(1/2), 100-115, DOI: 10.2307/2333009
//! Reference: Killick et al. (2012), JASA 107(500), 1590-1598, DOI: 10.1080/01621459.2012.737745

use ndarray::Array1;
use ndarray::ArrayView1;

use crate::StatsError;
use crate::input::as_slice_or_copy;
use crate::input::reject_constant;
use crate::input::validate;
use crate::input::validate_view;
use crate::stationarity::common::about_mean;
use crate::stationarity::common::centred;
use crate::traits::FloatExt;

/// Result of a CUSUM control-chart pass.
#[derive(Debug, Clone)]
pub struct ChangepointCusumResult {
  /// Upper-side CUSUM statistic $S_t^+$.
  pub upper: Array1<f64>,
  /// Lower-side CUSUM statistic $S_t^-$.
  pub lower: Array1<f64>,
  /// Indices `t` at which $S_t^+ > h$ or $S_t^- < -h$.
  pub alarms: Vec<usize>,
}

/// Two-sided CUSUM chart, reference `k` (half the smallest shift, in sds) and threshold `h`; `Err`
/// below 2 points, on a non-finite or constant series; panics unless `k` and `h` are finite, ≥ 0.
pub fn cusum<T: FloatExt>(
  series: ArrayView1<T>,
  k: f64,
  h: f64,
) -> Result<ChangepointCusumResult, StatsError> {
  assert!(
    k.is_finite() && k >= 0.0,
    "k must satisfy `k.is_finite() && k >= 0.0`, got k = {k:?}"
  );
  assert!(
    h.is_finite() && h >= 0.0,
    "h must satisfy `h.is_finite() && h >= 0.0`, got h = {h:?}"
  );
  let y = as_slice_or_copy(series);
  validate(&y, 2)?;
  reject_constant(&y, "constant series")?;
  let n = series.len();
  let (x, _) = about_mean(
    series
      .iter()
      .map(|v| v.to_f64().unwrap())
      .collect::<Vec<_>>(),
  );
  let mean = x.iter().sum::<f64>() / n as f64;
  let var = x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n as f64 - 1.0);
  let sd = var.sqrt();
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
  Ok(ChangepointCusumResult {
    upper,
    lower,
    alarms,
  })
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

/// PELT with the mean-shift cost $\sum_{t=a}^{b-1}(y_t - \bar y_{a:b})^2$ and `penalty` per change;
/// `Err` on no or a non-finite point, panics unless `penalty` is finite and `min_size >= 1`.
pub fn pelt<T: FloatExt>(
  series: ArrayView1<T>,
  penalty: f64,
  min_size: usize,
) -> Result<PeltResult, StatsError> {
  assert!(
    penalty.is_finite(),
    "penalty must satisfy `penalty.is_finite()`, got penalty = {penalty:?}"
  );
  assert!(
    min_size >= 1,
    "min_size must satisfy `min_size >= 1`, got min_size = {min_size:?}"
  );
  validate_view(series, 1)?;
  let n = series.len();
  // About the mean: `s₂ − s²/len` cancels at a level, which no segment's cost depends on.
  let values = centred(
    &series
      .iter()
      .map(|v| v.to_f64().unwrap())
      .collect::<Vec<_>>(),
  );
  let mut prefix = vec![0.0_f64; n + 1];
  let mut prefix_sq = vec![0.0_f64; n + 1];
  for (i, v) in values.into_iter().enumerate() {
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
  Ok(PeltResult {
    changepoints: cps,
    cost: f[n],
  })
}

#[cfg(test)]
mod tests {
  use ndarray::Array1;
  use stochastic_rs_core::simd_rng::Deterministic;
  use stochastic_rs_distributions::DistributionSampler;
  use stochastic_rs_distributions::SimdDistribution;
  use stochastic_rs_distributions::normal::SimdNormal;

  use super::*;

  #[test]
  fn cusum_few_alarms_under_pure_noise() {
    let mut dist = SimdNormal::<f64>::new(0.0, 1.0).seeded(&Deterministic::new(5));
    let mut buf = vec![0.0_f64; 1_000];
    dist.fill_slice(&mut buf);
    let s = Array1::from(buf);
    let res = cusum(s.view(), 0.5, 8.0).unwrap();
    assert!(res.alarms.is_empty());
  }

  #[test]
  fn cusum_detects_mean_shift() {
    let mut dist = SimdNormal::<f64>::new(0.0, 1.0).seeded(&Deterministic::new(7));
    let mut buf = vec![0.0_f64; 500];
    dist.fill_slice(&mut buf);
    for v in buf.iter_mut().take(500).skip(250) {
      *v += 5.0;
    }
    let s = Array1::from(buf);
    let res = cusum(s.view(), 0.5, 4.0).unwrap();
    assert!(!res.alarms.is_empty());
  }

  #[test]
  fn pelt_no_changepoints_for_constant_series() {
    let s = Array1::<f64>::from_elem(100, 1.5);
    let res = pelt(s.view(), 5.0, 5).unwrap();
    assert!(res.changepoints.is_empty());
  }

  #[test]
  fn pelt_finds_changepoint_at_known_break() {
    let mut buf = Array1::<f64>::zeros(200);
    for i in 0..200 {
      buf[i] = if i < 100 { 0.0 } else { 5.0 };
    }
    let res = pelt(buf.view(), 1.0, 5).unwrap();
    assert!(!res.changepoints.is_empty());
    assert!(
      res
        .changepoints
        .iter()
        .any(|&cp| (cp as i64 - 100).abs() <= 5)
    );
  }

  #[test]
  fn cusum_rejects_a_constant_series() {
    assert_eq!(
      cusum(Array1::<f64>::from_elem(50, 1.5).view(), 0.5, 4.0).unwrap_err(),
      StatsError::DegenerateData("constant series")
    );
  }
}
