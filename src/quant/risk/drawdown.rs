//! Drawdown metrics for cumulative return / price series.
//!
//! Reference: Magdon-Ismail & Atiya, "Maximum Drawdown", Risk Magazine,
//! October 2004.
//!
//! Given an equity series $V_t$,
//! $$
//! \mathrm{DD}_t=\frac{V_t-\max_{s\le t}V_s}{\max_{s\le t}V_s},\qquad
//! \mathrm{MDD}=\min_t\mathrm{DD}_t.
//! $$
//! Drawdown is reported as a negative fraction; e.g. `-0.25` is a 25 %
//! drawdown.  The maximum drawdown magnitude can be recovered as `-mdd`.

use ndarray::Array1;
use ndarray::ArrayView1;

use crate::traits::FloatExt;

/// Running drawdown series.  Entry $i$ is $V_i/\max_{s\le i}V_s - 1$.
pub fn running_drawdown<T: FloatExt>(equity: ArrayView1<T>) -> Array1<T> {
  let mut out = Array1::zeros(equity.len());
  let mut peak = T::min_positive_val();
  for (i, &v) in equity.iter().enumerate() {
    if v > peak {
      peak = v;
    }
    out[i] = if peak > T::zero() {
      v / peak - T::one()
    } else {
      T::zero()
    };
  }
  out
}

/// Maximum drawdown (most negative value of the running drawdown series).
pub fn max_drawdown<T: FloatExt>(equity: ArrayView1<T>) -> T {
  let dd = running_drawdown(equity);
  dd
    .iter()
    .copied()
    .fold(T::zero(), |a, b| if b < a { b } else { a })
}

/// Length of the longest drawdown period (in observations).
pub fn max_drawdown_duration<T: FloatExt>(equity: ArrayView1<T>) -> usize {
  let mut peak = T::min_positive_val();
  let mut current = 0usize;
  let mut longest = 0usize;
  for &v in equity.iter() {
    if v >= peak {
      peak = v;
      current = 0;
    } else {
      current += 1;
      if current > longest {
        longest = current;
      }
    }
  }
  longest
}

/// Full drawdown statistics for an equity series.
#[derive(Debug, Clone)]
pub struct DrawdownStats<T: FloatExt> {
  /// Running drawdown series.
  pub series: Array1<T>,
  /// Most negative drawdown (≤ 0).
  pub max: T,
  /// Index at which the maximum drawdown was reached.
  pub max_index: usize,
  /// Length of the longest drawdown period in observations.
  pub longest_duration: usize,
  /// Average drawdown (arithmetic mean of `series`).
  pub average: T,
}

impl<T: FloatExt> DrawdownStats<T> {
  /// Compute all drawdown statistics from an equity series.
  pub fn from_equity(equity: ArrayView1<T>) -> Self {
    let series = running_drawdown(equity);
    let mut max = T::zero();
    let mut max_index = 0usize;
    for (i, &v) in series.iter().enumerate() {
      if v < max {
        max = v;
        max_index = i;
      }
    }
    let n = series.len().max(1);
    let average = series.iter().fold(T::zero(), |acc, &v| acc + v) / T::from_usize_(n);
    Self {
      series,
      max,
      max_index,
      longest_duration: max_drawdown_duration(equity),
      average,
    }
  }
}

/// Convert a return series into a cumulative equity curve starting at `start`.
pub fn equity_from_returns<T: FloatExt>(returns: ArrayView1<T>, start: T) -> Array1<T> {
  let mut out = Array1::zeros(returns.len() + 1);
  out[0] = start;
  for (i, &r) in returns.iter().enumerate() {
    out[i + 1] = out[i] * (T::one() + r);
  }
  out
}
