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
///
/// `peak` is initialised to `T::neg_infinity()` so the first observation
/// always becomes the running peak — even for series that start with a
/// negative value or below `T::min_positive_val()`. The drawdown formula
/// itself only makes sense on strictly-positive equity series; values
/// outside that domain pin `out[i] = 0` since the ratio `v / peak - 1`
/// has no economic meaning for non-positive peaks.
pub fn running_drawdown<T: FloatExt>(equity: ArrayView1<T>) -> Array1<T> {
  let mut out = Array1::zeros(equity.len());
  let mut peak = T::neg_infinity();
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
  dd.iter()
    .copied()
    .fold(T::zero(), |a, b| if b < a { b } else { a })
}

/// Length of the longest drawdown period (in observations).
///
/// Counts the longest run of *consecutive* observations strictly below the
/// most-recent running peak (excludes the peak observation itself). For an
/// alternative definition that measures the longest peak-to-recovery
/// distance — including the unrecovered tail when the series never
/// returns to its previous peak — see [`max_peak_to_recovery_duration`].
pub fn max_drawdown_duration<T: FloatExt>(equity: ArrayView1<T>) -> usize {
  let mut peak = T::neg_infinity();
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

/// Longest peak-to-recovery distance (in observations).
///
/// For each peak that is followed by at least one strictly-below-peak
/// observation, counts the number of bars from the peak until the
/// equity series next reaches that peak (or higher). Returns the
/// maximum such distance. Unrecovered tail-drawdowns at the end of the
/// series count as `equity.len() - 1 - peak_index`, since the recovery
/// never happens — this matches the "time-under-water" convention used
/// by Magdon-Ismail & Atiya (2004) and the `MaxDD Duration` metric on
/// Bloomberg / Refinitiv.
///
/// Monotonically-increasing series report `0` (no underwater episodes).
pub fn max_peak_to_recovery_duration<T: FloatExt>(equity: ArrayView1<T>) -> usize {
  let n = equity.len();
  if n == 0 {
    return 0;
  }
  let mut peak = T::neg_infinity();
  let mut peak_idx: usize = 0;
  let mut been_below = false;
  let mut longest = 0usize;
  for (i, &v) in equity.iter().enumerate() {
    if v >= peak {
      if been_below {
        let dist = i - peak_idx;
        if dist > longest {
          longest = dist;
        }
      }
      peak = v;
      peak_idx = i;
      been_below = false;
    } else {
      been_below = true;
    }
  }
  if been_below {
    let dist = n - 1 - peak_idx;
    if dist > longest {
      longest = dist;
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
