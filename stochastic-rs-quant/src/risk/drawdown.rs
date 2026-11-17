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

use super::input::validate_samples;
use crate::QuantError;
use crate::traits::RealExt;

/// Running drawdown `V_i / max_{s≤i} V_s − 1`, NaN where that peak is not positive.
pub fn running_drawdown<T: RealExt>(equity: ArrayView1<T>) -> Array1<T> {
  let mut out = Array1::zeros(equity.len());
  let mut peak = T::neg_infinity();
  for (i, &v) in equity.iter().enumerate() {
    if v > peak {
      peak = v;
    }
    out[i] = if peak > T::zero() {
      v / peak - T::one()
    } else {
      T::nan()
    };
  }
  out
}

/// Maximum drawdown, the most negative running drawdown; NaN for an empty series, a NaN value or a
/// non-positive start.
pub fn max_drawdown<T: RealExt>(equity: ArrayView1<T>) -> T {
  if equity.is_empty() {
    return T::nan();
  }
  T::min_or_nan_fold(running_drawdown(equity).iter().copied())
}

/// Length of the longest drawdown period (in observations).
///
/// Counts the longest run of *consecutive* observations strictly below the
/// most-recent running peak (excludes the peak observation itself). For an
/// alternative definition that measures the longest peak-to-recovery
/// distance — including the unrecovered tail when the series never
/// returns to its previous peak — see [`max_peak_to_recovery_duration`].
pub fn max_drawdown_duration<T: RealExt>(equity: ArrayView1<T>) -> usize {
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
pub fn max_peak_to_recovery_duration<T: RealExt>(equity: ArrayView1<T>) -> usize {
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
pub struct DrawdownStats<T: RealExt> {
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

impl<T: RealExt> DrawdownStats<T> {
  /// `InsufficientData` for an empty series, `InvalidInput` for a non-finite value or a
  /// non-positive start.
  pub fn from_equity(equity: ArrayView1<T>) -> Result<Self, QuantError> {
    validate_samples(equity, "drawdown statistics", 1)?;
    if equity[0] <= T::zero() {
      let reason = format!("sample 0 is {:?}, not positive", equity[0]);
      return Err(QuantError::invalid("samples", reason));
    }
    let series = running_drawdown(equity);
    let mut max = T::zero();
    let mut max_index = 0usize;
    for (i, &v) in series.iter().enumerate() {
      if v < max {
        max = v;
        max_index = i;
      }
    }
    let average = series.iter().fold(T::zero(), |acc, &v| acc + v) / T::from_usize_(series.len());
    Ok(Self {
      series,
      max,
      max_index,
      longest_duration: max_drawdown_duration(equity),
      average,
    })
  }
}

/// Convert a return series into a cumulative equity curve starting at `start`.
pub fn equity_from_returns<T: RealExt>(returns: ArrayView1<T>, start: T) -> Array1<T> {
  let mut out = Array1::zeros(returns.len() + 1);
  out[0] = start;
  for (i, &r) in returns.iter().enumerate() {
    out[i + 1] = out[i] * (T::one() + r);
  }
  out
}
