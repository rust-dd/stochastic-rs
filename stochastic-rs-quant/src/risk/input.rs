//! The checks the risk estimators run on their samples, and the error for a moment that over- or
//! underflows on finite ones.

use ndarray::ArrayView1;

use crate::QuantError;
use crate::traits::RealExt;

/// `InsufficientData` below `needed` samples, `InvalidInput` naming the first non-finite one.
pub(crate) fn validate_samples<T: RealExt>(
  samples: ArrayView1<T>,
  what: &'static str,
  needed: usize,
) -> Result<(), QuantError> {
  let got = samples.len();
  if got < needed {
    return Err(QuantError::InsufficientData { what, needed, got });
  }
  match samples.iter().position(|v| !v.is_finite()) {
    Some(i) => Err(QuantError::invalid(
      "samples",
      format!("sample {i} is {:?}", samples[i]),
    )),
    None => Ok(()),
  }
}

/// Every sample equal to the next; exact, as a rounded mean leaves a constant series a deviation.
pub(crate) fn is_constant<T: RealExt>(samples: ArrayView1<T>) -> bool {
  samples.windows(2).into_iter().all(|w| w[0] == w[1])
}

/// `value` if finite; from finite samples anything else is an overflow, so `Numerical`.
pub(crate) fn finite<T: RealExt>(
  value: T,
  what: &'static str,
  quantity: &str,
) -> Result<T, QuantError> {
  if value.is_finite() {
    Ok(value)
  } else {
    Err(numerical(what, quantity, value))
  }
}

/// `value` if finite and positive, as a deviation of non-constant samples is; else `Numerical`.
pub(crate) fn finite_positive<T: RealExt>(
  value: T,
  what: &'static str,
  quantity: &str,
) -> Result<T, QuantError> {
  if value > T::zero() && value.is_finite() {
    Ok(value)
  } else {
    Err(numerical(what, quantity, value))
  }
}

fn numerical<T: RealExt>(what: &'static str, quantity: &str, value: T) -> QuantError {
  let how = if value.is_finite() {
    "underflows"
  } else {
    "overflows"
  };
  QuantError::Numerical {
    what,
    reason: format!("{quantity} of finite samples {how} to {value:?}"),
  }
}
