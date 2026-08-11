//! Convenience helpers: estimate `H` directly from a close-price
//! series.  Wraps [`crate::fractal_dim::Higuchi`] (via the FBM relation
//! `H = 2 - D`) and cross-validates two volatility proxies (rolling
//! mean-absolute-return vs raw absolute returns).

use ndarray::Array1;
use ndarray::ArrayView1;

use super::HurstError;
use super::HurstEstimator;
use crate::fractal_dim::Higuchi;
use crate::traits::FloatExt;

/// Estimate `H` from a close-price series.
///
/// Uses Higuchi-fractal-dim on a rolling realized-vol proxy
/// (rolling mean absolute return), cross-validated against an absolute-
/// return Higuchi estimate.  Returns `H ∈ [0.05, 0.45]` on success.
///
/// # Errors
///
/// Returns [`HurstError::TooFewObservations`] when `closes` (or the
/// finite log-returns / vol-proxy derived from it) is too short, and
/// propagates whatever [`hurst_from_signal`] returns for the underlying
/// Higuchi fit. Callers that want the pre-2.7 clamp-to-default behavior
/// can write `estimate_hurst(x).unwrap_or(0.1)` explicitly at the call
/// site — the function itself no longer guesses on your behalf.
pub fn estimate_hurst<T: FloatExt>(closes: ArrayView1<T>) -> Result<f64, HurstError> {
  let n = closes.len();
  if n < 30 {
    return Err(HurstError::TooFewObservations {
      got: n,
      required: 30,
    });
  }
  let rets: Vec<f64> = (1..n)
    .filter_map(|i| {
      let c0 = closes[i - 1].to_f64().unwrap_or(f64::NAN);
      let c1 = closes[i].to_f64().unwrap_or(f64::NAN);
      let r = (c1 / c0).ln();
      if r.is_finite() { Some(r) } else { None }
    })
    .collect();
  if rets.len() < 30 {
    return Err(HurstError::TooFewObservations {
      got: rets.len(),
      required: 30,
    });
  }

  let window = 5.min(rets.len() / 4).max(2);
  let vol_proxy: Vec<f64> = rets
    .windows(window)
    .map(|w| {
      let sum: f64 = w.iter().map(|r| r.abs()).sum();
      sum / window as f64
    })
    .filter(|v| v.is_finite() && *v > 0.0)
    .collect();

  if vol_proxy.len() < 20 {
    let abs_rets: Array1<f64> = Array1::from_vec(
      rets
        .iter()
        .map(|r| r.abs())
        .filter(|r| r.is_finite() && *r > 0.0)
        .collect(),
    );
    return hurst_from_signal(abs_rets.view());
  }

  let h_rv = hurst_from_signal(Array1::from_vec(vol_proxy).view())?;
  let abs_arr = Array1::from_vec(
    rets
      .iter()
      .map(|r| r.abs())
      .filter(|r| r.is_finite() && *r > 0.0)
      .collect(),
  );
  let h_abs = hurst_from_signal(abs_arr.view())?;

  Ok(if (h_rv - h_abs).abs() > 0.15 {
    h_rv.min(h_abs).clamp(0.05, 0.45)
  } else {
    (0.65 * h_rv + 0.35 * h_abs).clamp(0.05, 0.45)
  })
}

/// Estimate `H` from an arbitrary positive signal via Higuchi FD.
///
/// Result is clamped to `[0.05, 0.45]` on success.
///
/// # Errors
///
/// Returns [`HurstError::TooFewObservations`] when `signal` has fewer
/// than 20 points, propagates [`Higuchi`]'s own error, and returns
/// [`HurstError::RegressionFailed`] when the fit produces a Hurst value
/// outside `(0, 1)` (a degenerate log-log regression). Callers that want
/// the pre-2.7 clamp-to-default behavior can write
/// `hurst_from_signal(x).unwrap_or(0.1)` explicitly.
pub fn hurst_from_signal<T: FloatExt>(signal: ArrayView1<T>) -> Result<f64, HurstError> {
  let n = signal.len();
  if n < 20 {
    return Err(HurstError::TooFewObservations {
      got: n,
      required: 20,
    });
  }
  let kmax = 64.min(n / 4).max(4);
  let est = Higuchi { kmax };
  let r = HurstEstimator::<T>::estimate(&est, signal)?;
  let h = r.hurst.to_f64().ok_or(HurstError::RegressionFailed)?;
  if h.is_finite() && h > 0.0 && h < 1.0 {
    Ok(h.clamp(0.05, 0.45))
  } else {
    Err(HurstError::RegressionFailed)
  }
}

#[cfg(test)]
mod tests {
  use ndarray::Array1;

  use super::*;

  #[test]
  fn estimate_hurst_errs_on_degenerate_input() {
    let closes = Array1::<f64>::from_elem(200, 100.0);
    let result = estimate_hurst(closes.view());
    assert!(
      result.is_err(),
      "constant price series must signal an error, not silently return 0.1"
    );
  }

  #[test]
  fn estimate_hurst_errs_on_short_input() {
    let closes = Array1::from_vec(vec![100.0, 101.0, 99.5]);
    assert_eq!(
      estimate_hurst(closes.view()),
      Err(HurstError::TooFewObservations {
        got: 3,
        required: 30
      })
    );
  }

  #[test]
  fn hurst_from_signal_errs_on_short_input() {
    let signal = Array1::from_vec(vec![0.1, 0.2, 0.3]);
    assert_eq!(
      hurst_from_signal(signal.view()),
      Err(HurstError::TooFewObservations {
        got: 3,
        required: 20
      })
    );
  }
}
