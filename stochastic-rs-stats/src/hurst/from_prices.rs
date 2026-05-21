//! Convenience helpers: estimate `H` directly from a close-price
//! series.  Wraps [`crate::fractal_dim::Higuchi`] (via the FBM relation
//! `H = 2 - D`) and cross-validates two volatility proxies as in the
//! legacy `fd::estimate_hurst` API.

use ndarray::Array1;
use ndarray::ArrayView1;

use super::HurstEstimator;
use crate::fractal_dim::Higuchi;
use crate::traits::FloatExt;

/// Estimate `H` from a close-price series.
///
/// Uses Higuchi-fractal-dim on a rolling realized-vol proxy
/// (rolling mean absolute return), cross-validated against an absolute-
/// return Higuchi estimate.  Returns `H ∈ [0.05, 0.45]`.  Falls back to
/// `0.1` on insufficient data or unreliable estimate.
///
/// Matches the v2.2 `crate::fd::estimate_hurst` behaviour bit-for-bit
/// so the deprecated re-export remains a drop-in.
pub fn estimate_hurst<T: FloatExt>(closes: ArrayView1<T>) -> f64 {
  let n = closes.len();
  if n < 30 {
    return 0.1;
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
    return 0.1;
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

  let h_rv = hurst_from_signal(Array1::from_vec(vol_proxy).view());
  let abs_arr = Array1::from_vec(
    rets
      .iter()
      .map(|r| r.abs())
      .filter(|r| r.is_finite() && *r > 0.0)
      .collect(),
  );
  let h_abs = hurst_from_signal(abs_arr.view());

  if (h_rv - h_abs).abs() > 0.15 {
    h_rv.min(h_abs).clamp(0.05, 0.45)
  } else {
    (0.65 * h_rv + 0.35 * h_abs).clamp(0.05, 0.45)
  }
}

/// Estimate `H` from an arbitrary positive signal via Higuchi FD.
///
/// Falls back to `0.1` for degenerate / too-short input.  Result is
/// clamped to `[0.05, 0.45]`.
pub fn hurst_from_signal<T: FloatExt>(signal: ArrayView1<T>) -> f64 {
  let n = signal.len();
  if n < 20 {
    return 0.1;
  }
  let kmax = 64.min(n / 4).max(4);
  let est = Higuchi { kmax };
  match HurstEstimator::<T>::estimate(&est, signal) {
    Ok(r) => {
      let h = r.hurst.to_f64().unwrap_or(0.1);
      if h.is_finite() && h > 0.0 && h < 1.0 {
        h.clamp(0.05, 0.45)
      } else {
        0.1
      }
    }
    Err(_) => 0.1,
  }
}
