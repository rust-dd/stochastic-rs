//! # Leverage
//!
//! $$
//! \hat\rho = \text{Corr}\!\bigl(r_t,\;\lvert r_{t+1}\rvert - \lvert r_t\rvert\bigr)
//! $$
//!

use ndarray::ArrayView1;

use crate::hurst::HurstError;

/// Estimate the leverage correlation (rho) between log-returns and
/// volatility changes from a close-price series.
///
/// Uses the empirical Pearson correlation between r_t and
/// (|r_{t+1}| − |r_t|) as a proxy for the Heston-style
/// dW_S · dW_V = ρ dt.
///
/// Equities typically exhibit ρ ∈ [−0.9, −0.3] (leverage effect:
/// negative returns increase future volatility).
///
/// # Arguments
/// * `closes` — Price series as `ArrayView1`, length >= 20.
///
/// # Errors
///
/// Returns [`HurstError::TooFewObservations`] when `closes` (or the
/// finite log-returns / return-pairs derived from it) is too short, and
/// [`HurstError::DegeneratePath`] when the return / vol-change series has
/// (near-)zero variance, making the Pearson correlation undefined.
/// Reuses [`HurstError`] rather than a bespoke type: both estimators fail
/// for the same reasons (too little data, degenerate input) on the same
/// close-price series shape. Callers that want the pre-2.7 clamp-to-default
/// behavior can write `estimate_leverage_rho(x).unwrap_or(-0.5)` explicitly
/// — the function itself no longer guesses on your behalf.
pub fn estimate_leverage_rho(closes: ArrayView1<f64>) -> Result<f64, HurstError> {
  let n = closes.len();
  if n < 20 {
    return Err(HurstError::TooFewObservations {
      got: n,
      required: 20,
    });
  }

  let mut rets = Vec::with_capacity(n - 1);
  for i in 1..n {
    let r = (closes[i] / closes[i - 1]).ln();
    if r.is_finite() {
      rets.push(r);
    }
  }
  if rets.len() < 20 {
    return Err(HurstError::TooFewObservations {
      got: rets.len(),
      required: 20,
    });
  }

  // Vol proxy: |r_t|
  let vol: Vec<f64> = rets.iter().map(|r| r.abs()).collect();
  let pairs = rets.len() - 1;
  if pairs < 10 {
    return Err(HurstError::TooFewObservations {
      got: pairs,
      required: 10,
    });
  }

  let mut mean_x = 0.0;
  let mut mean_y = 0.0;
  for i in 0..pairs {
    mean_x += rets[i];
    mean_y += vol[i + 1] - vol[i];
  }
  mean_x /= pairs as f64;
  mean_y /= pairs as f64;

  let mut sum_xy = 0.0;
  let mut sum_x2 = 0.0;
  let mut sum_y2 = 0.0;
  for i in 0..pairs {
    let dx = rets[i] - mean_x;
    let dy = (vol[i + 1] - vol[i]) - mean_y;
    sum_xy += dx * dy;
    sum_x2 += dx * dx;
    sum_y2 += dy * dy;
  }

  let denom = (sum_x2 * sum_y2).sqrt();
  if denom < 1e-15 {
    return Err(HurstError::DegeneratePath);
  }
  Ok((sum_xy / denom).clamp(-0.99, 0.99))
}

#[cfg(test)]
mod tests {
  use ndarray::Array1;

  use super::*;

  #[test]
  fn leverage_rho_short_data_errs() {
    let closes = Array1::from_elem(10, 100.0);
    assert_eq!(
      estimate_leverage_rho(closes.view()),
      Err(HurstError::TooFewObservations {
        got: 10,
        required: 20
      })
    );
  }

  #[test]
  fn leverage_rho_constant_series_errs_degenerate() {
    let closes = Array1::from_elem(50, 100.0);
    assert_eq!(
      estimate_leverage_rho(closes.view()),
      Err(HurstError::DegeneratePath)
    );
  }

  #[test]
  fn leverage_rho_in_valid_range() {
    let mut vals = Vec::with_capacity(200);
    vals.push(100.0);
    let mut rng_state: u64 = 42;
    for _ in 1..200 {
      rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
      let u = (rng_state >> 33) as f64 / (1u64 << 31) as f64;
      let r = (u - 0.5) * 0.04;
      let last = *vals.last().unwrap();
      vals.push(last * (1.0 + r));
    }
    let closes = Array1::from_vec(vals);
    let rho = estimate_leverage_rho(closes.view()).expect("200-point noisy series must fit");
    assert!((-0.99..=0.99).contains(&rho), "rho out of range: {rho}");
  }
}
