//! # Tail Index
//!
//! $$
//! P(\lvert X\rvert > x) \sim x^{-\alpha},\qquad \alpha = 2 - \tfrac{2}{\hat\xi}
//! $$
//!
//! Log-log regression on the empirical survival function to estimate
//! the tail exponent, with Cgmy-specific mappings for α, λ₊, λ₋.

use ndarray::ArrayView1;

/// Estimate the tail exponent of a sample via log-log regression on
/// the empirical survival function.
///
/// For standardised absolute values > 1, computes
/// slope of log P(|X| > x) vs log x.  Returns the exponent ξ
/// (positive; typically 2..6 for equity returns).
///
/// Falls back to a kurtosis-based estimate when fewer than 50
/// observations are available.
///
/// # Arguments
/// * `data`  — Raw sample (e.g. log-returns) as `ArrayView1`.
/// * `mean`  — Pre-computed sample mean.
/// * `var`   — Pre-computed sample variance (> 0).
pub fn estimate_tail_exponent(data: &ArrayView1<f64>, mean: f64, var: f64) -> f64 {
  if data.len() < 50 {
    return tail_exponent_from_kurtosis(data, mean, var);
  }

  let std = var.sqrt().max(1e-12);
  let mut abs_std: Vec<f64> = data
    .iter()
    .map(|&r| ((r - mean) / std).abs())
    .filter(|&x| x > 1.0)
    .collect();
  abs_std.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

  if abs_std.len() < 10 {
    return tail_exponent_from_kurtosis(data, mean, var);
  }

  // Log-log OLS on survival function
  let n = abs_std.len();
  let points = n.min(50);
  let mut sum_x = 0.0;
  let mut sum_y = 0.0;
  let mut sum_xy = 0.0;
  let mut sum_x2 = 0.0;

  for (i, &x) in abs_std.iter().rev().take(points).enumerate() {
    let log_x = x.ln();
    let log_surv = ((i + 1) as f64 / n as f64).ln();
    sum_x += log_x;
    sum_y += log_surv;
    sum_xy += log_x * log_surv;
    sum_x2 += log_x * log_x;
  }

  let n_pts = points as f64;
  let denom = n_pts * sum_x2 - sum_x * sum_x;
  if denom.abs() < 1e-12 {
    return 3.0;
  }
  let slope = (n_pts * sum_xy - sum_x * sum_y) / denom;
  (-slope).clamp(1.0, 6.0)
}

/// Kurtosis-based fallback for the tail exponent.
fn tail_exponent_from_kurtosis(data: &ArrayView1<f64>, mean: f64, var: f64) -> f64 {
  if data.len() < 4 || var < 1e-15 {
    return 3.0;
  }
  let n = data.len() as f64;
  let kurt = data
    .iter()
    .map(|&x| ((x - mean).powi(4)) / (var * var))
    .sum::<f64>()
    / n;
  let excess = (kurt - 3.0).abs().min(6.0);
  (4.0 - 0.5 * excess).clamp(1.5, 5.0)
}

/// Map a tail exponent to the Cgmy α (Y) parameter.
///
/// α = 2 − 2/ξ, clamped to (0.1, 1.9).
pub fn tail_exponent_to_cgmy_alpha(xi: f64) -> f64 {
  (2.0 - 2.0 / xi.max(0.5)).clamp(0.1, 1.9)
}

/// Estimate Cgmy α directly from sample data.
///
/// Combines [`estimate_tail_exponent`] and [`tail_exponent_to_cgmy_alpha`].
pub fn estimate_cgmy_alpha(data: &ArrayView1<f64>, mean: f64, var: f64) -> f64 {
  tail_exponent_to_cgmy_alpha(estimate_tail_exponent(data, mean, var))
}

/// Estimate Cgmy λ₊ / λ₋ (exponential tempering) from tail quantile
/// magnitudes.
///
/// λ ≈ 1 / E[|tail returns|], computed separately for positive and
/// negative tails.  Falls back to 1/σ when insufficient tail data.
///
/// # Arguments
/// * `data`  — Raw sample (e.g. log-returns) as `ArrayView1`.
/// * `mean`  — Pre-computed sample mean.
/// * `sigma` — Pre-computed sample standard deviation (> 0).
///
/// # Returns
/// (λ₊, λ₋)
pub fn estimate_cgmy_lambdas(data: &ArrayView1<f64>, mean: f64, sigma: f64) -> (f64, f64) {
  if data.len() < 30 {
    let base = (1.0 / sigma).max(1.0);
    return (base, base);
  }

  let thresh = sigma * 0.01;

  let pos_tail: Vec<f64> = data
    .iter()
    .filter(|&&r| r - mean > thresh)
    .map(|&r| r - mean)
    .collect();

  let neg_tail: Vec<f64> = data
    .iter()
    .filter(|&&r| mean - r > thresh)
    .map(|&r| mean - r)
    .collect();

  let lambda_plus = if pos_tail.len() >= 5 {
    let tail_mean = pos_tail.iter().sum::<f64>() / pos_tail.len() as f64;
    (1.0 / tail_mean.max(1e-6)).clamp(0.5, 50.0)
  } else {
    (1.0 / sigma).max(1.0)
  };

  let lambda_minus = if neg_tail.len() >= 5 {
    let tail_mean = neg_tail.iter().sum::<f64>() / neg_tail.len() as f64;
    (1.0 / tail_mean.max(1e-6)).clamp(0.5, 50.0)
  } else {
    (1.0 / sigma).max(1.0)
  };

  (lambda_plus, lambda_minus)
}

#[cfg(test)]
mod tests {
  use ndarray::Array1;

  use super::*;

  #[test]
  fn tail_exponent_gaussian_is_high() {
    let data = Array1::from_vec(
      (0..2000)
        .map(|i| {
          let u = (i as f64 + 0.5) / 2000.0;
          let p = u.clamp(1e-8, 1.0 - 1e-8);
          let q = if p > 0.5 { 1.0 - p } else { p };
          let t = (-2.0 * q.ln()).sqrt();
          let sign = if p > 0.5 { 1.0 } else { -1.0 };
          sign
            * (t
              - (2.515517 + 0.802853 * t + 0.010328 * t * t)
                / (1.0 + 1.432788 * t + 0.189269 * t * t + 0.001308 * t * t * t))
        })
        .collect(),
    );

    let m: f64 = data.sum() / data.len() as f64;
    let v: f64 = data.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (data.len() - 1) as f64;
    let xi = estimate_tail_exponent(&data.view(), m, v);
    assert!(
      xi > 2.0,
      "Expected high tail exponent for Gaussian, got {xi}"
    );
  }

  #[test]
  fn cgmy_alpha_in_range() {
    let data = Array1::from_vec(vec![0.01, -0.02, 0.015, -0.005, 0.03, -0.01, 0.005, -0.025]);
    let m = data.sum() / data.len() as f64;
    let v = data.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (data.len() - 1) as f64;
    let alpha = estimate_cgmy_alpha(&data.view(), m, v);
    assert!(alpha >= 0.1 && alpha <= 1.9, "alpha out of range: {alpha}");
  }

  #[test]
  fn cgmy_lambdas_positive() {
    let data = Array1::from_vec((0..100).map(|i| (i as f64 - 50.0) * 0.001).collect());
    let m = data.sum() / data.len() as f64;
    let v = data.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (data.len() - 1) as f64;
    let sigma = v.sqrt();
    let (lp, lm) = estimate_cgmy_lambdas(&data.view(), m, sigma);
    assert!(
      lp > 0.0 && lm > 0.0,
      "lambdas must be positive: ({lp}, {lm})"
    );
  }
}
