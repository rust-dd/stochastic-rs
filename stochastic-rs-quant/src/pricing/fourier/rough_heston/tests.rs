use num_complex::Complex64;
use stochastic_rs_distributions::special::gamma;

mod adams;
mod scalar;

/// `E_{α,β}(x) = Σ xᵏ/Γ(αk + β)`, 80 terms: for `|x| ≤ 3`, where the tail is below rounding.
fn mittag_leffler(alpha: f64, beta: f64, x: f64) -> f64 {
  let term = |k: i32| x.powi(k) / gamma(alpha * f64::from(k) + beta);
  (0..80).map(term).sum()
}

/// `ξ = 3 − i/2` as `z = iξ`: the point Gatheral & Radoičić plot at.
fn z_gr() -> Complex64 {
  Complex64::new(0.5, 3.0)
}

/// Every ratio of consecutive errors, as halving the step shrinks them.
fn ratios(errors: &[f64]) -> Vec<f64> {
  errors.windows(2).map(|pair| pair[0] / pair[1]).collect()
}
