//! Shared standard-error arithmetic for the bespoke Monte Carlo pricers.

/// Standard error of a Monte Carlo mean from the running sums a payoff
/// pass produces: sample variance `(sum_sq - sum²/n) / (n - 1)`, then
/// `sqrt(var / n)` (Glasserman 2003, §1.1).
///
/// Undiscounted — the caller multiplies by the same factor its mean uses.
/// A poisoned accumulator (`NaN` in either sum) comes back as `NaN`, never
/// as a small plausible number; a tiny negative variance from cancellation
/// floors at zero; fewer than two samples have no variance estimate and
/// return `NaN`.
pub(crate) fn std_err_from_sums(sum: f64, sum_sq: f64, n: usize) -> f64 {
  if n < 2 {
    return f64::NAN;
  }
  let nf = n as f64;
  let var = (sum_sq - sum * sum / nf) / (nf - 1.0);
  if var.is_nan() {
    return f64::NAN;
  }
  (var.max(0.0) / nf).sqrt()
}

#[cfg(test)]
mod tests {
  use super::*;

  /// Cross-checked against the two-pass definition on the same data.
  #[test]
  fn matches_the_two_pass_definition() {
    let xs = [1.0_f64, 2.0, 4.0, 8.0, 16.0];
    let n = xs.len() as f64;
    let mean = xs.iter().sum::<f64>() / n;
    let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    let want = (var / n).sqrt();
    let sum = xs.iter().sum::<f64>();
    let sum_sq = xs.iter().map(|x| x * x).sum::<f64>();
    let got = std_err_from_sums(sum, sum_sq, xs.len());
    assert!((got - want).abs() < 1e-12, "got {got}, want {want}");
  }

  #[test]
  fn a_poisoned_sum_stays_nan() {
    assert!(std_err_from_sums(f64::NAN, 1.0, 10).is_nan());
    assert!(std_err_from_sums(1.0, f64::NAN, 10).is_nan());
  }

  #[test]
  fn fewer_than_two_samples_have_no_estimate() {
    assert!(std_err_from_sums(1.0, 1.0, 1).is_nan());
  }

  /// Identical samples cancel to a tiny negative variance in floating
  /// point; the floor keeps the error at exactly zero.
  #[test]
  fn constant_payoffs_report_zero_error() {
    let sum = 0.1_f64 * 3.0;
    let sum_sq = 0.1_f64 * 0.1 * 3.0;
    assert_eq!(std_err_from_sums(sum, sum_sq, 3), 0.0);
  }
}
