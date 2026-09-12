pub(super) fn mean_reversion_factor(x: f64) -> f64 {
  if x == 0.0 { 1.0 } else { -(-x).exp_m1() / x }
}

/// Time average of the CIR second moment, including its zero-kappa limit.
pub(super) fn mean_square_variance(v0: f64, kappa: f64, theta: f64, sigma: f64, tau: f64) -> f64 {
  let x = kappa * tau;
  let a1 = mean_reversion_factor(x);
  let a2 = mean_reversion_factor(2.0 * x);
  let delta = v0 - theta;
  let mean_squared = theta * theta + 2.0 * theta * delta * a1 + delta * delta * a2;
  // These divided differences cancel near zero; the series retain the CIR limit.
  let (q1, q2) = if x.abs() < 1e-4 {
    (
      0.5 + x * (-0.5 + x * (7.0 / 24.0 + x * (-0.125 + x * 31.0 / 720.0))),
      x * (1.0 / 6.0 + x * (-0.125 + x * (7.0 / 120.0 - x / 48.0))),
    )
  } else {
    ((a1 - a2) / x, (1.0 - 2.0 * a1 + a2) / (2.0 * x))
  };
  mean_squared + sigma * sigma * tau * (v0 * q1 + theta * q2)
}
