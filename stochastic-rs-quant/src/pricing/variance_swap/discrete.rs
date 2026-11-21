use crate::pricing::heston::HestonParams;

pub(super) fn mean_reversion_factor(x: f64) -> f64 {
  if x == 0.0 {
    1.0
  } else if x == f64::NEG_INFINITY {
    f64::INFINITY
  } else {
    -(-x).exp_m1() / x
  }
}

/// `Var(V)` of `V = (1/τ)∫v dt` from the CIR covariance `Cov(v_s, v_t) = e^{-κ(t-s)} Var(v_s)`:
/// $$
/// \mathrm{Var}(V)=\frac{\sigma^2}{\kappa^3\tau^2}\Bigl[(v_0-\theta)\bigl(1-e^{-2x}-2xe^{-x}\bigr)
/// +\theta\bigl(x-\tfrac32+2e^{-x}-\tfrac12e^{-2x}\bigr)\Bigr],\quad x=\kappa\tau
/// $$
pub(super) fn realized_variance_variance(params: &HestonParams, tau: f64) -> f64 {
  let HestonParams {
    v0,
    kappa,
    theta,
    sigma,
    ..
  } = *params;
  let x = if kappa == 0.0 { 0.0 } else { kappa * tau };
  if x.abs() < 0.5 {
    let (g1, g2) = cir_variance_series(x);
    return sigma * sigma * tau * ((v0 - theta) * g1 + theta * g2);
  }
  let h1 = -(-2.0 * x).exp_m1() / x - 2.0 * (-x).exp();
  let h2 = 1.0 - 1.5 / x + (2.0 * (-x).exp() - 0.5 * (-2.0 * x).exp()) / x;
  sigma * sigma / (kappa * kappa * tau) * ((v0 - theta) * h1 + theta * h2)
}

/// The two brackets above over `x³` by their Taylor series, where the closed forms cancel.
fn cir_variance_series(x: f64) -> (f64, f64) {
  let (mut g1, mut g2) = (0.0, 0.0);
  let (mut factorial, mut pow2, mut power) = (6.0, 8.0, 1.0);
  for n in 3..24 {
    if n > 3 {
      factorial *= f64::from(n);
      pow2 *= 2.0;
      power *= x;
    }
    let sign = if n % 2 == 0 { 1.0 } else { -1.0 };
    g1 -= sign * (pow2 - 2.0 * f64::from(n)) / factorial * power;
    g2 += sign * (2.0 - 0.5 * pow2) / factorial * power;
  }
  (g1, g2)
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
