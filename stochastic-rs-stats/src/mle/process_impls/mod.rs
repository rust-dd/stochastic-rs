mod cir;
#[cfg(test)]
mod cir_tests;
#[cfg(test)]
mod closed_form_tests;
#[cfg(test)]
mod derivative_tests;
mod general;
mod linear;
mod logistic;

#[cfg(test)]
mod tests;

/// The state, or `1e-12` within that of zero, where a drift in `1/x` or `ln x` has no value: for
/// either sign, so inside the guard a drift's derivatives are those at `+1e-12`.
fn off_zero(x: f64) -> f64 {
  if x.abs() < 1e-12 { 1e-12 } else { x }
}

/// `a[0]/x + a[1] + a[2]x + a[3]x²`, the drift `NonlinearSde` and `AitSahalia` share.
fn pole_quadratic(a: [f64; 4], x: f64) -> f64 {
  a[0] / off_zero(x) + a[1] + a[2] * x + a[3] * x * x
}

fn pole_quadratic_x(a: [f64; 4], x: f64) -> f64 {
  let safe_x = off_zero(x);
  -a[0] / (safe_x * safe_x) + a[2] + 2.0 * a[3] * x
}

fn pole_quadratic_xx(a: [f64; 4], x: f64) -> f64 {
  let safe_x = off_zero(x);
  2.0 * a[0] / (safe_x * safe_x * safe_x) + 2.0 * a[3]
}

/// `d/dx` of `c|x|^p`: below `p = 1` unbounded at zero, where a fixed difference step misses it.
fn power_x(c: f64, p: f64, x: f64) -> f64 {
  c * p * x.signum() * x.abs().powf(p - 1.0)
}

/// `d²/dx²` of `c|x|^p`, unbounded at zero below `p = 2`.
fn power_xx(c: f64, p: f64, x: f64) -> f64 {
  c * p * (p - 1.0) * x.abs().powf(p - 2.0)
}

/// `d/dx` of `√|g|` from `g` and `g′`, unbounded where `g` crosses zero.
fn root_x(g: f64, g1: f64) -> f64 {
  0.5 * g.signum() * g1 / g.abs().sqrt()
}

/// `d²/dx²` of `√|g|` from `g`, `g′` and `g″`.
fn root_xx(g: f64, g1: f64, g2: f64) -> f64 {
  let r = g.abs().sqrt();
  0.5 * g.signum() * g2 / r - 0.25 * g1 * g1 / (r * g.abs())
}
