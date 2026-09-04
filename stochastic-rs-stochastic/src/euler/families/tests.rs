//! One declaration has to produce a host step and a kernel body that agree,
//! so these tests pin both halves against the closed forms they came from.

use super::*;

fn gbm_closed(x: f64, mu: f64, sigma: f64, dt: f64, sqrt_dt: f64, z: f64) -> f64 {
  x + mu * x * dt + sigma * x * sqrt_dt * z
}

fn ou_closed(x: f64, theta: f64, mu: f64, sigma: f64, dt: f64, sqrt_dt: f64, z: f64) -> f64 {
  x + theta * (mu - x) * dt + sigma * sqrt_dt * z
}

fn cir_closed(x: f64, kappa: f64, theta: f64, sigma: f64, dt: f64, sqrt_dt: f64, z: f64) -> f64 {
  let positive = if x > 0.0 { x } else { 0.0 };
  x + kappa * (theta - positive) * dt + sigma * positive.sqrt() * sqrt_dt * z
}

/// The generated host step is the same operations in the same order as the
/// closed form, so it must agree bit for bit rather than approximately.
#[test]
fn the_host_step_matches_the_closed_forms() {
  let (dt, sqrt_dt) = (1.0 / 253.0, (1.0f64 / 253.0).sqrt());
  for (x, z) in [(100.0, 0.5), (1e-8, -2.25), (-0.01, 3.0)] {
    assert_eq!(
      host_step(Family::GeometricBrownian, x, &[0.05, 0.2], dt, sqrt_dt, z),
      gbm_closed(x, 0.05, 0.2, dt, sqrt_dt, z)
    );
    assert_eq!(
      host_step(
        Family::OrnsteinUhlenbeck,
        x,
        &[0.5, 0.02, 0.1],
        dt,
        sqrt_dt,
        z
      ),
      ou_closed(x, 0.5, 0.02, 0.1, dt, sqrt_dt, z)
    );
    assert_eq!(
      host_step(Family::SquareRoot, x, &[0.5, 0.04, 0.1], dt, sqrt_dt, z),
      cir_closed(x, 0.5, 0.04, 0.1, dt, sqrt_dt, z)
    );
  }
}

/// Full truncation reports the positive part; every other family reports its
/// state unchanged, at `t = 0` as much as later.
#[test]
fn the_host_report_truncates_only_the_square_root_family() {
  assert_eq!(host_report(Family::GeometricBrownian, -2.0), -2.0);
  assert_eq!(host_report(Family::OrnsteinUhlenbeck, -2.0), -2.0);
  assert_eq!(host_report(Family::SquareRoot, -2.0), 0.0);
  assert_eq!(host_report(Family::SquareRoot, 0.25), 0.25);
}

/// The family codes are the kernels' ABI: the generated C compares against
/// these numbers, so they are pinned rather than derived from declaration
/// order by accident.
#[test]
fn the_family_codes_are_pinned() {
  assert_eq!(Family::GeometricBrownian.code(), 0);
  assert_eq!(Family::OrnsteinUhlenbeck.code(), 1);
  assert_eq!(Family::SquareRoot.code(), 2);
}

/// The emitted C binds every parameter as a local in declaration order and
/// then assigns the step, which is what makes the same tokens compile in both
/// languages.
#[test]
fn the_emitted_c_binds_parameters_and_steps() {
  assert!(C_STEP.contains("if (family == 0u) {"));
  assert!(C_STEP.contains("const REAL mu = params[0];"));
  assert!(C_STEP.contains("const REAL sigma = params[1];"));
  assert!(C_STEP.contains("x = x + mu * x * dt + sigma * x * sqrt_dt * z;"));
  assert!(C_STEP.contains("const REAL kappa = params[0];"));
  assert!(C_STEP.contains("const REAL theta = params[1];"));
  assert!(C_STEP.contains("const REAL sigma = params[2];"));
  assert!(C_STEP.contains("sqrt(positive(x))"));
}

/// The report blocks carry the same expression the host runs.
#[test]
fn the_emitted_c_reports_per_family() {
  assert!(C_REPORT.contains("if (family == 2u) {"));
  assert!(C_REPORT.contains("reported = positive(x);"));
  assert!(C_REPORT.contains("reported = x;"));
}

/// Every function the families use has a C definition, or the kernel would
/// not link.
#[test]
fn every_function_used_has_a_c_definition() {
  for name in ["sqrt", "exp", "ln", "pow", "positive", "max", "min"] {
    assert!(
      C_PRELUDE.contains(&format!("#define {name}(")),
      "{name} has no C definition"
    );
  }
  for used in ["positive(", "sqrt("] {
    assert!(C_STEP.contains(used));
    assert!(C_PRELUDE.contains(&format!("#define {}", used.trim_end_matches('('))));
  }
}
