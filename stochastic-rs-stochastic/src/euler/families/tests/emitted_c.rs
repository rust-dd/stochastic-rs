use super::*;

/// The emitted C binds every parameter as a local in declaration order and then assigns the
/// step, which is what makes the same tokens compile in both languages.
#[test]
fn the_emitted_c_binds_parameters_and_steps() {
  assert!(C_STEP.contains("if (family == 0u) {"));
  assert!(C_STEP.contains("const REAL mu = params[0];"));
  assert!(C_STEP.contains("const REAL sigma = params[1];"));
  // The state and the noise are bound by position from the launch's own
  // vectors, so a one-component family reads slot zero of each.
  assert!(C_STEP.contains("const REAL x = state[0];"));
  assert!(C_STEP.contains("const REAL dz = noise[0];"));
  // Every component is computed into a temporary before any is stored, so a
  // system's second component still sees the state as it stood.
  assert!(C_STEP.contains("const REAL __n0 = x + mu * x * dt + sigma * x * dz;"));
  assert!(C_STEP.contains("state[0] = __n0;"));
  assert!(C_STEP.contains("const REAL kappa = params[0];"));
  assert!(C_STEP.contains("const REAL theta = params[1];"));
  assert!(C_STEP.contains("const REAL sigma = params[2];"));
  assert!(C_STEP.contains("sqrt(positive(x))"));
}

/// The report blocks carry the same expression the host runs.
#[test]
fn the_emitted_c_reports_per_family() {
  assert!(C_REPORT.contains("if (family == 2u) {"));
  assert!(C_REPORT.contains("const REAL __n0 = positive(x);"));
  assert!(C_REPORT.contains("const REAL __n0 = x;"));
  assert!(C_REPORT.contains("reported[0] = __n0;"));
}

/// A `bind` becomes a local in every language: a `let` on the host, a `const REAL` in the
/// emitted C, so a family names a clamped state once.
#[test]
fn a_bind_becomes_a_c_local() {
  assert!(C_STEP.contains("const REAL xi = positive(x);"));
  assert!(C_STEP.contains(
    "const REAL __n0 = positive(xi + kappa * (theta - xi) * xi * dt + sigma * sqrt(xi) * dz);"
  ));
  assert_eq!(
    step1(Family::FellerLogistic, -3.0, &[0.5, 0.04, 0.1], 0.01, 0.02),
    0.0,
    "a negative state truncates to zero before the coefficients see it"
  );
}

/// A report may name the family's parameters, which lets the displaced diffusion step the
/// shifted variable and report the shift back out.
#[test]
fn a_report_sees_the_family_parameters() {
  assert!(C_REPORT.contains("const REAL beta = params[2];"));
  assert!(C_REPORT.contains("const REAL __n0 = x - beta;"));
  assert_eq!(report1(Family::Displaced, 105.0, &[0.05, 0.2, 5.0]), 100.0);
}

/// Every function the families use has a C definition, or the kernel would
/// not link.
#[test]
fn every_function_used_has_a_c_definition() {
  for name in [
    "sqrt", "exp", "ln", "pow", "abs", "negate", "tanh", "positive", "max", "min", "lit", "less",
    "leq", "geq", "pick",
  ] {
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
