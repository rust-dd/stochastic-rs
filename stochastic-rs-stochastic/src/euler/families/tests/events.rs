use super::*;

/// The generated step under the frame's four uniforms and no noise, the rest of the frame at zero.
fn step_uniforms(
  family: Family,
  state: [f64; 4],
  params: &[f64],
  dt: f64,
  [u, u2, u3, u4]: [f64; 4],
) -> [f64; 4] {
  let mut out = [0.0; 4];
  host_step(
    family, &state, params, dt, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, u, u2,
    u3, u4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, &[0.0; 4], &mut out,
  );
  out
}

/// Dassios and Zhao's exact recursion: the wait is the smaller of the
/// baseline's exponential and the excitation's own arrival, which never
/// comes once `1 + β ln u / s` is not positive.
#[test]
fn hawkes_step_is_the_dassios_zhao_recursion() {
  let (mu, alpha, beta) = (1.0_f64, 0.5_f64, 1.5_f64);
  let params = [mu, alpha, beta];
  let (t, s) = (1.0_f64, 0.8_f64);
  let (u2, s2) = (0.3_f64, -(0.3_f64).ln() / mu);
  // A uniform near one: the excitation's own arrival comes first.
  let u = 0.95_f64;
  let d = 1.0 + beta * u.ln() / s;
  let s1 = -d.ln() / beta;
  assert!(
    s1 < s2,
    "the case is meant to have the excitation fire first"
  );
  let next = step_full(
    Family::HawkesEvents,
    [t, s, 0.0, 0.0],
    &params,
    1.0,
    0.0,
    u,
    u2,
    0.0,
    0.0,
    [0.0; 4],
  );
  assert!((next[0] - (t + s1)).abs() < 1e-12);
  assert!((next[1] - (s * (-beta * s1).exp() + alpha)).abs() < 1e-12);
  // A middling uniform: the excitation would fire, but the baseline is sooner.
  let u = 0.6_f64;
  let d = 1.0 + beta * u.ln() / s;
  assert!(
    d > 0.0 && -d.ln() / beta > s2,
    "the case is meant to have the baseline fire first"
  );
  let next = step_full(
    Family::HawkesEvents,
    [t, s, 0.0, 0.0],
    &params,
    1.0,
    0.0,
    u,
    u2,
    0.0,
    0.0,
    [0.0; 4],
  );
  assert!((next[0] - (t + s2)).abs() < 1e-12);
  // Too small a uniform: the excitation never fires and the baseline does.
  let next = step_full(
    Family::HawkesEvents,
    [t, s, 0.0, 0.0],
    &params,
    1.0,
    0.0,
    0.01,
    u2,
    0.0,
    0.0,
    [0.0; 4],
  );
  assert!((next[0] - (t + s2)).abs() < 1e-12);
  assert!((next[1] - (s * (-beta * s2).exp() + alpha)).abs() < 1e-12);
}

/// Two targets and the baselines: the step takes the earliest of the three exact clocks, marks its
/// component and excites both targets from that column of `α`.
#[test]
fn bivariate_hawkes_step_is_the_superposition_of_exact_clocks() {
  let params = [1.0_f64, 0.7, 0.3, 0.2, 0.1, 0.4, 1.5, 2.0];
  let (t, s1, s2) = (1.0_f64, 0.8_f64, 0.5_f64);
  let u2 = 0.3_f64;
  let w0 = -u2.ln() / 1.7;
  // The second target's uniform is small enough that its excess clock never fires.
  let u3 = 1.0 - (-0.25_f64).exp();
  let d2 = 1.0 + 2.0 * u3.ln() / s2;
  assert!(d2 < 0.0);
  // A mark uniform whose share of μ₁ + μ₂ exceeds μ₁ puts a baseline event on the second component.
  let u4 = 1.0 - (-1.0_f64).exp();
  assert!(u4 * 1.7 > 1.0);
  // A uniform near one: the first target's own excitation fires first.
  let u = 0.95_f64;
  let w1 = -(1.0 + 1.5 * u.ln() / s1).ln() / 1.5;
  assert!(
    w1 < w0,
    "the case is meant to have the first target's excess fire first"
  );
  let next = step_uniforms(
    Family::HawkesEvents2,
    [t, s1, s2, 1.0],
    &params,
    1.0,
    [u, u2, u3, u4],
  );
  assert!((next[0] - (t + w1)).abs() < 1e-9);
  assert_eq!(next[3], 0.0);
  assert!((next[1] - (s1 * (-1.5 * w1).exp() + 0.3)).abs() < 1e-9);
  assert!((next[2] - (s2 * (-2.0 * w1).exp() + 0.1)).abs() < 1e-9);
  // A middling uniform: the excess clocks are late and the baselines fire.
  let u = 0.6_f64;
  let w1 = -(1.0 + 1.5 * u.ln() / s1).ln() / 1.5;
  assert!(
    w1 > w0,
    "the case is meant to have the baselines fire first"
  );
  let next = step_uniforms(
    Family::HawkesEvents2,
    [t, s1, s2, 0.0],
    &params,
    1.0,
    [u, u2, u3, u4],
  );
  assert!((next[0] - (t + w0)).abs() < 1e-9);
  assert_eq!(next[3], 1.0);
  assert!((next[1] - (s1 * (-1.5 * w0).exp() + 0.2)).abs() < 1e-9);
  assert!((next[2] - (s2 * (-2.0 * w0).exp() + 0.4)).abs() < 1e-9);
}
