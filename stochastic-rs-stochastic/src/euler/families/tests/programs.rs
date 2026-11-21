use super::*;

/// The generated step with the two program values a family reads as `pv` and
/// `pv2`, the rest of the frame at zero.
fn step_program(
  family: Family,
  state: [f64; 4],
  params: &[f64],
  dt: f64,
  lv: f64,
  pv: f64,
  pv2: f64,
  noise: [f64; 4],
) -> [f64; 4] {
  let mut out = [0.0; 4];
  host_step(
    family, &state, params, dt, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, lv, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, pv, pv2, &noise, &mut out,
  );
  out
}

/// The Cheyette state steps by Euler–Maruyama with the program's value as
/// the local volatility of the step's start.
#[test]
fn cheyette_step_is_euler_with_the_program_volatility() {
  let (kappa, dt, dz) = (0.5_f64, 0.01_f64, 0.1_f64);
  let (x, y, s) = (0.02_f64, 0.001_f64, 0.03_f64);
  let next = step_program(
    Family::CheyetteLocalVol,
    [x, y, 0.0, 0.0],
    &[kappa],
    dt,
    0.0,
    s,
    0.0,
    [dz, 0.0, 0.0, 0.0],
  );
  assert!((next[0] - (x + (y - kappa * x) * dt + s * dz)).abs() < 1e-15);
  assert!((next[1] - (y + (s * s - 2.0 * kappa * y) * dt)).abs() < 1e-15);
}

/// The SLV spot steps its logarithm under the program's value as the leverage at the step's
/// start, and the variance under the mixed vol-of-vol, the two shocks correlated in the step.
#[test]
fn heston_slv_step_is_the_log_euler_scheme_under_the_program_leverage() {
  let (mu, kappa, theta, sigma, rho) = (0.03_f64, 2.0_f64, 0.04_f64, 0.21_f64, -0.6_f64);
  let (dt, dw, dz) = (0.01_f64, 0.05_f64, -0.02_f64);
  let (s, v, l) = (100.0_f64, 0.05_f64, 0.9_f64);
  let next = step_program(
    Family::HestonSlv,
    [s, v, 0.0, 0.0],
    &[mu, kappa, theta, sigma, rho],
    dt,
    0.0,
    l,
    0.0,
    [dw, dz, 0.0, 0.0],
  );
  let db = rho * dw + (1.0 - rho * rho).sqrt() * dz;
  let s_next = s * ((mu - 0.5 * l * l * v) * dt + l * v.sqrt() * dw).exp();
  let v_next = v + kappa * (theta - v) * dt + sigma * v.sqrt() * db;
  assert!(
    (next[0] - s_next).abs() < 1e-12,
    "spot {} vs {s_next}",
    next[0]
  );
  assert!(
    (next[1] - v_next).abs() < 1e-15,
    "variance {} vs {v_next}",
    next[1]
  );
}

/// The Volterra family is its lift: the step is the lifted value, and the
/// lift's drift and diffusion are the two programs, its shock the noise.
#[test]
fn volterra_program_step_is_the_lift_of_its_programs() {
  let noise = [0.07, 0.0, 0.0, 0.0];
  let next = step_program(
    Family::VolterraProgram,
    [0.2, 0.0, 0.0, 0.0],
    &[],
    0.01,
    0.31,
    0.5,
    0.25,
    noise,
  );
  assert_eq!(next[0], 0.31);
  let lift = host_lift(
    Family::VolterraProgram,
    &[0.2, 0.0, 0.0, 0.0],
    &[],
    0.01,
    0.5,
    0.25,
    &noise,
  );
  assert_eq!(lift, [0.5, 0.25, 0.07]);
}
