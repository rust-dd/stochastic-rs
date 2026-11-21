//! One declaration has to produce a host step and a kernel body that agree,
//! so these tests pin both halves against the closed forms they came from.

use super::vocabulary::C_PRELUDE;
use super::*;

/// The generated step and report take the whole state and noise vectors; these two shims
/// exercise a one-component family without spelling out the four slots.
fn step1(family: Family, x: f64, params: &[f64], dt: f64, dz: f64) -> f64 {
  let mut out = [0.0; 4];
  host_step(
    family,
    &[x, 0.0, 0.0, 0.0],
    params,
    dt,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    &[dz, 0.0, 0.0, 0.0],
    &mut out,
  );
  out[0]
}

fn report1(family: Family, x: f64, params: &[f64]) -> f64 {
  let mut out = [0.0; 4];
  host_report(
    family,
    &[x, 0.0, 0.0, 0.0],
    params,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    &mut out,
  );
  out[0]
}

/// The family codes are the kernels' ABI: the generated C compares against these numbers, so
/// they are pinned, not left to declaration order.
#[test]
fn the_family_codes_are_pinned() {
  assert_eq!(Family::GeometricBrownian.code(), 0);
  assert_eq!(Family::OrnsteinUhlenbeck.code(), 1);
  assert_eq!(Family::SquareRoot.code(), 2);
}

/// rustfmt wraps a call whose arguments pass 60 columns and `build.rs` reads a wrapped entry as
/// stale under locked artifacts: a three-digit family's name has 31 characters at most.
#[test]
fn every_kernel_entry_stays_on_the_one_line_rustfmt_keeps() {
  for family in Family::ALL {
    let arguments = format!("euler_{}_0_0, {family:?}, false, false", family.code());
    let columns = arguments.len();
    assert!(columns <= 60, "{family:?}: {columns} columns");
  }
}

/// The generated step over the full state and noise, for the families whose
/// state has more than one slot.
fn step_full(
  family: Family,
  state: [f64; 4],
  params: &[f64],
  dt: f64,
  ct: f64,
  u: f64,
  u2: f64,
  sj: f64,
  iv: f64,
  noise: [f64; 4],
) -> [f64; 4] {
  let mut out = [0.0; 4];
  host_step(
    family, &state, params, dt, ct, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, u, u2,
    0.0, 0.0, 0.0, 0.0, sj, 0.0, 0.0, 0.0, 0.0, iv, 0.0, 0.0, 0.0, &noise, &mut out,
  );
  out
}

mod emitted_c;
mod events;
mod host_step;
mod per_path_blocks;
mod programs;
