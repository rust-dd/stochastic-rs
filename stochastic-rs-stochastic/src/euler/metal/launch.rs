use ndarray::Array3;

use super::{Increments, Result, context::run};
use crate::device::DeviceError;
use crate::device::metal::bindings::EulerArgs;
use crate::euler::EulerSpec;

/// Lends the completed plane-major output to `finish` with its component count.
#[allow(clippy::too_many_arguments)]
pub(super) fn launch_paths<O>(
  ordinal: usize,
  spec: EulerSpec<f32>,
  x0: [f32; 4],
  n: usize,
  dt: f32,
  first: usize,
  m: usize,
  seed: u64,
  increments: Increments<'_>,
  curves: Option<Vec<Vec<f32>>>,
  jump_lambda: Option<f32>,
  sizes: Option<crate::euler::JumpSizes<f32>>,
  step_first: bool,
  gammas: Option<crate::euler::GammaDraws<f32>>,
  lift: Option<crate::euler::LiftSpec<'_, f32>>,
  series: Option<u32>,
  table: Option<crate::euler::TableSpec<f32>>,
  program: Option<crate::euler::ProgramSpec<'_>>,
  finish: impl FnOnce(&[f32], usize) -> O,
) -> Result<O> {
  let (family, params) = spec.encode();
  let (program_buf, program_n) = crate::euler::encode_programs::<f32>(program.as_ref());
  let arity = crate::euler::families::Family::from_code(family).expect("a declared family");
  let (components, noises) = (arity.components(), arity.noises());
  if n == 0 || m == 0 {
    return Ok(finish(&[], components));
  }
  let streams = match increments {
    Increments::Hashed => 0,
    Increments::Device {
      buffer,
      streams,
      row,
    } => {
      let steps = if step_first { n } else { n - 1 };
      validate_increments(buffer.len(), m, steps, streams, row)?;
      streams
    }
  };
  let (steps, paths, first_path) = crate::euler::launch_scalars(n, first, m, streams)?;
  validate_curve_extent(steps, curves.as_ref().map_or(0, Vec::len))?;
  let (law, jump_a, jump_b, jump_c) = sizes.map_or((0, 0.0, 0.0, 0.0), |s| s.encode());
  let (gamma_law, g1_shape, g1_scale, g1_per, g2_shape, g2_scale, g2_per) =
    gammas.map_or((0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), |g| g.encode());
  let (curve, n_curves) = crate::euler::flatten_curves(curves, n);
  let (lift_tables, has_lift, lift_n, lift_db, lift_fb, lift_x0) =
    crate::euler::encode_lift(lift.as_ref());
  let hist_slot = crate::euler::history_slot(family, n);
  if hist_slot != u32::MAX && hist_slot >= n_curves {
    return Err(DeviceError::Launch(format!(
      "history weights read curve {hist_slot}, but the process supplies {n_curves}"
    )));
  }
  let series_n = crate::euler::series_terms(family, n, series);
  let series_live = crate::euler::series_live(family);
  let (table_n, table_u0) = crate::euler::table_terms(family, table);
  let args = EulerArgs {
    family,
    components: components as u32,
    noises: noises as u32,
    seed: (seed ^ (seed >> 32)) as u32,
    steps,
    paths,
    first_path,
    increments: streams,
    n_curves,
    has_jumps: u32::from(jump_lambda.is_some()),
    jump_law: law,
    step_first: u32::from(step_first),
    gamma_law,
    dt,
    sqrt_dt: dt.sqrt(),
    jump_lambda: jump_lambda.unwrap_or(0.0),
    jump_a,
    jump_b,
    jump_c,
    g1_shape,
    g1_scale,
    g1_per,
    g2_shape,
    g2_scale,
    g2_per,
    has_lift,
    lift_n,
    lift_db,
    lift_fb,
    lift_x0,
    hist_slot,
    series_n,
    series_live,
    table_n,
    program_n,
    table_u0,
    x0,
  };
  run(
    ordinal,
    params,
    args,
    increments,
    &curve,
    lift_tables,
    &program_buf,
    |data| finish(data, components),
  )
}

/// The frame reads one or two streams, each a row of one increment per step.
fn validate_increments(
  len: usize,
  paths: usize,
  steps: usize,
  streams: u32,
  row: usize,
) -> Result<()> {
  if !(1..=2).contains(&streams) || row != steps {
    return Err(DeviceError::Launch(format!(
      "the Euler frame reads one or two rows of {steps} increments, not {streams} of {row}"
    )));
  }
  let needed = paths
    .checked_mul(streams as usize)
    .and_then(|rows| rows.checked_mul(row));
  if needed.is_none_or(|needed| needed > len) {
    return Err(DeviceError::Launch(
      "the increment buffer does not cover every path".into(),
    ));
  }
  Ok(())
}

fn validate_curve_extent(steps: u32, curves: usize) -> Result<()> {
  let curves = crate::device::launch_len::<u32>(curves, "curves")?;
  steps
    .checked_mul(curves)
    .ok_or_else(|| DeviceError::Launch("curves * steps exceeds the kernel index range".into()))?;
  Ok(())
}

/// Returns an owned `components × m × n` output array.
#[allow(clippy::too_many_arguments)]
pub(super) fn device_paths(
  ordinal: usize,
  spec: EulerSpec<f32>,
  x0: [f32; 4],
  n: usize,
  dt: f32,
  first: usize,
  m: usize,
  seed: u64,
  increments: Increments<'_>,
  curves: Option<Vec<Vec<f32>>>,
  jump_lambda: Option<f32>,
  sizes: Option<crate::euler::JumpSizes<f32>>,
  step_first: bool,
  gammas: Option<crate::euler::GammaDraws<f32>>,
  lift: Option<crate::euler::LiftSpec<'_, f32>>,
  series: Option<u32>,
  table: Option<crate::euler::TableSpec<f32>>,
  program: Option<crate::euler::ProgramSpec<'_>>,
) -> Result<Array3<f32>> {
  launch_paths(
    ordinal,
    spec,
    x0,
    n,
    dt,
    first,
    m,
    seed,
    increments,
    curves,
    jump_lambda,
    sizes,
    step_first,
    gammas,
    lift,
    series,
    table,
    program,
    |data, components| {
      Array3::from_shape_vec((components, m, n), data.to_vec())
        .expect("the kernel returns components * m * n values")
    },
  )
}

#[cfg(test)]
mod tests {
  use super::{validate_curve_extent, validate_increments};

  #[test]
  fn increments_cover_every_stream_row_before_launch() {
    assert!(validate_increments(12, 3, 4, 1, 4).is_ok());
    assert!(validate_increments(24, 3, 4, 2, 4).is_ok());
    assert!(validate_increments(23, 3, 4, 2, 4).is_err());
    assert!(validate_increments(100, 3, 4, 1, 5).is_err());
    assert!(validate_increments(100, 3, 4, 0, 4).is_err());
    assert!(validate_increments(100, 3, 4, 3, 4).is_err());
    assert!(validate_increments(usize::MAX, usize::MAX, 4, 2, 4).is_err());
  }

  #[test]
  fn curve_indices_fit_the_kernel_before_flattening() {
    assert!(validate_curve_extent(u32::MAX, 0).is_ok());
    assert!(validate_curve_extent(u32::MAX, 1).is_ok());
    assert!(validate_curve_extent(u32::MAX / 6, 6).is_ok());
    assert!(validate_curve_extent(u32::MAX / 6 + 1, 6).is_err());
    assert!(validate_curve_extent(1 << 30, 6).is_err());
    assert!(validate_curve_extent(1, usize::MAX).is_err());
  }
}
