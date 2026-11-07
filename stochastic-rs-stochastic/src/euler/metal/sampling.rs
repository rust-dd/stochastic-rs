use crate::noise::fgn::metal::OutputRental;
use ndarray::{Array2, Array3};

use super::{Increments, Result, device_paths, launch_paths};
use crate::device::metal::working_set;
use crate::device::{DeviceError, Metal};
use crate::euler::{EulerCoefficients, EulerKernel, EulerSpec, FgnSpec};

impl EulerKernel<f32> for Metal {
  fn euler_kernel<P: EulerCoefficients<f32>>(
    &self,
    process: &P,
    first: usize,
    m: usize,
    seed: u64,
  ) -> Result<Array2<f32>> {
    let fractional = fgn_buffer(process.fgn_spec(), first, m, seed, self.ordinal)?;
    let planes = device_paths(
      self.ordinal,
      process.euler_spec(),
      process.initial_state(),
      process.grid_points(),
      process.time_step(),
      first,
      m,
      seed,
      increments(fractional.as_ref()),
      process.curves(),
      process.jump_intensity(),
      process.jump_sizes(),
      process.step_first(),
      process.gamma_draws(),
      process.lift_spec(),
      process.series_terms(),
      process.table_spec(),
      process.program_spec(),
    )?;
    Ok(planes.index_axis_move(ndarray::Axis(0), 0))
  }

  fn euler_kernel_lend<P: EulerCoefficients<f32>, O>(
    &self,
    process: &P,
    first: usize,
    m: usize,
    seed: u64,
    f: impl FnOnce(ndarray::ArrayView2<f32>) -> O,
  ) -> Result<O> {
    let fractional = fgn_buffer(process.fgn_spec(), first, m, seed, self.ordinal)?;
    let n = process.grid_points();
    launch_paths(
      self.ordinal,
      process.euler_spec(),
      process.initial_state(),
      n,
      process.time_step(),
      first,
      m,
      seed,
      increments(fractional.as_ref()),
      process.curves(),
      process.jump_intensity(),
      process.jump_sizes(),
      process.step_first(),
      process.gamma_draws(),
      process.lift_spec(),
      process.series_terms(),
      process.table_spec(),
      process.program_spec(),
      // The scalar sampling interface exposes the first reported component.
      |data, _| {
        f(ndarray::ArrayView2::from_shape((m, n), &data[..m * n])
          .expect("m * n values in the first plane"))
      },
    )
  }

  fn euler_system_kernel<const D: usize, P: crate::euler::EulerSystem<f32, D>>(
    &self,
    process: &P,
    first: usize,
    m: usize,
    seed: u64,
  ) -> Result<Array3<f32>> {
    let spec = process.euler_spec();
    crate::euler::check_arity(&spec, D);
    let slots = process.initial_state();
    let fractional = fgn_buffer(process.fgn_spec(), first, m, seed, self.ordinal)?;
    device_paths(
      self.ordinal,
      spec,
      slots,
      process.grid_points(),
      process.time_step(),
      first,
      m,
      seed,
      increments(fractional.as_ref()),
      process.curves(),
      process.jump_intensity(),
      process.jump_sizes(),
      process.step_first(),
      process.gamma_draws(),
      process.lift_spec(),
      process.series_terms(),
      process.table_spec(),
      process.program_spec(),
    )
  }

  fn batch_budget(&self) -> usize {
    self.batch_budget.min(working_set(self.ordinal))
  }

  /// Every buffer index is a `uint`: a path writes `components` planes of `steps` values and, when
  /// fractional, reads `streams` increment rows the fGN pipeline wrote, indexed in `uint` as well.
  fn index_rows(
    &self,
    rows: usize,
    spec: EulerSpec<f32>,
    steps: usize,
    fgn: Option<FgnSpec<'_, f32>>,
  ) -> usize {
    let (family, _) = spec.encode();
    let planes = crate::euler::families::Family::from_code(family)
      .expect("a declared family")
      .components();
    let streams = fgn.as_ref().map_or(0, |f| f.streams);
    let rows = crate::device::rows_within_index_limit(
      rows,
      planes.max(streams).saturating_mul(steps),
      u32::MAX as usize,
    );
    match fgn {
      Some(f) => crate::noise::fgn::metal::index_rows(rows, f.streams, f.n),
      None => rows,
    }
  }
}

/// Fractional streams share one embedding and remain adjacent within each path.
/// The Euler frame reads stream `s` of path `p` from row `p * streams + s`.
fn fgn_buffer(
  spec: Option<crate::euler::FgnSpec<'_, f32>>,
  first: usize,
  m: usize,
  seed: u64,
  ordinal: usize,
) -> Result<Option<(OutputRental, u32, usize)>> {
  match spec {
    Some(spec) => {
      let streams = crate::device::launch_len::<u32>(spec.streams, "increment streams")?;
      crate::euler::launch_scalars(spec.n, first, m, streams)?;
      // Counters skip every preceding path's streams, preserving draws across chunks.
      let produced = [first, 2, spec.n]
        .into_iter()
        .try_fold(spec.streams as u64, |total, factor| {
          total.checked_mul(factor as u64)
        })
        .ok_or_else(|| DeviceError::Launch("fGN counter offset exceeds u64".into()))?;
      let (buf, row) = crate::noise::fgn::metal::sample_f32_buffer(
        spec.sqrt_eigenvalues,
        spec.n,
        spec.streams * m,
        spec.offset,
        spec.hurst,
        spec.t,
        seed as u32,
        produced,
        ordinal,
      )?;
      Ok(Some((buf, streams, row)))
    }
    None => Ok(None),
  }
}

fn increments(fractional: Option<&(OutputRental, u32, usize)>) -> Increments<'_> {
  match fractional {
    Some((buffer, streams, row)) => Increments::Device {
      buffer: buffer.as_buffer(),
      streams: *streams,
      row: *row,
    },
    None => Increments::Hashed,
  }
}
