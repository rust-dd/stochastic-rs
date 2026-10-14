use std::cell::RefCell;
use std::rc::Rc;

use metal_oxide::{Argument, Buffer, Dim3, DynamicLaunchConfig, Operation, Pipeline, sequence};

use super::{Launch, Result, build_bit_reverse_table, launch_len, plan, twiddle_table};
use crate::device::DeviceError;
use crate::device::metal::{self, Context as DeviceContext, device_error};

pub(super) struct Context {
  pub(super) common: Rc<DeviceContext>,
  tile: Pipeline,
  single: Pipeline,
  /// Passes of two and three stages; a single-stage pass only ever reads out.
  columns: Vec<Pipeline>,
  extract: Vec<Pipeline>,
  max_threads: usize,
  sized: Vec<SizedMetal>,
}

struct SizedMetal {
  eigs: Buffer<f32>,
  tile_twiddles: Buffer<f32>,
  scratch: Option<Scratch>,
  n: usize,
  m: usize,
  offset: usize,
  parity: u32,
  hurst_bits: u64,
  t_bits: u64,
}

/// Intermediate transforms and full-length twiddles for the column passes.
struct Scratch {
  real: Buffer<f32>,
  imag: Buffer<f32>,
  twiddles: Buffer<f32>,
}

thread_local! {
  static CONTEXT: RefCell<Option<(usize, Context)>> = const { RefCell::new(None) };
}

pub(super) fn with<R>(
  ordinal: usize,
  use_context: impl FnOnce(&mut Context) -> Result<R>,
) -> Result<R> {
  CONTEXT.with_borrow_mut(|slot| {
    if slot.as_ref().is_none_or(|(index, _)| *index != ordinal) {
      *slot = Some((ordinal, Context::new(ordinal)?));
    }
    use_context(&mut slot.as_mut().unwrap().1)
  })
}

impl Context {
  fn new(ordinal: usize) -> Result<Self> {
    let common = metal::context(ordinal)?;
    let pipeline =
      |name: &str| Pipeline::new(&common.device, &common.module, name).map_err(device_error);
    let tile = pipeline("gen_tile_fft")?;
    let single = pipeline("gen_fft_extract")?;
    if tile.thread_execution_width() != 32 || single.thread_execution_width() != 32 {
      return Err(DeviceError::Unavailable(
        "fGN tiles need 32-lane SIMD groups".into(),
      ));
    }
    let columns = (2..=3)
      .map(|stages| pipeline(&format!("fft_columns_{stages}")))
      .collect::<Result<Vec<_>>>()?;
    let extract = (1..=3)
      .map(|stages| pipeline(&format!("fft_columns_extract_{stages}")))
      .collect::<Result<Vec<_>>>()?;
    Ok(Self {
      max_threads: tile
        .max_threads_per_block()
        .min(single.max_threads_per_block()),
      tile,
      single,
      columns,
      extract,
      common,
      sized: Vec::new(),
    })
  }

  pub(super) fn run(&mut self, launch: Launch<'_>, output: &mut Buffer<f32>) -> Result<usize> {
    let Launch {
      sqrt_eigs,
      n,
      m,
      offset,
      hurst,
      t,
      seed,
      first_cell,
      ..
    } = launch;
    let traj_size = 2 * n;
    let out_size = n - offset;
    if m == 0 {
      return Ok(out_size);
    }
    let scale = (out_size.max(1) as f32).powf(-(hurst as f32)) * (t as f32).powf(hurst as f32);
    let parity = ((first_cell / traj_size as u64) & 1) as u32;
    let base_cell = first_cell - u64::from(parity) * traj_size as u64;
    let transforms = (parity as usize + m).div_ceil(2);
    let total = launch_len::<u32>(transforms * traj_size, "real")?;
    launch_len::<u32>(m * out_size, "output")?;
    launch_len::<i32>(m, "rows")?;
    let size = launch_len::<u32>(out_size, "out_size")?;
    let rows = launch_len::<u32>(m, "rows")?;
    let log_n = traj_size.trailing_zeros() as usize;
    let (tile_log, columns) = plan(log_n, self.max_threads);
    let device = &self.common.device;
    let buffers = crate::device::lru_slot(
      &mut self.sized,
      |s| {
        s.n == n
          && s.m == m
          && s.offset == offset
          && s.parity == parity
          && s.hurst_bits == hurst.to_bits()
          && s.t_bits == t.to_bits()
      },
      || -> Result<_> {
        let eigs = build_bit_reverse_table(traj_size)
          .iter()
          .map(|&natural| sqrt_eigs[natural as usize])
          .collect::<Vec<_>>();
        let scratch = if columns.is_empty() {
          None
        } else {
          Some(Scratch {
            real: device.buffer_zeroed(total as usize).map_err(device_error)?,
            imag: device.buffer_zeroed(total as usize).map_err(device_error)?,
            twiddles: device
              .buffer_from_slice(&twiddle_table(traj_size))
              .map_err(device_error)?,
          })
        };
        Ok(SizedMetal {
          eigs: device.buffer_from_slice(&eigs).map_err(device_error)?,
          tile_twiddles: device
            .buffer_from_slice(&twiddle_table(1 << tile_log))
            .map_err(device_error)?,
          scratch,
          n,
          m,
          offset,
          parity,
          hurst_bits: hurst.to_bits(),
          t_bits: t.to_bits(),
        })
      },
    )?;
    let log_n = log_n as u32;
    let tile = 1_u32 << tile_log;
    let shared = (tile + tile / 32) as usize;
    let SizedMetal {
      eigs,
      tile_twiddles,
      scratch,
      ..
    } = buffers;
    // SAFETY: tiles and column groups partition complete transforms; every tile thread reaches
    // each barrier and shuffle. The planner bounds every buffer and the readout guards partial rows.
    unsafe {
      sequence(|batch| {
        let Some(Scratch {
          real,
          imag,
          twiddles,
        }) = scratch
        else {
          return batch
            .push(metal_oxide::Launch::new(
              Ok(&self.single),
              DynamicLaunchConfig::new(Dim3::x(total / tile), Dim3::x(tile / 2)),
              (
                &*eigs,
                &*tile_twiddles,
                (log_n, seed, base_cell),
                &mut *output,
                (size, scale, rows, parity),
                shared,
              ),
              |(
                eigs,
                tile_twiddles,
                (log_n, seed, base_cell),
                output,
                (size, scale, rows, parity),
                shared,
              )| {
                Ok([
                  Argument::read(eigs),
                  Argument::read(tile_twiddles),
                  Argument::value(*log_n)?,
                  Argument::value(*seed)?,
                  Argument::value(*base_cell)?,
                  Argument::write(output),
                  Argument::value(*size)?,
                  Argument::value(*scale)?,
                  Argument::value(*rows)?,
                  Argument::value(*parity)?,
                  Argument::threadgroup::<f32>(*shared)?,
                  Argument::threadgroup::<f32>(*shared)?,
                ])
              },
            ))
            .map(drop);
        };
        batch.push(metal_oxide::Launch::new(
          Ok(&self.tile),
          DynamicLaunchConfig::new(Dim3::x(total / tile), Dim3::x(tile / 2)),
          (
            &mut *real,
            &mut *imag,
            &*eigs,
            &*tile_twiddles,
            (log_n, tile_log as u32, seed, base_cell),
            shared,
          ),
          |(real, imag, eigs, tile_twiddles, (log_n, tile_log, seed, base_cell), shared)| {
            Ok([
              Argument::write(real),
              Argument::write(imag),
              Argument::read(eigs),
              Argument::read(tile_twiddles),
              Argument::value(*log_n)?,
              Argument::value(*tile_log)?,
              Argument::value(*seed)?,
              Argument::value(*base_cell)?,
              Argument::threadgroup::<f32>(*shared)?,
              Argument::threadgroup::<f32>(*shared)?,
            ])
          },
        ))?;
        let mut s0 = tile_log as u32;
        for (index, &stages) in columns.iter().enumerate() {
          let threads = total >> stages;
          let config = DynamicLaunchConfig::for_elements(threads, 256)?;
          if index + 1 < columns.len() {
            batch.push(metal_oxide::Launch::new(
              Ok(&self.columns[stages - 2]),
              config,
              (&mut *real, &mut *imag, &*twiddles, [log_n, s0, threads]),
              |(real, imag, twiddles, [log_n, s0, threads])| {
                Ok([
                  Argument::read_write(real),
                  Argument::read_write(imag),
                  Argument::read(twiddles),
                  Argument::value(*log_n)?,
                  Argument::value(*s0)?,
                  Argument::value(*threads)?,
                ])
              },
            ))?;
          } else {
            batch.push(metal_oxide::Launch::new(
              Ok(&self.extract[stages - 1]),
              config,
              (
                &*real,
                &*imag,
                &*twiddles,
                [log_n, s0],
                &mut *output,
                (size, scale, rows, parity, threads),
              ),
              |(
                real,
                imag,
                twiddles,
                [log_n, s0],
                output,
                (size, scale, rows, parity, threads),
              )| {
                Ok([
                  Argument::read(real),
                  Argument::read(imag),
                  Argument::read(twiddles),
                  Argument::value(*log_n)?,
                  Argument::value(*s0)?,
                  Argument::write(output),
                  Argument::value(*size)?,
                  Argument::value(*scale)?,
                  Argument::value(*rows)?,
                  Argument::value(*parity)?,
                  Argument::value(*threads)?,
                ])
              },
            ))?;
          }
          s0 += stages as u32;
        }
        Ok(())
      })
      .wait()
    }
    .map(|_| out_size)
    .map_err(device_error)
  }
}
