use std::cell::RefCell;
use std::rc::Rc;

use metal_oxide::{
  Argument, Batch, Buffer, Dim3, DynamicLaunchConfig, Launch, Operation, Pipeline, sequence,
};

use super::{Result, SheetLaunch, launch_len};
use crate::device::DeviceError;
use crate::device::metal::{self, device_error};
use crate::noise::fgn::metal::{plan, twiddle_table};

pub(super) struct Context {
  common: Rc<metal::Context>,
  rows: Pipeline,
  tile: Pipeline,
  /// In-place passes of one, two and three stages.
  columns: Vec<Pipeline>,
  transpose: Pipeline,
  extract: Pipeline,
  max_threads: usize,
  sized: Vec<SizedSheet>,
}

struct SizedSheet {
  real: Buffer<f32>,
  imag: Buffer<f32>,
  real_t: Buffer<f32>,
  imag_t: Buffer<f32>,
  output: Buffer<f32>,
  lam: Buffer<f32>,
  row_twiddles: Buffer<f32>,
  column_twiddles: Buffer<f32>,
  m: usize,
  n: usize,
  sheets: usize,
  key: (u64, u64),
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
    let rows = pipeline("sheet_rows")?;
    let tile = pipeline("fft_tile")?;
    if rows.thread_execution_width() != 32 || tile.thread_execution_width() != 32 {
      return Err(DeviceError::Unavailable(
        "sheet tiles need 32-lane SIMD groups".into(),
      ));
    }
    Ok(Self {
      columns: (1..=3)
        .map(|stages| pipeline(&format!("fft_columns_{stages}")))
        .collect::<Result<Vec<_>>>()?,
      transpose: pipeline("sheet_transpose")?,
      extract: pipeline("sheet_extract")?,
      max_threads: rows
        .max_threads_per_block()
        .min(tile.max_threads_per_block()),
      rows,
      tile,
      common,
      sized: Vec::new(),
    })
  }

  pub(super) fn sample_chunk(
    &mut self,
    sheet: &SheetLaunch<'_, f32>,
    sheets: usize,
    first: usize,
    correction_cell: u64,
    seed: u32,
  ) -> Result<&[f32]> {
    let (m, n) = (sheet.m, sheet.n);
    let big_m = 2 * (m - 1);
    let big_n = 2 * (n - 1);
    let cells = big_m * big_n;
    let total = launch_len::<u32>(sheets * cells, "total")?;
    let out_len = launch_len::<u32>(sheets * m * n, "out_len")?;
    let cells_u32 = launch_len::<u32>(cells, "cells")?;
    let rows = launch_len::<u32>(big_m, "rows")?;
    let m_u32 = launch_len::<u32>(m, "m")?;
    let n_u32 = launch_len::<u32>(n, "n")?;
    let (log_rows, log_cols) = (big_m.trailing_zeros(), big_n.trailing_zeros());
    let row_plan = plan(log_cols as usize, self.max_threads);
    let column_plan = plan(log_rows as usize, self.max_threads);
    // Public parameters can change without rebuilding the stored embedding.
    let buffers = crate::device::lru_slot(
      &mut self.sized,
      |s| {
        s.m == m
          && s.n == n
          && s.sheets == sheets
          && s.key == sheet.key
          && s.lam.read().wait() == sheet.lam
      },
      || -> Result<_> {
        let floats = |len| {
          self
            .common
            .device
            .buffer_zeroed::<f32>(len)
            .map_err(device_error)
        };
        Ok(SizedSheet {
          real: floats(total as usize)?,
          imag: floats(total as usize)?,
          real_t: floats(total as usize)?,
          imag_t: floats(total as usize)?,
          output: floats(out_len as usize)?,
          lam: self
            .common
            .device
            .buffer_from_slice(sheet.lam)
            .map_err(device_error)?,
          row_twiddles: self
            .common
            .device
            .buffer_from_slice(&twiddle_table(big_n))
            .map_err(device_error)?,
          column_twiddles: self
            .common
            .device
            .buffer_from_slice(&twiddle_table(big_m))
            .map_err(device_error)?,
          m,
          n,
          sheets,
          key: sheet.key,
        })
      },
    )?;
    let SizedSheet {
      real,
      imag,
      real_t,
      imag_t,
      output,
      lam,
      row_twiddles,
      column_twiddles,
      ..
    } = buffers;
    let first_cell = (first * cells) as u64;
    let row_tile = row_plan.0 as u32;
    let column_tile = column_plan.0 as u32;
    // SAFETY: tiles and column groups partition complete rows, then complete columns; every
    // tile thread reaches each barrier and shuffle. Batch orders the dependent launches.
    unsafe {
      sequence(|batch| {
        batch.push(Launch::new(
          Ok(&self.rows),
          tile_config(total, row_tile),
          (
            &mut *real,
            &mut *imag,
            &*lam,
            &*row_twiddles,
            (log_cols, log_rows + log_cols, row_tile, seed, first_cell),
            scratch(row_tile),
          ),
          |(real, imag, lam, twiddles, (log_cols, log_n, tile, seed, first_cell), scratch)| {
            Ok([
              Argument::write(real),
              Argument::write(imag),
              Argument::read(lam),
              Argument::read(twiddles),
              Argument::value(*log_cols)?,
              Argument::value(*log_n)?,
              Argument::value(*tile)?,
              Argument::value(*seed)?,
              Argument::value(*first_cell)?,
              Argument::threadgroup::<f32>(*scratch)?,
              Argument::threadgroup::<f32>(*scratch)?,
            ])
          },
        ))?;
        let row_pass = Transforms {
          log_n: log_cols,
          twiddles: row_twiddles,
          total,
        };
        encode_passes(
          batch,
          &self.columns,
          real,
          imag,
          row_pass,
          row_tile,
          &row_plan.1,
        )?;
        batch.push(Launch::new(
          Ok(&self.transpose),
          DynamicLaunchConfig::for_elements(total, 256)?,
          (
            &*real,
            &*imag,
            &mut *real_t,
            &mut *imag_t,
            (log_rows, log_cols, total),
          ),
          |(real, imag, real_t, imag_t, (log_rows, log_cols, total))| {
            Ok([
              Argument::read(real),
              Argument::read(imag),
              Argument::write(real_t),
              Argument::write(imag_t),
              Argument::value(*log_rows)?,
              Argument::value(*log_cols)?,
              Argument::value(*total)?,
            ])
          },
        ))?;
        batch.push(Launch::new(
          Ok(&self.tile),
          tile_config(total, column_tile),
          (
            &mut *real_t,
            &mut *imag_t,
            &*column_twiddles,
            (log_rows, column_tile),
            scratch(column_tile),
          ),
          |(real_t, imag_t, twiddles, (log_rows, tile), scratch)| {
            Ok([
              Argument::read_write(real_t),
              Argument::read_write(imag_t),
              Argument::read(twiddles),
              Argument::value(*log_rows)?,
              Argument::value(*tile)?,
              Argument::threadgroup::<f32>(*scratch)?,
              Argument::threadgroup::<f32>(*scratch)?,
            ])
          },
        ))?;
        let column_pass = Transforms {
          log_n: log_rows,
          twiddles: column_twiddles,
          total,
        };
        encode_passes(
          batch,
          &self.columns,
          real_t,
          imag_t,
          column_pass,
          column_tile,
          &column_plan.1,
        )?;
        batch
          .push(Launch::new(
            Ok(&self.extract),
            DynamicLaunchConfig::for_elements(out_len, 256)?,
            (
              &*real_t,
              &mut *output,
              (cells_u32, rows, m_u32, n_u32),
              (sheet.r, sheet.corr, seed, correction_cell, out_len),
            ),
            |(real_t, output, (cells, rows, m, n), (r, corr, seed, correction_cell, out_len))| {
              Ok([
                Argument::read(real_t),
                Argument::write(output),
                Argument::value(*cells)?,
                Argument::value(*rows)?,
                Argument::value(*m)?,
                Argument::value(*n)?,
                Argument::value(*r)?,
                Argument::value(*corr)?,
                Argument::value(*seed)?,
                Argument::value(*correction_cell)?,
                Argument::value(*out_len)?,
              ])
            },
          ))
          .map(drop)
      })
      .wait()
    }
    .map_err(device_error)?;
    Ok(output.read().wait())
  }
}

/// One block per tile of 2^tile_log values, two values per thread.
fn tile_config(total: u32, tile_log: u32) -> DynamicLaunchConfig {
  DynamicLaunchConfig::new(Dim3::x(total >> tile_log), Dim3::x(1 << (tile_log - 1)))
}

/// Padded threadgroup words for one tile of 2^tile_log values.
fn scratch(tile_log: u32) -> usize {
  let tile = 1_usize << tile_log;
  tile + tile / 32
}

/// Complete length-2^log_n transforms covering `total` values.
struct Transforms<'a> {
  log_n: u32,
  twiddles: &'a Buffer<f32>,
  total: u32,
}

/// Runs the stages after the tile in place. Safety: `real`/`imag` hold `transforms` done below
/// `tile_log`, `passes` sum to the rest and the twiddles hold 2^(log_n - 1) entries.
unsafe fn encode_passes(
  batch: &mut Batch,
  columns: &[Pipeline],
  real: &mut Buffer<f32>,
  imag: &mut Buffer<f32>,
  transforms: Transforms<'_>,
  tile_log: u32,
  passes: &[usize],
) -> metal_oxide::Result<()> {
  let mut s0 = tile_log;
  for &stages in passes {
    let threads = transforms.total >> stages;
    // SAFETY: each thread owns 2^stages positions of one complete transform.
    unsafe {
      batch.push(Launch::new(
        Ok(&columns[stages - 1]),
        DynamicLaunchConfig::for_elements(threads, 256)?,
        (
          &mut *real,
          &mut *imag,
          transforms.twiddles,
          (transforms.log_n, s0, threads),
        ),
        |(real, imag, twiddles, (log_n, s0, threads))| {
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
    }
    s0 += stages as u32;
  }
  Ok(())
}
