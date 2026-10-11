use std::cell::RefCell;
use std::rc::Rc;

use metal_oxide::{Argument, Batch, Buffer, Dim3, DynamicLaunchConfig, Pipeline};

use super::{Result, SheetLaunch, launch_len};
use crate::device::metal::{self, device_error};
use crate::noise::fgn::metal::build_bit_reverse_table;

pub(super) struct Context {
  common: Rc<metal::Context>,
  generate: Pipeline,
  butterfly: Pipeline,
  tile: Pipeline,
  transpose: Pipeline,
  extract: Pipeline,
  sized: Vec<SizedSheet>,
}

struct SizedSheet {
  real: Buffer<f32>,
  imag: Buffer<f32>,
  real_t: Buffer<f32>,
  imag_t: Buffer<f32>,
  output: Buffer<f32>,
  lam: Buffer<f32>,
  rev_cols: Buffer<u32>,
  rev_rows: Buffer<u32>,
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
    let pipeline = |name| Pipeline::new(&common.device, &common.module, name).map_err(device_error);
    Ok(Self {
      generate: pipeline("sheet_generate")?,
      butterfly: pipeline("sheet_fft_butterfly")?,
      tile: pipeline("sheet_fft_tile")?,
      transpose: pipeline("sheet_transpose")?,
      extract: pipeline("sheet_extract")?,
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
    let cols = launch_len::<u32>(big_n, "cols")?;
    let m_u32 = launch_len::<u32>(m, "m")?;
    let n_u32 = launch_len::<u32>(n, "n")?;
    // Public parameters can change without rebuilding the stored embedding.
    let buffers = crate::device::lru_slot(
      &mut self.sized,
      |s| {
        s.m == m
          && s.n == n
          && s.sheets == sheets
          && s.key == sheet.key
          && s.lam.as_slice() == sheet.lam
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
          rev_cols: self
            .common
            .device
            .buffer_from_slice(&build_bit_reverse_table(big_n))
            .map_err(device_error)?,
          rev_rows: self
            .common
            .device
            .buffer_from_slice(&build_bit_reverse_table(big_m))
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
      rev_cols,
      rev_rows,
      ..
    } = buffers;
    let first_cell = (first * cells) as u64;
    // SAFETY: each launch owns disjoint buffers and bounds all rounded dispatches.
    // Complete rows and columns use valid radix-2 stages; Batch orders their dependencies.
    unsafe {
      self.common.device.submit(|batch| {
        batch.launch(
          &self.generate,
          DynamicLaunchConfig::for_elements(total, 256)?,
          &[
            Argument::write(real),
            Argument::write(imag),
            Argument::read(lam),
            Argument::read(rev_cols),
            Argument::value(cells_u32)?,
            Argument::value(cols)?,
            Argument::value(seed)?,
            Argument::value(first_cell)?,
            Argument::value(total)?,
          ],
        )?;
        encode_fft(batch, &self.tile, &self.butterfly, real, imag, cols, total)?;
        batch.launch(
          &self.transpose,
          DynamicLaunchConfig::for_elements(total, 256)?,
          &[
            Argument::read(real),
            Argument::read(imag),
            Argument::write(real_t),
            Argument::write(imag_t),
            Argument::read(rev_rows),
            Argument::value(cells_u32)?,
            Argument::value(rows)?,
            Argument::value(cols)?,
            Argument::value(total)?,
          ],
        )?;
        encode_fft(
          batch,
          &self.tile,
          &self.butterfly,
          real_t,
          imag_t,
          rows,
          total,
        )?;
        batch.launch(
          &self.extract,
          DynamicLaunchConfig::for_elements(out_len, 256)?,
          &[
            Argument::read(real_t),
            Argument::write(output),
            Argument::value(cells_u32)?,
            Argument::value(rows)?,
            Argument::value(m_u32)?,
            Argument::value(n_u32)?,
            Argument::value(sheet.r)?,
            Argument::value(sheet.corr)?,
            Argument::value(seed)?,
            Argument::value(correction_cell)?,
            Argument::value(out_len)?,
          ],
        )
      })
    }
    .and_then(|submission| submission.wait())
    .map_err(device_error)?;
    Ok(output.as_slice())
  }
}

/// # Safety
/// `n` is a power of two; `real` and `imag` each hold `total / n` complete
/// length-`n` transforms.
unsafe fn encode_fft(
  batch: &mut Batch<'_>,
  tile_pipeline: &Pipeline,
  butterfly: &Pipeline,
  real: &mut Buffer<f32>,
  imag: &mut Buffer<f32>,
  n: u32,
  total: u32,
) -> metal_oxide::Result<()> {
  let mut stages = n.trailing_zeros().min(11);
  while (1_usize << (stages - 1)) > tile_pipeline.max_threads_per_block() {
    stages -= 1;
  }
  let tile = 1_u32 << stages;
  // SAFETY: a tile divides each complete transform and all its lanes reach every barrier.
  unsafe {
    batch.launch(
      tile_pipeline,
      DynamicLaunchConfig::new(Dim3::x(total / tile), Dim3::x(tile / 2)),
      &[
        Argument::read_write(real),
        Argument::read_write(imag),
        Argument::value(stages)?,
        Argument::threadgroup::<f32>(tile as usize)?,
        Argument::threadgroup::<f32>(tile as usize)?,
      ],
    )?;
  }
  let count = total / 2;
  for stage in stages..n.trailing_zeros() {
    // SAFETY: a power-of-two transform partitions into disjoint butterflies at each stage.
    unsafe {
      batch.launch(
        butterfly,
        DynamicLaunchConfig::for_elements(count, 256)?,
        &[
          Argument::read_write(real),
          Argument::read_write(imag),
          Argument::value(n)?,
          Argument::value(1_u32 << stage)?,
          Argument::value(count)?,
        ],
      )?;
    }
  }
  Ok(())
}
