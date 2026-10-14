//! Circulant-embedding sheets with ordered Metal transforms and owned readout.
//! Kroese & Botev (2015), Spatial process simulation, §4.4, arXiv:1308.0399.

mod context;

use ndarray::Array2;
use stochastic_rs_core::simd_rng::SeedExt;

use super::{Fbs, SheetLaunch};
use crate::device::{DeviceError, launch_len};
use crate::traits::FloatExt;

type Result<T> = std::result::Result<T, DeviceError>;

impl<T: FloatExt, S: SeedExt, B> Fbs<T, S, B> {
  /// One seed drives the batch; cell and correction counters continue across chunks.
  pub(crate) fn sample_metal_sheets(
    &self,
    sheets: usize,
    device: &crate::device::Metal,
  ) -> Result<Vec<Array2<T>>> {
    let seed = self.seed.next_seed() as u32;
    if sheets == 0 {
      return Ok(Vec::new());
    }
    let (m, n) = (self.m, self.n);
    let cells = self.metal_cells()?;
    let lam = self
      .lam
      .iter()
      .map(|x| x.to_f32().unwrap())
      .collect::<Vec<_>>();
    let launch = SheetLaunch {
      lam: &lam,
      m,
      n,
      r: self.r.to_f32().unwrap(),
      corr: self.correction().to_f32().unwrap(),
      key: self.launch_key(),
    };
    let budget = device
      .batch_budget
      .min(crate::device::metal::working_set(device.ordinal));
    let rows = crate::device::rows_within_index_limit(
      crate::device::chunk_rows(budget, 4 * cells + m * n, 4),
      cells,
      u32::MAX as usize,
    );
    let mut out = Vec::with_capacity(sheets);
    let mut first = 0;
    while first < sheets {
      let len = rows.min(sheets - first);
      let correction_cell = (sheets * cells + first) as u64;
      context::with(device.ordinal, |context| {
        let flat = context.sample_chunk(&launch, len, first, correction_cell, seed)?;
        out.extend(self.sheets_from_flat(flat));
        Ok(())
      })?;
      first += len;
    }
    Ok(out)
  }

  fn metal_cells(&self) -> Result<usize> {
    let side = |size: usize| {
      size
        .checked_sub(1)
        .filter(|value| value.is_power_of_two())
        .and_then(|value| value.checked_mul(2))
        .ok_or_else(|| {
          DeviceError::Launch("sheet embedding sides must be nonzero powers of two".into())
        })
    };
    let shape = (side(self.m)?, side(self.n)?);
    if self.lam.dim() != shape {
      return Err(DeviceError::Launch(
        "sheet dimensions do not match the stored embedding".into(),
      ));
    }
    let cells = self.lam.len();
    launch_len::<u32>(cells, "cells")?;
    Ok(cells)
  }
}

#[cfg(test)]
mod chunk_tests {
  use stochastic_rs_core::simd_rng::Deterministic;

  use super::*;
  use crate::device::Metal;
  use crate::traits::ProcessExt;

  /// Cell and correction counters keep chunked batches identical to one launch.
  #[test]
  fn chunks_are_bit_identical_to_one_launch() {
    let fbs = |device: Metal| {
      Fbs::<f32, _>::new(0.7, 9, 5, 1.0, Deterministic::new(5)).with_backend(device)
    };
    let whole = fbs(Metal::default()).sample_par(7);
    // Two sheets per chunk: four launches for seven sheets.
    let per_sheet = (4 * 4 * 8 * 4 + 9 * 5) * 4;
    let chunked = fbs(Metal::default().with_batch_budget(2 * per_sheet)).sample_par(7);
    assert_eq!(whole, chunked);
    assert_ne!(whole[0], whole[1]);
  }

  /// The process's own seed drives its device sheets: two sheets built from
  /// the same `Deterministic` seed agree, a different seed differs.
  #[test]
  fn a_sheet_honours_its_own_seed_on_the_device() {
    let fbs =
      |seed: u64| Fbs::<f32, _>::new(0.7, 9, 9, 1.0, Deterministic::new(seed)).on::<Metal>();
    assert_eq!(fbs(3).sample_par(3), fbs(3).sample_par(3));
    assert_ne!(fbs(3).sample(), fbs(4).sample());
    assert_eq!(fbs(3).sample(), fbs(3).sample());
  }
}
