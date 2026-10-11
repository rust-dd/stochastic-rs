//! Circulant-embedding fGN with a paired FFT and zero-copy Metal readout.
//! Dietrich & Newsam (1997), DOI: 10.1137/S1064827592240555.

mod context;
mod kernels;
mod pool;
mod sampling;

use metal::Buffer;
use parking_lot::Mutex;

use super::Fgn;
use crate::device::{DeviceError, launch_len};
use crate::euler::metal::checked_buffer;

pub(crate) use kernels::MSL_COMMON;
pub(crate) use pool::sample_f32_buffer;
pub(crate) use sampling::index_rows;

type Result<T> = std::result::Result<T, DeviceError>;

fn shared_buffer(device: &metal::Device, bytes: u64) -> Result<Buffer> {
  let bytes = bytes.max(4);
  let buffer = checked_buffer(
    device.new_buffer(bytes, metal::MTLResourceOptions::StorageModeShared),
    bytes,
  )?;
  // SAFETY: this new shared allocation is exclusively owned and has `bytes` writable bytes.
  unsafe { std::ptr::write_bytes(buffer.contents().cast::<u8>(), 0, bytes as usize) };
  Ok(buffer)
}

#[derive(Clone, Copy)]
struct Launch<'a> {
  sqrt_eigs: &'a [f32],
  n: usize,
  m: usize,
  offset: usize,
  hurst: f64,
  t: f64,
  seed: u32,
  first_cell: u64,
  ordinal: usize,
}

impl Launch<'_> {
  fn out_len(&self) -> usize {
    self.m * (self.n - self.offset)
  }
}

fn tile_plan(log_n: usize, max_threads: usize) -> (usize, usize) {
  let mut tile_log = log_n.min(11);
  while tile_log > 1 && (1usize << (tile_log - 1)) > max_threads {
    tile_log -= 1;
  }
  let mut stages = tile_log.min(log_n - 1);
  if (log_n - stages) % 2 == 1 && stages >= 1 {
    stages -= 1;
  }
  (tile_log, stages)
}

pub(crate) fn build_bit_reverse_table(n: usize) -> Vec<u32> {
  let log_n = n.trailing_zeros() as usize;
  let bits = usize::BITS as usize;
  (0..n)
    .map(|i| (i.reverse_bits() >> (bits - log_n)) as u32)
    .collect()
}
