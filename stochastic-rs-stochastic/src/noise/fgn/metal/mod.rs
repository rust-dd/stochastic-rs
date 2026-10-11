//! Circulant-embedding fGN with a paired FFT and zero-copy Metal readout.
//! Dietrich & Newsam (1997), DOI: 10.1137/S1064827592240555.

mod context;
mod pool;
mod sampling;

use super::Fgn;
use crate::device::{DeviceError, launch_len};

pub(crate) use pool::{OutputRental, sample_f32_buffer};
pub(crate) use sampling::index_rows;

type Result<T> = std::result::Result<T, DeviceError>;

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
  fn out_len(&self) -> Result<usize> {
    let embedding = self
      .n
      .checked_mul(2)
      .ok_or_else(|| DeviceError::Launch("fGN embedding dimensions overflow".into()))?;
    if !self.n.is_power_of_two() || self.offset > self.n || self.sqrt_eigs.len() != embedding {
      return Err(DeviceError::Launch(
        "invalid fGN embedding dimensions".into(),
      ));
    }
    launch_len::<u32>(embedding, "fGN embedding")?;
    launch_len::<i32>(self.m, "fGN rows")?;
    let len = self
      .m
      .checked_mul(self.n - self.offset)
      .ok_or_else(|| DeviceError::Launch("fGN output dimensions overflow".into()))?;
    launch_len::<u32>(len, "fGN output")?;
    Ok(len)
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

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn launch_dimensions_reject_invalid_spectra_and_wrapped_lengths() {
    let valid = Launch {
      sqrt_eigs: &[1.0; 8],
      n: 4,
      m: 3,
      offset: 1,
      hurst: 0.5,
      t: 1.0,
      seed: 0,
      first_cell: 0,
      ordinal: 0,
    };
    assert_eq!(valid.out_len(), Ok(9));
    assert_eq!(Launch { offset: 4, ..valid }.out_len(), Ok(0));
    for invalid in [
      Launch { n: 0, ..valid },
      Launch { n: 3, ..valid },
      Launch {
        n: usize::MAX,
        ..valid
      },
      Launch { offset: 5, ..valid },
      Launch {
        sqrt_eigs: &[1.0; 7],
        ..valid
      },
      Launch {
        m: usize::MAX,
        ..valid
      },
      Launch {
        m: i32::MAX as usize,
        ..valid
      },
    ] {
      assert!(matches!(invalid.out_len(), Err(DeviceError::Launch(_))));
    }
  }
}
