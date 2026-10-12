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

/// Largest tile, so two padded scratch arrays stay well inside threadgroup memory.
const MAX_TILE_LOG: usize = 11;

/// Tile stages, then column passes of two or three stages (only the read-out may take one).
/// Depends on the length alone, so rows are computed alike however a batch is cut.
fn plan(log_n: usize, max_threads: usize) -> (usize, Vec<usize>) {
  let mut tile_log = log_n.min(MAX_TILE_LOG);
  while tile_log > 1 && 1 << (tile_log - 1) > max_threads {
    tile_log -= 1;
  }
  let remaining = log_n - tile_log;
  let passes = remaining.div_ceil(3);
  let columns = (0..passes)
    .map(|pass| remaining / passes + usize::from(pass < remaining % passes))
    .collect();
  (tile_log, columns)
}

/// Interleaved cos/sin of -2πk/len for k < len / 2, evaluated in f64.
fn twiddle_table(len: usize) -> Vec<f32> {
  (0..(len / 2).max(1))
    .flat_map(|k| {
      let angle = -2.0 * std::f64::consts::PI * k as f64 / len as f64;
      [angle.cos() as f32, angle.sin() as f32]
    })
    .collect()
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
  fn plans_depend_on_length_and_split_columns_evenly() {
    assert_eq!(plan(1, 1024), (1, vec![]));
    assert_eq!(plan(11, 1024), (11, vec![]));
    assert_eq!(plan(12, 1024), (11, vec![1]));
    assert_eq!(plan(14, 1024), (11, vec![3]));
    assert_eq!(plan(15, 1024), (11, vec![2, 2]));
    assert_eq!(plan(17, 1024), (11, vec![3, 3]));
    assert_eq!(plan(18, 1024), (11, vec![3, 2, 2]));
    assert_eq!(plan(13, 256), (9, vec![2, 2]));
    assert_eq!(plan(4, 2), (2, vec![2]));
    for log_n in 1..=31 {
      for max_threads in [1, 32, 256, 1024] {
        let (tile_log, columns) = plan(log_n, max_threads);
        assert_eq!(tile_log + columns.iter().sum::<usize>(), log_n);
        assert!(columns.iter().all(|&stages| stages <= 3));
        assert!(columns.iter().rev().skip(1).all(|&stages| stages >= 2));
      }
    }
  }

  #[test]
  fn twiddles_hold_half_a_turn_at_full_precision() {
    let table = twiddle_table(8);
    assert_eq!(table.len(), 8);
    assert_eq!((table[0], table[1]), (1.0, 0.0));
    assert_eq!(
      (table[4], table[5]),
      ((-std::f64::consts::FRAC_PI_2).cos() as f32, -1.0)
    );
    assert_eq!(twiddle_table(1), vec![1.0, 0.0]);
  }

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
