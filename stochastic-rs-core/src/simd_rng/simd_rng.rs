//! The single-stream [`SimdRng`]: construction, scalar and batch draws, and [`rand::TryRng`]; bulk
//! fills live in [`super::fill`].

use core::convert::Infallible;

use rand::TryRng;
use wide::f32x8;
use wide::f64x4;
use wide::i32x8;
use wide::u32x8;
use wide::u64x4;

use super::next_global_seed;
use super::xoshiro::F32_MAGIC_X8;
use super::xoshiro::F32_ONE_X8;
use super::xoshiro::F64_MAGIC_X4;
use super::xoshiro::F64_ONE_X4;
use super::xoshiro::Xoshiro128PP8;
use super::xoshiro::Xoshiro256PP4;
use super::xoshiro::splitmix64_next;

/// SIMD pseudo-random generator: two xoshiro engines (64- and 32-bit) and scalar buffers that
/// amortise lane extraction; [`SimdRng::from_seed`] replays, [`SimdRng::new()`] seeds uniquely.
#[derive(Clone, Debug)]
pub struct SimdRng {
  pub(super) f64_engine: Xoshiro256PP4,
  pub(super) f32_engine: Xoshiro128PP8,
  u64_buf: [u64; 4],
  u64_idx: usize,
  f64_scalar_buf: [f64; 8],
  f64_scalar_idx: usize,
  f32_scalar_buf: [f32; 8],
  f32_scalar_idx: usize,
  i32_scalar_buf: [i32; 8],
  i32_scalar_idx: usize,
}

impl SimdRng {
  /// A stream fixed by `seed`, expanded by SplitMix64 into both engines' state; equal seeds give
  /// equal streams.
  #[inline]
  pub fn from_seed(seed: u64) -> Self {
    let mut state = seed;
    let seed64 = splitmix64_next(&mut state);
    let seed32 = splitmix64_next(&mut state);
    Self {
      f64_engine: Xoshiro256PP4::new_from_u64(seed64),
      f32_engine: Xoshiro128PP8::new_from_u64(seed32),
      u64_buf: [0; 4],
      u64_idx: 4,
      f64_scalar_buf: [0.0; 8],
      f64_scalar_idx: 8,
      f32_scalar_buf: [0.0; 8],
      f32_scalar_idx: 8,
      i32_scalar_buf: [0; 8],
      i32_scalar_idx: 8,
    }
  }

  /// A fresh, independent stream from a globally unique seed (an atomic counter, thread-safe).
  #[inline]
  pub fn new() -> Self {
    Self::from_seed(next_global_seed())
  }

  #[inline(always)]
  pub(crate) fn next_i32x8(&mut self) -> i32x8 {
    let raw = self.f32_engine.next();
    unsafe { core::mem::transmute::<u32x8, i32x8>(raw) }
  }

  /// One uniform `f64` in `[0, 1)` from an 8-element buffer refilled by two `f64x4` stores, no
  /// stack round-trip, as every transcendental law calls it per draw.
  #[inline(always)]
  pub fn next_f64(&mut self) -> f64 {
    if self.f64_scalar_idx >= 8 {
      let buf_ptr = self.f64_scalar_buf.as_mut_ptr();
      unsafe {
        let bits0 = (self.f64_engine.next() >> 12u32) | F64_MAGIC_X4;
        let f0: f64x4 = core::mem::transmute::<u64x4, f64x4>(bits0);
        core::ptr::write_unaligned(buf_ptr as *mut f64x4, f0 - F64_ONE_X4);
        let bits1 = (self.f64_engine.next() >> 12u32) | F64_MAGIC_X4;
        let f1: f64x4 = core::mem::transmute::<u64x4, f64x4>(bits1);
        core::ptr::write_unaligned(buf_ptr.add(4) as *mut f64x4, f1 - F64_ONE_X4);
      }
      self.f64_scalar_idx = 0;
    }
    let v = self.f64_scalar_buf[self.f64_scalar_idx];
    self.f64_scalar_idx += 1;
    v
  }

  /// One uniform `f32` in `[0, 1)`, refilled by one `f32x8` store as in
  /// [`next_f64`](Self::next_f64).
  #[inline(always)]
  pub fn next_f32(&mut self) -> f32 {
    if self.f32_scalar_idx >= 8 {
      let buf_ptr = self.f32_scalar_buf.as_mut_ptr();
      unsafe {
        let bits = (self.f32_engine.next() >> 9u32) | F32_MAGIC_X8;
        let f: f32x8 = core::mem::transmute::<u32x8, f32x8>(bits);
        core::ptr::write_unaligned(buf_ptr as *mut f32x8, f - F32_ONE_X8);
      }
      self.f32_scalar_idx = 0;
    }
    let v = self.f32_scalar_buf[self.f32_scalar_idx];
    self.f32_scalar_idx += 1;
    v
  }

  /// A uniform `i32` over its whole range.
  #[inline(always)]
  pub fn next_i32(&mut self) -> i32 {
    if self.i32_scalar_idx >= 8 {
      self.i32_scalar_buf = self.next_i32x8().to_array();
      self.i32_scalar_idx = 0;
    }
    let v = self.i32_scalar_buf[self.i32_scalar_idx];
    self.i32_scalar_idx += 1;
    v
  }
}

impl Default for SimdRng {
  fn default() -> Self {
    Self::new()
  }
}

impl TryRng for SimdRng {
  type Error = Infallible;

  #[inline(always)]
  fn try_next_u32(&mut self) -> Result<u32, Infallible> {
    self.try_next_u64().map(|x| x as u32)
  }

  #[inline(always)]
  fn try_next_u64(&mut self) -> Result<u64, Infallible> {
    let idx = self.u64_idx;
    if idx >= 4 {
      self.u64_buf = self.f64_engine.next().to_array();
      self.u64_idx = 1;
      return Ok(self.u64_buf[0]);
    }
    self.u64_idx = idx + 1;
    Ok(self.u64_buf[idx])
  }

  /// The little-endian bytes of the words `next_u64` would return; a tail discards its word's unused bytes.
  fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), Infallible> {
    let (words, tail) = dest.as_chunks_mut::<8>();
    let (buffered, words) = words.split_at_mut(words.len().min(4 - self.u64_idx));
    for word in buffered {
      *word = self.u64_buf[self.u64_idx].to_le_bytes();
      self.u64_idx += 1;
    }
    // Past the buffered words, four words at a time are a fresh block, as four `next_u64` calls would draw.
    let (blocks, words) = words.as_chunks_mut::<4>();
    for block in blocks {
      for (word, w) in block.iter_mut().zip(self.f64_engine.next().to_array()) {
        *word = w.to_le_bytes();
      }
    }
    for word in words {
      *word = self.try_next_u64()?.to_le_bytes();
    }
    if !tail.is_empty() {
      let n = tail.len();
      tail.copy_from_slice(&self.try_next_u64()?.to_le_bytes()[..n]);
    }
    Ok(())
  }
}
