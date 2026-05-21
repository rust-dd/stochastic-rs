//! Single-stream [`SimdRng`] struct, construction, scalar / batch sampling,
//! and the [`rand::RngCore`] implementation. Bulk fill helpers live in
//! [`super::fill`].

use rand::RngCore;
use wide::f32x8;
use wide::f64x4;
use wide::i32x8;
use wide::u32x8;
use wide::u64x4;

use super::next_global_seed;
use super::xoshiro::F32_MAGIC;
use super::xoshiro::F64_MAGIC;
use super::xoshiro::Xoshiro128PP8;
use super::xoshiro::Xoshiro256PP4;
use super::xoshiro::splitmix64_next;

/// SIMD-accelerated pseudo-random number generator.
///
/// Internally maintains two xoshiro engines (64-bit and 32-bit) and
/// multiple scalar buffers for amortising SIMD-to-scalar extraction.
///
/// # Construction
///
/// - [`SimdRng::new()`] — globally-unique automatic seed
/// - [`SimdRng::from_seed(seed)`] — deterministic, reproducible stream
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
  /// Creates a deterministically-seeded RNG.
  ///
  /// The `seed` is expanded via SplitMix64 into the full internal state
  /// of both xoshiro engines. Two instances created with the same seed
  /// will produce identical output sequences.
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

  /// Creates an RNG with a globally-unique automatic seed.
  ///
  /// Every call returns a fresh, independent stream. Thread-safe via an
  /// internal atomic counter.
  #[inline]
  pub fn new() -> Self {
    Self::from_seed(next_global_seed())
  }

  /// Returns 8 random `i32` values as a SIMD vector.
  ///
  /// Raw bit pattern from the 32-bit engine, reinterpreted as signed.
  #[inline(always)]
  pub fn next_i32x8(&mut self) -> i32x8 {
    let raw = self.f32_engine.next();
    unsafe { core::mem::transmute::<u32x8, i32x8>(raw) }
  }

  /// Returns 8 uniform `f64` values in `[0, 1)`.
  ///
  /// Two SIMD iterations of the 64-bit xoshiro256++ engine give 4×u64 each.
  /// The top 52 bits of each lane are OR-ed into the bit pattern of `1.0`,
  /// producing a vector in `[1.0, 2.0)` reinterpretable as `f64x4`; the
  /// subsequent SIMD subtract of `1.0` lands the result in `[0, 1)`.
  /// 52-bit precision (1 ULP shy of the 53-bit scalar variant) in exchange
  /// for a fully vectorised pipeline.
  ///
  /// For bulk fills prefer [`fill_uniform_f64`](Self::fill_uniform_f64),
  /// which writes f64x4 stores directly into the caller's slice and avoids
  /// the `[f64; 8]` return-by-value stack round-trip.
  #[inline(always)]
  pub fn next_f64_array(&mut self) -> [f64; 8] {
    let a = self.f64_engine.next();
    let b = self.f64_engine.next();
    let magic = u64x4::splat(F64_MAGIC);
    let one = f64x4::splat(1.0);
    let bits_a = (a >> 12u32) | magic;
    let bits_b = (b >> 12u32) | magic;
    let fa: f64x4 = unsafe { core::mem::transmute::<u64x4, f64x4>(bits_a) };
    let fb: f64x4 = unsafe { core::mem::transmute::<u64x4, f64x4>(bits_b) };
    let ra = (fa - one).to_array();
    let rb = (fb - one).to_array();
    [ra[0], ra[1], ra[2], ra[3], rb[0], rb[1], rb[2], rb[3]]
  }

  /// Returns 8 uniform `f32` values in `[0, 1)`.
  ///
  /// One SIMD iteration of the 32-bit xoshiro128++ engine gives 8×u32. The
  /// top 23 bits of each lane are OR-ed into the bit pattern of `1.0_f32`,
  /// producing a vector in `[1.0, 2.0)` reinterpretable as `f32x8`; the
  /// subsequent SIMD subtract of `1.0` lands the result in `[0, 1)`.
  /// 23-bit precision in exchange for zero integer-to-float conversion cost.
  ///
  /// For bulk fills prefer [`fill_uniform_f32`](Self::fill_uniform_f32).
  #[inline(always)]
  pub fn next_f32_array(&mut self) -> [f32; 8] {
    let a = self.f32_engine.next();
    let bits = (a >> 9u32) | u32x8::splat(F32_MAGIC);
    let f: f32x8 = unsafe { core::mem::transmute::<u32x8, f32x8>(bits) };
    (f - f32x8::splat(1.0)).to_array()
  }

  /// Returns a single uniform `f64` in `[0, 1)`.
  ///
  /// Draws from an internal 8-element buffer. Refills via two unaligned
  /// `f64x4` stores (magic-number trick) directly into the buffer — avoids
  /// the `[f64; 8]` return-by-value round-trip that
  /// [`next_f64_array`](Self::next_f64_array) pays when the caller copies
  /// the result. This matters because every transcendental-heavy distribution
  /// (Gamma, Beta, NIG, …) hits `next_f64` repeatedly.
  #[inline(always)]
  pub fn next_f64(&mut self) -> f64 {
    if self.f64_scalar_idx >= 8 {
      let magic = u64x4::splat(F64_MAGIC);
      let one = f64x4::splat(1.0);
      let buf_ptr = self.f64_scalar_buf.as_mut_ptr();
      unsafe {
        let bits0 = (self.f64_engine.next() >> 12u32) | magic;
        let f0: f64x4 = core::mem::transmute::<u64x4, f64x4>(bits0);
        core::ptr::write_unaligned(buf_ptr as *mut f64x4, f0 - one);
        let bits1 = (self.f64_engine.next() >> 12u32) | magic;
        let f1: f64x4 = core::mem::transmute::<u64x4, f64x4>(bits1);
        core::ptr::write_unaligned(buf_ptr.add(4) as *mut f64x4, f1 - one);
      }
      self.f64_scalar_idx = 0;
    }
    let v = self.f64_scalar_buf[self.f64_scalar_idx];
    self.f64_scalar_idx += 1;
    v
  }

  /// Returns a single uniform `f32` in `[0, 1)`.
  ///
  /// Same direct-refill strategy as [`next_f64`](Self::next_f64): one
  /// unaligned `f32x8` store via the magic-number trick.
  #[inline(always)]
  pub fn next_f32(&mut self) -> f32 {
    if self.f32_scalar_idx >= 8 {
      let buf_ptr = self.f32_scalar_buf.as_mut_ptr();
      unsafe {
        let bits = (self.f32_engine.next() >> 9u32) | u32x8::splat(F32_MAGIC);
        let f: f32x8 = core::mem::transmute::<u32x8, f32x8>(bits);
        core::ptr::write_unaligned(buf_ptr as *mut f32x8, f - f32x8::splat(1.0));
      }
      self.f32_scalar_idx = 0;
    }
    let v = self.f32_scalar_buf[self.f32_scalar_idx];
    self.f32_scalar_idx += 1;
    v
  }

  /// Returns a single random `i32`.
  ///
  /// Draws from an internal 8-element buffer, refilling via
  /// [`next_i32x8`](Self::next_i32x8) when exhausted.
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

impl RngCore for SimdRng {
  #[inline(always)]
  fn next_u32(&mut self) -> u32 {
    self.next_u64() as u32
  }

  #[inline(always)]
  fn next_u64(&mut self) -> u64 {
    let idx = self.u64_idx;
    if idx >= 4 {
      self.u64_buf = self.f64_engine.next().to_array();
      self.u64_idx = 1;
      return self.u64_buf[0];
    }
    self.u64_idx = idx + 1;
    self.u64_buf[idx]
  }

  fn fill_bytes(&mut self, dest: &mut [u8]) {
    let mut written = 0;
    let total = dest.len();
    while self.u64_idx < 4 && total - written >= 8 {
      let v = self.u64_buf[self.u64_idx];
      self.u64_idx += 1;
      dest[written..written + 8].copy_from_slice(&v.to_le_bytes());
      written += 8;
    }
    while total - written >= 32 {
      let block = self.f64_engine.next().to_array();
      dest[written..written + 8].copy_from_slice(&block[0].to_le_bytes());
      dest[written + 8..written + 16].copy_from_slice(&block[1].to_le_bytes());
      dest[written + 16..written + 24].copy_from_slice(&block[2].to_le_bytes());
      dest[written + 24..written + 32].copy_from_slice(&block[3].to_le_bytes());
      written += 32;
    }
    if written == total {
      return;
    }
    self.u64_buf = self.f64_engine.next().to_array();
    self.u64_idx = 0;
    while total - written >= 8 {
      let v = self.u64_buf[self.u64_idx];
      self.u64_idx += 1;
      dest[written..written + 8].copy_from_slice(&v.to_le_bytes());
      written += 8;
    }
    if written < total {
      let bytes = self.u64_buf[self.u64_idx].to_le_bytes();
      let take = total - written;
      dest[written..written + take].copy_from_slice(&bytes[..take]);
      self.u64_idx += 1;
    }
  }
}
