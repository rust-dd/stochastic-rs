//! Bulk-fill helpers — `fill_uniform_f64` / `fill_uniform_f32` write SIMD
//! stores straight into a caller-supplied slice, skipping the `[f; 8]`
//! return-by-value round-trip that the `next_*_array` accessors pay.

use wide::f32x8;
use wide::f64x4;
use wide::u32x8;
use wide::u64x4;

use super::simd_rng::SimdRng;
use super::xoshiro::F32_MAGIC;
use super::xoshiro::F64_MAGIC;

impl SimdRng {
  /// Fills `out` with uniform `f64` values in `[0, 1)` using direct SIMD
  /// stores. Avoids the `[f64; 8]` return-by-value round-trip that
  /// [`next_f64_array`](Self::next_f64_array) pays — every 4-lane chunk is
  /// written via an unaligned `f64x4` store (`vmovupd`), and any tail
  /// shorter than 4 falls back to a scalar copy.
  ///
  /// Same 52-bit precision as [`next_f64_array`](Self::next_f64_array).
  #[inline]
  pub fn fill_uniform_f64(&mut self, out: &mut [f64]) {
    let magic = u64x4::splat(F64_MAGIC);
    let one = f64x4::splat(1.0);
    let len = out.len();
    let full_chunks = len / 4;
    let ptr = out.as_mut_ptr();

    for i in 0..full_chunks {
      let u = self.f64_engine.next();
      let bits = (u >> 12u32) | magic;
      let f: f64x4 = unsafe { core::mem::transmute::<u64x4, f64x4>(bits) };
      let result = f - one;
      unsafe {
        core::ptr::write_unaligned(ptr.add(i * 4) as *mut f64x4, result);
      }
    }

    let tail = len - full_chunks * 4;
    if tail > 0 {
      let u = self.f64_engine.next();
      let bits = (u >> 12u32) | magic;
      let f: f64x4 = unsafe { core::mem::transmute::<u64x4, f64x4>(bits) };
      let arr: [f64; 4] = (f - one).to_array();
      let dst = unsafe { core::slice::from_raw_parts_mut(ptr.add(full_chunks * 4), tail) };
      dst.copy_from_slice(&arr[..tail]);
    }
  }

  /// Fills `out` with uniform `f32` values in `[0, 1)`. Each 8-lane chunk
  /// is written via an unaligned `f32x8` store; tails shorter than 8 fall
  /// back to a scalar copy. 23-bit precision (same as
  /// [`next_f32_array`](Self::next_f32_array)).
  #[inline]
  pub fn fill_uniform_f32(&mut self, out: &mut [f32]) {
    let magic = u32x8::splat(F32_MAGIC);
    let one = f32x8::splat(1.0);
    let len = out.len();
    let full_chunks = len / 8;
    let ptr = out.as_mut_ptr();

    for i in 0..full_chunks {
      let u = self.f32_engine.next();
      let bits = (u >> 9u32) | magic;
      let f: f32x8 = unsafe { core::mem::transmute::<u32x8, f32x8>(bits) };
      let result = f - one;
      unsafe {
        core::ptr::write_unaligned(ptr.add(i * 8) as *mut f32x8, result);
      }
    }

    let tail = len - full_chunks * 8;
    if tail > 0 {
      let u = self.f32_engine.next();
      let bits = (u >> 9u32) | magic;
      let f: f32x8 = unsafe { core::mem::transmute::<u32x8, f32x8>(bits) };
      let arr: [f32; 8] = (f - one).to_array();
      let dst = unsafe { core::slice::from_raw_parts_mut(ptr.add(full_chunks * 8), tail) };
      dst.copy_from_slice(&arr[..tail]);
    }
  }
}
