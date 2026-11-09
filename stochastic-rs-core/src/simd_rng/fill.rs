//! Bulk fills that store SIMD lanes straight into the caller's slice, with no `[f; 8]` returned by
//! value.

use std::mem::MaybeUninit;

use wide::f32x8;
use wide::f64x4;
use wide::u32x8;
use wide::u64x4;

use super::as_uninit_mut;
use super::simd_rng::SimdRng;
use super::xoshiro::F32_MAGIC_X8;
use super::xoshiro::F32_ONE_X8;
use super::xoshiro::F64_MAGIC_X4;
use super::xoshiro::F64_ONE_X4;

impl SimdRng {
  /// Uniform `f64` in `[0, 1)` by unaligned `f64x4` stores (`vmovupd`), a tail under 4 copied
  /// scalar; 52 bits by the magic-number trick, an ulp shy of the scalar 53, for a vector pipeline.
  #[inline]
  pub fn fill_uniform_f64(&mut self, out: &mut [f64]) {
    // SAFETY: `fill_uniform_f64_uninit` writes every element of `out`.
    self.fill_uniform_f64_uninit(unsafe { as_uninit_mut(out) });
  }

  #[inline]
  pub(crate) fn fill_uniform_f64_uninit(&mut self, out: &mut [MaybeUninit<f64>]) {
    let len = out.len();
    let full_chunks = len / 4;
    let ptr = out.as_mut_ptr().cast::<f64>();

    for i in 0..full_chunks {
      let u = self.f64_engine.next();
      let bits = (u >> 12u32) | F64_MAGIC_X4;
      // SAFETY: `u64x4` and `f64x4` are both 256 bits, and every bit pattern is an `f64`.
      let f: f64x4 = unsafe { core::mem::transmute::<u64x4, f64x4>(bits) };
      let result = f - F64_ONE_X4;
      // SAFETY: lanes `4i..4i + 4` lie inside `out`, and `write_unaligned` needs no alignment.
      unsafe {
        core::ptr::write_unaligned(ptr.add(i * 4) as *mut f64x4, result);
      }
    }

    let tail = len - full_chunks * 4;
    if tail > 0 {
      let u = self.f64_engine.next();
      let bits = (u >> 12u32) | F64_MAGIC_X4;
      // SAFETY: `u64x4` and `f64x4` are both 256 bits, and every bit pattern is an `f64`.
      let f: f64x4 = unsafe { core::mem::transmute::<u64x4, f64x4>(bits) };
      let arr: [f64; 4] = (f - F64_ONE_X4).to_array();
      out[full_chunks * 4..].copy_from_slice(&arr.map(MaybeUninit::new)[..tail]);
    }
  }

  /// Uniform `f32` in `[0, 1)` by unaligned `f32x8` stores, a tail under 8 copied scalar; 23 bits
  /// by the magic-number trick, with no integer-to-float conversion.
  #[inline]
  pub fn fill_uniform_f32(&mut self, out: &mut [f32]) {
    // SAFETY: `fill_uniform_f32_uninit` writes every element of `out`.
    self.fill_uniform_f32_uninit(unsafe { as_uninit_mut(out) });
  }

  #[inline]
  pub(crate) fn fill_uniform_f32_uninit(&mut self, out: &mut [MaybeUninit<f32>]) {
    let len = out.len();
    let full_chunks = len / 8;
    let ptr = out.as_mut_ptr().cast::<f32>();

    for i in 0..full_chunks {
      let u = self.f32_engine.next();
      let bits = (u >> 9u32) | F32_MAGIC_X8;
      // SAFETY: `u32x8` and `f32x8` are both 256 bits, and every bit pattern is an `f32`.
      let f: f32x8 = unsafe { core::mem::transmute::<u32x8, f32x8>(bits) };
      let result = f - F32_ONE_X8;
      // SAFETY: lanes `8i..8i + 8` lie inside `out`, and `write_unaligned` needs no alignment.
      unsafe {
        core::ptr::write_unaligned(ptr.add(i * 8) as *mut f32x8, result);
      }
    }

    let tail = len - full_chunks * 8;
    if tail > 0 {
      let u = self.f32_engine.next();
      let bits = (u >> 9u32) | F32_MAGIC_X8;
      // SAFETY: `u32x8` and `f32x8` are both 256 bits, and every bit pattern is an `f32`.
      let f: f32x8 = unsafe { core::mem::transmute::<u32x8, f32x8>(bits) };
      let arr: [f32; 8] = (f - F32_ONE_X8).to_array();
      out[full_chunks * 8..].copy_from_slice(&arr.map(MaybeUninit::new)[..tail]);
    }
  }
}
