//! Parallel xoshiro engines and the SplitMix64 mixer used to seed them.
//!
//! Two engines:
//! - [`Xoshiro256PP4`] — 4×64-bit lanes for `f64` / `u64` output
//! - [`Xoshiro128PP8`] — 8×32-bit lanes for `f32` / `i32` output
//!
//! Both are seeded via SplitMix64 expansion of a single `u64`, which gives
//! enough decorrelated state for the full 4×u64 / 8×u32 lane vectors.

use wide::u32x8;
use wide::u64x4;

/// Rotate-left on 4 parallel u64 lanes.
#[inline(always)]
pub(super) fn rotl_u64x4(x: u64x4, k: u32) -> u64x4 {
  (x << k) | (x >> (64 - k))
}

/// Rotate-left on 8 parallel u32 lanes.
#[inline(always)]
pub(super) fn rotl_u32x8(x: u32x8, k: u32) -> u32x8 {
  (x << k) | (x >> (32 - k))
}

/// SplitMix64 bijective mixer (post-increment stage).
///
/// Two rounds of xor-shift-multiply applied to a pre-incremented state.
/// Use this together with an external increment by `0x9e37_79b9_7f4a_7c15`
/// (golden-ratio odd constant) to mirror the canonical SplitMix64 stream.
#[inline(always)]
pub(super) fn splitmix64_mix(state: u64) -> u64 {
  let mut z = state;
  z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
  z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
  z ^ (z >> 31)
}

/// SplitMix64 bijective mixer. Advances `state` by a golden-ratio constant
/// and applies the two-round mixer to produce a well-distributed output.
/// Used both for initial seeding and for [`derive_seed`](super::derive_seed).
#[inline(always)]
pub(super) fn splitmix64_next(state: &mut u64) -> u64 {
  *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
  splitmix64_mix(*state)
}

/// IEEE-754 bit pattern of `1.0_f64` (biased exponent 1023, mantissa 0).
/// OR-ing the high 52 bits of a `u64` into this constant gives a bit pattern
/// in `[1.0, 2.0)` whose mantissa is the upper-52-bit fraction of the input;
/// the subsequent subtract of `1.0` puts the value in `[0, 1)`.
pub(super) const F64_MAGIC: u64 = 0x3FF0_0000_0000_0000;
/// IEEE-754 bit pattern of `1.0_f32` (biased exponent 127, mantissa 0). Same
/// trick as [`F64_MAGIC`] but with the upper 23 bits of a `u32`.
pub(super) const F32_MAGIC: u32 = 0x3F80_0000;

/// 4-lane parallel xoshiro256++ engine (64-bit output per lane).
pub struct Xoshiro256PP4 {
  s0: u64x4,
  s1: u64x4,
  s2: u64x4,
  s3: u64x4,
}

impl Xoshiro256PP4 {
  pub(super) fn new_from_u64(seed: u64) -> Self {
    let mut state = seed;
    let mut u = [0u64; 16];
    for x in &mut u {
      *x = splitmix64_next(&mut state);
    }
    Self {
      s0: u64x4::new([u[0], u[1], u[2], u[3]]),
      s1: u64x4::new([u[4], u[5], u[6], u[7]]),
      s2: u64x4::new([u[8], u[9], u[10], u[11]]),
      s3: u64x4::new([u[12], u[13], u[14], u[15]]),
    }
  }

  #[inline(always)]
  pub(super) fn next(&mut self) -> u64x4 {
    let result = rotl_u64x4(self.s0 + self.s3, 23) + self.s0;
    let t = self.s1 << 17u32;
    self.s2 ^= self.s0;
    self.s3 ^= self.s1;
    self.s1 ^= self.s2;
    self.s0 ^= self.s3;
    self.s2 ^= t;
    self.s3 = rotl_u64x4(self.s3, 45);
    result
  }
}

/// 8-lane parallel xoshiro128++ engine (32-bit output per lane).
pub struct Xoshiro128PP8 {
  s0: u32x8,
  s1: u32x8,
  s2: u32x8,
  s3: u32x8,
}

impl Xoshiro128PP8 {
  pub(super) fn new_from_u64(seed: u64) -> Self {
    let mut state = seed;
    let mut u = [0u32; 32];
    for i in (0..32).step_by(2) {
      let x = splitmix64_next(&mut state);
      u[i] = x as u32;
      u[i + 1] = (x >> 32) as u32;
    }
    Self {
      s0: u32x8::new([u[0], u[1], u[2], u[3], u[4], u[5], u[6], u[7]]),
      s1: u32x8::new([u[8], u[9], u[10], u[11], u[12], u[13], u[14], u[15]]),
      s2: u32x8::new([u[16], u[17], u[18], u[19], u[20], u[21], u[22], u[23]]),
      s3: u32x8::new([u[24], u[25], u[26], u[27], u[28], u[29], u[30], u[31]]),
    }
  }

  #[inline(always)]
  pub(super) fn next(&mut self) -> u32x8 {
    let result = rotl_u32x8(self.s0 + self.s3, 7) + self.s0;
    let t = self.s1 << 9u32;
    self.s2 ^= self.s0;
    self.s3 ^= self.s1;
    self.s1 ^= self.s2;
    self.s0 ^= self.s3;
    self.s2 ^= t;
    self.s3 = rotl_u32x8(self.s3, 11);
    result
  }
}
