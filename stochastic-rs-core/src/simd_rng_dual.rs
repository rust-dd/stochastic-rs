//! # Dual-stream SIMD RNG (experimental)
//!
//! Two `Xoshiro256++` / `Xoshiro128++` engines run side-by-side and the
//! public API exposes a `_pair` flavour that returns one batch from each.
//! Modern out-of-order CPUs can overlap the two engines' state updates with
//! the consumer's compute / memory ops, exposing more ILP than a
//! single-stream design where the state update is a hard serial dependency.
//!
//! This is a **non-production experiment** parallel to
//! [`crate::simd_rng::SimdRng`] — same algorithms, doubled state, different
//! deterministic output sequence under [`SimdRngDual::from_seed`]. Promotion
//! to the production `SimdRng` is gated on a measurable bulk-fill speedup.
//!
//! # Layout
//!
//! ```text
//!   f64_a ────────────► consumer (chunk i)
//!   f64_b ────────────► consumer (chunk i+1)   ↘
//!                                                interleaved by OoO core
//!   f32_a ────────────► consumer (lanes 0..8)  ↗
//!   f32_b ────────────► consumer (lanes 8..16)
//! ```

use std::sync::OnceLock;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;
use std::time::SystemTime;
use std::time::UNIX_EPOCH;

use rand::RngCore;
use wide::f32x8;
use wide::f64x4;
use wide::i32x8;
use wide::u32x8;
use wide::u64x4;

#[inline(always)]
fn rotl_u64x4(x: u64x4, k: u32) -> u64x4 {
  (x << k) | (x >> (64 - k))
}

#[inline(always)]
fn rotl_u32x8(x: u32x8, k: u32) -> u32x8 {
  (x << k) | (x >> (32 - k))
}

/// 4-lane parallel xoshiro256++ engine (identical to the single-stream variant).
struct Xoshiro256PP4 {
  s0: u64x4,
  s1: u64x4,
  s2: u64x4,
  s3: u64x4,
}

/// SplitMix64 step used to seed both engines from a single user-provided `u64`.
#[inline(always)]
fn splitmix64_mix(state: u64) -> u64 {
  let mut z = state;
  z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
  z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
  z ^ (z >> 31)
}

#[inline(always)]
fn splitmix64_next(state: &mut u64) -> u64 {
  *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
  splitmix64_mix(*state)
}

impl Xoshiro256PP4 {
  fn new_from_u64(seed: u64) -> Self {
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
  fn next(&mut self) -> u64x4 {
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

struct Xoshiro128PP8 {
  s0: u32x8,
  s1: u32x8,
  s2: u32x8,
  s3: u32x8,
}

impl Xoshiro128PP8 {
  fn new_from_u64(seed: u64) -> Self {
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
  fn next(&mut self) -> u32x8 {
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

/// IEEE-754 bit pattern of `1.0_f64`.
const F64_MAGIC: u64 = 0x3FF0_0000_0000_0000;
/// IEEE-754 bit pattern of `1.0_f32`.
const F32_MAGIC: u32 = 0x3F80_0000;
const SEED_GAMMA: u64 = 0x9e37_79b9_7f4a_7c15;

#[inline]
fn global_seed_counter() -> &'static AtomicU64 {
  static SEED_COUNTER: OnceLock<AtomicU64> = OnceLock::new();
  SEED_COUNTER.get_or_init(|| AtomicU64::new(initial_seed()))
}

#[inline(always)]
fn next_global_seed() -> u64 {
  let base = global_seed_counter().fetch_add(SEED_GAMMA, Ordering::Relaxed);
  let mut seed = base;
  splitmix64_next(&mut seed)
}

#[inline]
fn initial_seed() -> u64 {
  let t = SystemTime::now()
    .duration_since(UNIX_EPOCH)
    .map(|d| d.as_nanos())
    .unwrap_or(0);
  let t_lo = t as u64;
  let pid = std::process::id() as u64;
  let x = 0u64;
  let stack_addr = (&x as *const u64 as usize) as u64;
  let text_addr = (initial_seed as fn() -> u64 as usize) as u64;
  let mut seed =
    t_lo ^ pid.rotate_left(11) ^ stack_addr.rotate_left(37) ^ text_addr.rotate_left(53);
  splitmix64_next(&mut seed)
}

/// Experimental dual-stream SIMD RNG.
///
/// Carries two of each underlying engine; the public methods that target
/// bulk consumers expose both streams so the OoO core can pipeline state
/// updates against the consumer's compute / memory ops.
pub struct SimdRngDual {
  f64_a: Xoshiro256PP4,
  f64_b: Xoshiro256PP4,
  f32_a: Xoshiro128PP8,
  f32_b: Xoshiro128PP8,
  f64_scalar_buf: [f64; 8],
  f64_scalar_idx: usize,
  i32_scalar_buf: [i32; 8],
  i32_scalar_idx: usize,
}

impl SimdRngDual {
  /// Seed each of the four sub-engines from independent SplitMix64 segments.
  #[inline]
  pub fn from_seed(seed: u64) -> Self {
    let mut state = seed;
    let s_a64 = splitmix64_next(&mut state);
    let s_b64 = splitmix64_next(&mut state);
    let s_a32 = splitmix64_next(&mut state);
    let s_b32 = splitmix64_next(&mut state);
    Self {
      f64_a: Xoshiro256PP4::new_from_u64(s_a64),
      f64_b: Xoshiro256PP4::new_from_u64(s_b64),
      f32_a: Xoshiro128PP8::new_from_u64(s_a32),
      f32_b: Xoshiro128PP8::new_from_u64(s_b32),
      f64_scalar_buf: [0.0; 8],
      f64_scalar_idx: 8,
      i32_scalar_buf: [0; 8],
      i32_scalar_idx: 8,
    }
  }

  /// Globally-unique automatic seed (thread-safe atomic counter).
  #[inline]
  pub fn new() -> Self {
    Self::from_seed(next_global_seed())
  }

  /// Returns two independent `i32x8` batches — one from engine A, one from
  /// engine B. Consumers should process them as a 16-lane unrolled body so
  /// the compiler can interleave the two `xoshiro` state updates with the
  /// inner-loop compute / memory ops.
  #[inline(always)]
  pub fn next_i32x8_pair(&mut self) -> (i32x8, i32x8) {
    let raw_a = self.f32_a.next();
    let raw_b = self.f32_b.next();
    unsafe {
      (
        core::mem::transmute::<u32x8, i32x8>(raw_a),
        core::mem::transmute::<u32x8, i32x8>(raw_b),
      )
    }
  }

  /// Fills `out` with uniform `f64` values in `[0, 1)` using **alternating**
  /// engines per 4-lane chunk. The two engines have independent state, so
  /// every other chunk's xoshiro update can overlap with the previous
  /// chunk's store on a modern OoO core.
  ///
  /// 52-bit precision via the magic-number trick (same as
  /// [`crate::simd_rng::SimdRng::fill_uniform_f64`]).
  #[inline]
  pub fn fill_uniform_f64(&mut self, out: &mut [f64]) {
    let magic = u64x4::splat(F64_MAGIC);
    let one = f64x4::splat(1.0);
    let len = out.len();
    let ptr = out.as_mut_ptr();

    let pair_chunks = len / 8;
    for i in 0..pair_chunks {
      let bits_a = (self.f64_a.next() >> 12u32) | magic;
      let bits_b = (self.f64_b.next() >> 12u32) | magic;
      let fa: f64x4 = unsafe { core::mem::transmute::<u64x4, f64x4>(bits_a) };
      let fb: f64x4 = unsafe { core::mem::transmute::<u64x4, f64x4>(bits_b) };
      let ra = fa - one;
      let rb = fb - one;
      unsafe {
        core::ptr::write_unaligned(ptr.add(i * 8) as *mut f64x4, ra);
        core::ptr::write_unaligned(ptr.add(i * 8 + 4) as *mut f64x4, rb);
      }
    }

    let mut written = pair_chunks * 8;
    while written + 4 <= len {
      let bits = (self.f64_a.next() >> 12u32) | magic;
      let f: f64x4 = unsafe { core::mem::transmute::<u64x4, f64x4>(bits) };
      unsafe {
        core::ptr::write_unaligned(ptr.add(written) as *mut f64x4, f - one);
      }
      written += 4;
    }
    if written < len {
      let bits = (self.f64_a.next() >> 12u32) | magic;
      let f: f64x4 = unsafe { core::mem::transmute::<u64x4, f64x4>(bits) };
      let arr: [f64; 4] = (f - one).to_array();
      let tail = unsafe { core::slice::from_raw_parts_mut(ptr.add(written), len - written) };
      tail.copy_from_slice(&arr[..len - written]);
    }
  }

  /// Fills `out` with uniform `f32` values in `[0, 1)`. Each 8-lane f32x8
  /// chunk alternates between engines A and B; tails shorter than 8 fall
  /// back to a scalar copy.
  #[inline]
  pub fn fill_uniform_f32(&mut self, out: &mut [f32]) {
    let magic = u32x8::splat(F32_MAGIC);
    let one = f32x8::splat(1.0);
    let len = out.len();
    let ptr = out.as_mut_ptr();

    let pair_chunks = len / 16;
    for i in 0..pair_chunks {
      let bits_a = (self.f32_a.next() >> 9u32) | magic;
      let bits_b = (self.f32_b.next() >> 9u32) | magic;
      let fa: f32x8 = unsafe { core::mem::transmute::<u32x8, f32x8>(bits_a) };
      let fb: f32x8 = unsafe { core::mem::transmute::<u32x8, f32x8>(bits_b) };
      let ra = fa - one;
      let rb = fb - one;
      unsafe {
        core::ptr::write_unaligned(ptr.add(i * 16) as *mut f32x8, ra);
        core::ptr::write_unaligned(ptr.add(i * 16 + 8) as *mut f32x8, rb);
      }
    }

    let mut written = pair_chunks * 16;
    while written + 8 <= len {
      let bits = (self.f32_a.next() >> 9u32) | magic;
      let f: f32x8 = unsafe { core::mem::transmute::<u32x8, f32x8>(bits) };
      unsafe {
        core::ptr::write_unaligned(ptr.add(written) as *mut f32x8, f - one);
      }
      written += 8;
    }
    if written < len {
      let bits = (self.f32_a.next() >> 9u32) | magic;
      let f: f32x8 = unsafe { core::mem::transmute::<u32x8, f32x8>(bits) };
      let arr: [f32; 8] = (f - one).to_array();
      let tail = unsafe { core::slice::from_raw_parts_mut(ptr.add(written), len - written) };
      tail.copy_from_slice(&arr[..len - written]);
    }
  }

  /// Returns a single uniform `f64` in `[0, 1)`. Refills its 8-element buffer
  /// from one engine A batch and one engine B batch in alternation —
  /// each refill exposes both state updates to the OoO core simultaneously.
  #[inline(always)]
  pub fn next_f64(&mut self) -> f64 {
    if self.f64_scalar_idx >= 8 {
      let magic = u64x4::splat(F64_MAGIC);
      let one = f64x4::splat(1.0);
      let buf_ptr = self.f64_scalar_buf.as_mut_ptr();
      unsafe {
        let bits_a = (self.f64_a.next() >> 12u32) | magic;
        let bits_b = (self.f64_b.next() >> 12u32) | magic;
        let fa: f64x4 = core::mem::transmute::<u64x4, f64x4>(bits_a);
        let fb: f64x4 = core::mem::transmute::<u64x4, f64x4>(bits_b);
        core::ptr::write_unaligned(buf_ptr as *mut f64x4, fa - one);
        core::ptr::write_unaligned(buf_ptr.add(4) as *mut f64x4, fb - one);
      }
      self.f64_scalar_idx = 0;
    }
    let v = self.f64_scalar_buf[self.f64_scalar_idx];
    self.f64_scalar_idx += 1;
    v
  }

  /// Returns a single random `i32`. Refills its 8-element buffer via one
  /// `next_i32x8_pair` call (writes both A and B's first batch — only
  /// engine A's lanes are used here, B's lanes feed the next 8 draws).
  #[inline(always)]
  pub fn next_i32(&mut self) -> i32 {
    if self.i32_scalar_idx >= 8 {
      let raw = self.f32_a.next();
      self.i32_scalar_buf = unsafe { core::mem::transmute::<u32x8, [i32; 8]>(raw) };
      self.i32_scalar_idx = 0;
    }
    let v = self.i32_scalar_buf[self.i32_scalar_idx];
    self.i32_scalar_idx += 1;
    v
  }
}

impl Default for SimdRngDual {
  fn default() -> Self {
    Self::new()
  }
}

impl crate::simd_rng::SimdRngExt for SimdRngDual {
  /// Two independent xoshiro engines means
  /// [`next_i32x8_pair`](Self::next_i32x8_pair) genuinely returns two
  /// state-independent batches — consumers should branch on this const to
  /// pick the unrolled 16-lane body.
  const HAS_PAIR_ILP: bool = true;

  #[inline(always)]
  fn new() -> Self {
    Self::new()
  }

  #[inline(always)]
  fn from_seed(seed: u64) -> Self {
    Self::from_seed(seed)
  }

  #[inline(always)]
  fn next_i32x8(&mut self) -> wide::i32x8 {
    // Single-batch consumers just see engine A's stream.
    let raw = self.f32_a.next();
    unsafe { core::mem::transmute::<u32x8, wide::i32x8>(raw) }
  }

  #[inline(always)]
  fn next_i32x8_pair(&mut self) -> (wide::i32x8, wide::i32x8) {
    self.next_i32x8_pair()
  }

  #[inline(always)]
  fn next_i32(&mut self) -> i32 {
    self.next_i32()
  }

  #[inline(always)]
  fn next_f64(&mut self) -> f64 {
    self.next_f64()
  }

  #[inline(always)]
  fn next_f32(&mut self) -> f32 {
    // Single-sample f32 path: pull from engine A, magic-number to [0, 1),
    // discard B's lanes (the dual-stream payoff lives in the bulk fills).
    let bits = (self.f32_a.next() >> 9u32) | u32x8::splat(F32_MAGIC);
    let f: f32x8 = unsafe { core::mem::transmute::<u32x8, f32x8>(bits) };
    let arr: [f32; 8] = (f - f32x8::splat(1.0)).to_array();
    arr[0]
  }

  #[inline(always)]
  fn fill_uniform_f64(&mut self, out: &mut [f64]) {
    self.fill_uniform_f64(out);
  }

  #[inline(always)]
  fn fill_uniform_f32(&mut self, out: &mut [f32]) {
    self.fill_uniform_f32(out);
  }
}

impl RngCore for SimdRngDual {
  #[inline(always)]
  fn next_u32(&mut self) -> u32 {
    self.next_u64() as u32
  }

  #[inline(always)]
  fn next_u64(&mut self) -> u64 {
    // Pull one u64 from engine A; ignore B for the scalar path.
    let arr = self.f64_a.next().to_array();
    arr[0]
  }

  fn fill_bytes(&mut self, dest: &mut [u8]) {
    let mut written = 0;
    let total = dest.len();
    while total - written >= 64 {
      let a = self.f64_a.next().to_array();
      let b = self.f64_b.next().to_array();
      for (i, lane) in a.iter().enumerate() {
        let off = written + i * 8;
        dest[off..off + 8].copy_from_slice(&lane.to_le_bytes());
      }
      for (i, lane) in b.iter().enumerate() {
        let off = written + 32 + i * 8;
        dest[off..off + 8].copy_from_slice(&lane.to_le_bytes());
      }
      written += 64;
    }
    while total - written >= 8 {
      let v = self.next_u64();
      dest[written..written + 8].copy_from_slice(&v.to_le_bytes());
      written += 8;
    }
    if written < total {
      let v = self.next_u64();
      let bytes = v.to_le_bytes();
      let take = total - written;
      dest[written..written + take].copy_from_slice(&bytes[..take]);
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn f64_in_range() {
    let mut rng = SimdRngDual::new();
    let mut buf = [0.0_f64; 1024];
    rng.fill_uniform_f64(&mut buf);
    for v in buf.iter() {
      assert!((0.0..1.0).contains(v), "out of range: {v}");
    }
  }

  #[test]
  fn f32_in_range() {
    let mut rng = SimdRngDual::new();
    let mut buf = [0.0_f32; 1024];
    rng.fill_uniform_f32(&mut buf);
    for v in buf.iter() {
      assert!((0.0..1.0).contains(v), "out of range: {v}");
    }
  }

  #[test]
  fn deterministic_streams_match() {
    let mut a = SimdRngDual::from_seed(123);
    let mut b = SimdRngDual::from_seed(123);
    for _ in 0..100 {
      assert_eq!(a.next_f64(), b.next_f64());
    }
  }

  #[test]
  fn engines_a_and_b_are_distinct() {
    // A single from_seed must NOT seed engine A and B identically.
    let mut rng = SimdRngDual::from_seed(0xdeadbeef);
    let (a, b) = rng.next_i32x8_pair();
    assert_ne!(a.to_array(), b.to_array());
  }
}
