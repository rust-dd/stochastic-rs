//! # SIMD-accelerated random number generation
//!
//! Fast, high-quality pseudo-random number generator using SIMD parallelism.
//!
//! Two xoshiro engines run in parallel:
//! - **Xoshiro256++** (4×64-bit lanes) for `f64` and `u64` output
//! - **Xoshiro128++** (8×32-bit lanes) for `f32` and `i32` output
//!
//! Scalar methods buffer SIMD results to amortise lane-extraction cost.
//!
//! ## Seeding
//!
//! | constructor | behaviour |
//! |---|---|
//! | [`SimdRng::new()`] | globally-unique automatic seed (thread-safe atomic counter) |
//! | [`SimdRng::from_seed(seed)`] | deterministic – same `seed` ⇒ same stream |
//! | [`rng()`] | shorthand for `SimdRng::new()` |
//!
//! [`derive_seed`] splits a parent seed into child seeds for
//! propagating determinism across composed distributions and processes.
//!
//! $$
//! u_{k+1}=F(u_k),\quad x_k = \mathrm{transform}(u_k)
//! $$
//!
use std::sync::OnceLock;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;
use std::time::SystemTime;
use std::time::UNIX_EPOCH;

use rand::RngCore;
use wide::i32x8;
use wide::u32x8;
use wide::u64x4;

/// Rotate-left on 4 parallel u64 lanes.
#[inline(always)]
fn rotl_u64x4(x: u64x4, k: u32) -> u64x4 {
  (x << k) | (x >> (64 - k))
}

/// Rotate-left on 8 parallel u32 lanes.
#[inline(always)]
fn rotl_u32x8(x: u32x8, k: u32) -> u32x8 {
  (x << k) | (x >> (32 - k))
}

/// 4-lane parallel xoshiro256++ engine (64-bit output per lane).
struct Xoshiro256PP4 {
  s0: u64x4,
  s1: u64x4,
  s2: u64x4,
  s3: u64x4,
}

/// SplitMix64 bijective mixer. Advances `state` by a golden-ratio constant
/// and applies two rounds of xor-shift-multiply to produce a well-distributed
/// output. Used both for initial seeding and for [`derive_seed`].
#[inline(always)]
fn splitmix64_next(state: &mut u64) -> u64 {
  *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
  let mut z = *state;
  z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
  z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
  z ^ (z >> 31)
}

impl Xoshiro256PP4 {
  fn new_from_rng(rng: &mut impl RngCore) -> Self {
    let mut seed = [0u8; 128];
    rng.fill_bytes(&mut seed);
    let u = unsafe { core::mem::transmute::<[u8; 128], [u64; 16]>(seed) };
    Self {
      s0: u64x4::new([u[0], u[1], u[2], u[3]]),
      s1: u64x4::new([u[4], u[5], u[6], u[7]]),
      s2: u64x4::new([u[8], u[9], u[10], u[11]]),
      s3: u64x4::new([u[12], u[13], u[14], u[15]]),
    }
  }

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
  fn new_from_rng(rng: &mut impl RngCore) -> Self {
    let mut seed = [0u8; 128];
    rng.fill_bytes(&mut seed);
    let u = unsafe { core::mem::transmute::<[u8; 128], [u32; 32]>(seed) };
    Self {
      s0: u32x8::new([u[0], u[1], u[2], u[3], u[4], u[5], u[6], u[7]]),
      s1: u32x8::new([u[8], u[9], u[10], u[11], u[12], u[13], u[14], u[15]]),
      s2: u32x8::new([u[16], u[17], u[18], u[19], u[20], u[21], u[22], u[23]]),
      s3: u32x8::new([u[24], u[25], u[26], u[27], u[28], u[29], u[30], u[31]]),
    }
  }

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

/// Scale factor to convert a 53-bit integer to a `f64` in `[0, 1)`.
const F64_SCALE: f64 = 1.0 / (1u64 << 53) as f64;
/// Scale factor to convert a 24-bit integer to a `f32` in `[0, 1)`.
const F32_SCALE: f32 = 1.0 / (1u32 << 24) as f32;
/// Golden-ratio increment for the global seed counter.
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
  let t_hi = (t >> 64) as u64;
  let pid = std::process::id() as u64;
  let x = 0u64;
  let addr = (&x as *const u64 as usize) as u64;
  let mut seed = t_lo ^ t_hi.rotate_left(23) ^ pid.rotate_left(11) ^ addr.rotate_left(37);
  splitmix64_next(&mut seed)
}

/// Creates a new [`SimdRng`] with a globally-unique automatic seed.
///
/// Each call returns an independent stream. Thread-safe.
#[inline]
pub fn rng() -> SimdRng {
  SimdRng::new()
}

/// Creates a new [`SimdRng`] deterministically seeded from `seed`.
///
/// Two calls with the same `seed` produce identical streams.
#[inline]
pub fn rng_seeded(seed: u64) -> SimdRng {
  SimdRng::from_seed(seed)
}

/// Compile-time seed strategy for zero-overhead determinism control.
///
/// Two built-in implementations:
/// - [`Unseeded`] — fresh random RNG each time (default, zero cost)
/// - [`Deterministic`] — reproducible streams from a fixed seed
///
/// Each call to [`rng()`](Seed::rng) produces an independent [`SimdRng`].
/// [`derive()`](Seed::derive) creates a child seed for propagation
/// to sub-components (distributions, noise modules).
///
/// All branching is resolved at compile time via monomorphisation.
pub trait Seed: Copy + Clone + Send + Sync + 'static {
  /// Create an independent [`SimdRng`] and advance internal state.
  fn rng(&mut self) -> SimdRng;

  /// Derive a child seed for sub-component propagation.
  fn derive(&mut self) -> Self;
}

/// No seed — each RNG is independently random. Zero overhead.
#[derive(Copy, Clone, Debug, Default)]
pub struct Unseeded;

/// Deterministic seed — reproducible output from a fixed `u64`.
#[derive(Copy, Clone, Debug)]
pub struct Deterministic(pub u64);

impl Seed for Unseeded {
  #[inline(always)]
  fn rng(&mut self) -> SimdRng {
    SimdRng::new()
  }

  #[inline(always)]
  fn derive(&mut self) -> Self {
    Unseeded
  }
}

impl Seed for Deterministic {
  #[inline(always)]
  fn rng(&mut self) -> SimdRng {
    SimdRng::from_seed(splitmix64_next(&mut self.0))
  }

  #[inline(always)]
  fn derive(&mut self) -> Self {
    Deterministic(splitmix64_next(&mut self.0))
  }
}

/// Derives a child seed from a mutable parent seed.
///
/// Each call advances `state` and returns a new well-mixed value,
/// so successive calls yield independent child seeds:
///
/// ```
/// # use stochastic_rs::simd_rng::derive_seed;
/// let mut parent = 42u64;
/// let child_a = derive_seed(&mut parent);
/// let child_b = derive_seed(&mut parent);
/// assert_ne!(child_a, child_b);
/// ```
///
/// This is the recommended way to propagate determinism through
/// nested distributions and processes without sharing RNG state.
#[inline]
pub fn derive_seed(state: &mut u64) -> u64 {
  splitmix64_next(state)
}

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
  f64_engine: Xoshiro256PP4,
  f32_engine: Xoshiro128PP8,
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
    let raw = self.f32_engine.next().to_array();
    i32x8::new(unsafe { core::mem::transmute::<[u32; 8], [i32; 8]>(raw) })
  }

  /// Returns 8 uniform `f64` values in `[0, 1)`.
  ///
  /// Uses the 64-bit xoshiro256++ engine. Two SIMD iterations produce
  /// 4 raw `u64` values each; the top 53 bits are scaled to `[0, 1)`.
  #[inline(always)]
  pub fn next_f64_array(&mut self) -> [f64; 8] {
    let a = self.f64_engine.next().to_array();
    let b = self.f64_engine.next().to_array();
    [
      (a[0] >> 11) as f64 * F64_SCALE,
      (a[1] >> 11) as f64 * F64_SCALE,
      (a[2] >> 11) as f64 * F64_SCALE,
      (a[3] >> 11) as f64 * F64_SCALE,
      (b[0] >> 11) as f64 * F64_SCALE,
      (b[1] >> 11) as f64 * F64_SCALE,
      (b[2] >> 11) as f64 * F64_SCALE,
      (b[3] >> 11) as f64 * F64_SCALE,
    ]
  }

  /// Returns 8 uniform `f32` values in `[0, 1)`.
  ///
  /// Uses the 32-bit xoshiro128++ engine. One SIMD iteration produces
  /// 8 raw `u32` values; the top 24 bits are scaled to `[0, 1)`.
  #[inline(always)]
  pub fn next_f32_array(&mut self) -> [f32; 8] {
    let a = self.f32_engine.next().to_array();
    [
      (a[0] >> 8) as f32 * F32_SCALE,
      (a[1] >> 8) as f32 * F32_SCALE,
      (a[2] >> 8) as f32 * F32_SCALE,
      (a[3] >> 8) as f32 * F32_SCALE,
      (a[4] >> 8) as f32 * F32_SCALE,
      (a[5] >> 8) as f32 * F32_SCALE,
      (a[6] >> 8) as f32 * F32_SCALE,
      (a[7] >> 8) as f32 * F32_SCALE,
    ]
  }

  /// Returns a single uniform `f64` in `[0, 1)`.
  ///
  /// Draws from an internal 8-element buffer, refilling via
  /// [`next_f64_array`](Self::next_f64_array) when exhausted.
  #[inline(always)]
  pub fn next_f64(&mut self) -> f64 {
    if self.f64_scalar_idx >= 8 {
      self.f64_scalar_buf = self.next_f64_array();
      self.f64_scalar_idx = 0;
    }
    let v = self.f64_scalar_buf[self.f64_scalar_idx];
    self.f64_scalar_idx += 1;
    v
  }

  /// Returns a single uniform `f32` in `[0, 1)`.
  ///
  /// Draws from an internal 8-element buffer, refilling via
  /// [`next_f32_array`](Self::next_f32_array) when exhausted.
  #[inline(always)]
  pub fn next_f32(&mut self) -> f32 {
    if self.f32_scalar_idx >= 8 {
      self.f32_scalar_buf = self.next_f32_array();
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
    let mut chunks = dest.chunks_exact_mut(8);
    for chunk in &mut chunks {
      chunk.copy_from_slice(&self.next_u64().to_le_bytes());
    }
    let rem = chunks.into_remainder();
    if !rem.is_empty() {
      let bytes = self.next_u64().to_le_bytes();
      rem.copy_from_slice(&bytes[..rem.len()]);
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn f64_in_range() {
    let mut rng = SimdRng::new();
    for _ in 0..1000 {
      let vals = rng.next_f64_array();
      for v in vals {
        assert!((0.0..1.0).contains(&v), "f64 out of range: {v}");
      }
    }
  }

  #[test]
  fn f32_in_range() {
    let mut rng = SimdRng::new();
    for _ in 0..1000 {
      let vals = rng.next_f32_array();
      for v in vals {
        assert!((0.0..1.0).contains(&v), "f32 out of range: {v}");
      }
    }
  }

  #[test]
  fn rng_core_works() {
    let mut rng = SimdRng::new();
    let a = rng.next_u64();
    let b = rng.next_u64();
    assert_ne!(a, b);
  }
}
