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
//!
//! Use the [`SeedExt`] trait ([`Unseeded`] / [`Deterministic`]) to propagate
//! determinism through composed distributions and processes at zero cost.
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
use wide::f32x8;
use wide::f64x4;
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
pub struct Xoshiro256PP4 {
  s0: u64x4,
  s1: u64x4,
  s2: u64x4,
  s3: u64x4,
}

/// SplitMix64 bijective mixer (post-increment stage).
///
/// Two rounds of xor-shift-multiply applied to a pre-incremented state.
/// Use this together with an external increment by `0x9e37_79b9_7f4a_7c15`
/// (golden-ratio odd constant) to mirror the canonical SplitMix64 stream.
#[inline(always)]
fn splitmix64_mix(state: u64) -> u64 {
  let mut z = state;
  z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
  z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
  z ^ (z >> 31)
}

/// SplitMix64 bijective mixer. Advances `state` by a golden-ratio constant
/// and applies the two-round mixer to produce a well-distributed output.
/// Used both for initial seeding and for [`derive_seed`].
#[inline(always)]
fn splitmix64_next(state: &mut u64) -> u64 {
  *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
  splitmix64_mix(*state)
}

impl Xoshiro256PP4 {
  pub fn new_from_rng(rng: &mut impl RngCore) -> Self {
    let mut seed = [0u8; 128];
    rng.fill_bytes(&mut seed);
    let mut u = [0u64; 16];
    for (i, lane) in u.iter_mut().enumerate() {
      let off = i * 8;
      *lane = u64::from_le_bytes(seed[off..off + 8].try_into().unwrap());
    }
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

pub struct Xoshiro128PP8 {
  s0: u32x8,
  s1: u32x8,
  s2: u32x8,
  s3: u32x8,
}

impl Xoshiro128PP8 {
  pub fn new_from_rng(rng: &mut impl RngCore) -> Self {
    let mut seed = [0u8; 128];
    rng.fill_bytes(&mut seed);
    let mut u = [0u32; 32];
    for (i, lane) in u.iter_mut().enumerate() {
      let off = i * 4;
      *lane = u32::from_le_bytes(seed[off..off + 4].try_into().unwrap());
    }
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

/// IEEE-754 bit pattern of `1.0_f64` (biased exponent 1023, mantissa 0).
/// OR-ing the high 52 bits of a `u64` into this constant gives a bit pattern
/// in `[1.0, 2.0)` whose mantissa is the upper-52-bit fraction of the input;
/// the subsequent subtract of `1.0` puts the value in `[0, 1)`.
const F64_MAGIC: u64 = 0x3FF0_0000_0000_0000;
/// IEEE-754 bit pattern of `1.0_f32` (biased exponent 127, mantissa 0). Same
/// trick as [`F64_MAGIC`] but with the upper 23 bits of a `u32`.
const F32_MAGIC: u32 = 0x3F80_0000;
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
  let pid = std::process::id() as u64;
  let x = 0u64;
  // Stack ASLR slide and text ASLR slide are independent on Linux/macOS/Windows,
  // so mixing both buys real entropy beyond the wall-clock + pid combination.
  let stack_addr = (&x as *const u64 as usize) as u64;
  let text_addr = (initial_seed as fn() -> u64 as usize) as u64;
  let mut seed =
    t_lo ^ pid.rotate_left(11) ^ stack_addr.rotate_left(37) ^ text_addr.rotate_left(53);
  splitmix64_next(&mut seed)
}

/// Creates a new [`SimdRng`] with a globally-unique automatic seed.
///
/// Each call returns an independent stream. Thread-safe.
#[inline]
pub fn rng() -> SimdRng {
  SimdRng::new()
}

/// Compile-time seed strategy for zero-overhead determinism control.
///
/// Two built-in implementations:
/// - [`Unseeded`] — fresh random RNG each time (default, zero cost)
/// - [`Deterministic`] — reproducible streams from a fixed seed
///
/// Each call to [`rng()`](SeedExt::rng) produces an independent [`SimdRng`]
/// **and advances** the seed's internal state, so successive calls produce
/// different streams. [`derive()`](SeedExt::derive) likewise advances state
/// and returns a child seed for propagation to sub-components.
///
/// State is stored with interior mutability (atomic for [`Deterministic`])
/// so methods take `&self` and remain callable from `&self` Process contexts
/// — e.g. `ProcessExt::sample(&self)` can advance the seed without an
/// outer `&mut`.
///
/// All branching is resolved at compile time via monomorphisation.
pub trait SeedExt: Clone + Send + Sync + 'static {
  /// Create an independent [`SimdRng`] and advance internal state.
  fn rng(&self) -> SimdRng;

  /// Derive a child seed for sub-component propagation, advancing internal state.
  #[doc(hidden)]
  fn derive(&self) -> Self;

  /// Create any [`SimdRngExt`] impl from this seed source, advancing the
  /// internal state. Used by generic distributions that are parametric over
  /// the underlying RNG type (e.g. `SimdNormal<T, N, R>`).
  fn rng_ext<R: SimdRngExt>(&self) -> R;
}

/// No seed — each RNG is independently random. Zero overhead.
#[derive(Copy, Clone, Debug, Default)]
pub struct Unseeded;

/// Deterministic seed — reproducible output from a fixed `u64`.
///
/// State is stored in an [`AtomicU64`] so `derive`/`rng` calls advance
/// state through `&self`. Cloning snapshots the current state.
#[derive(Debug)]
pub struct Deterministic {
  state: AtomicU64,
}

impl Deterministic {
  /// Construct from a raw seed value.
  #[inline]
  pub const fn new(seed: u64) -> Self {
    Self {
      state: AtomicU64::new(seed),
    }
  }

  /// Atomically advance the splitmix state and return the next mixed output.
  #[inline(always)]
  fn next_u64(&self) -> u64 {
    let new_state = self
      .state
      .fetch_add(0x9e37_79b9_7f4a_7c15, Ordering::Relaxed)
      .wrapping_add(0x9e37_79b9_7f4a_7c15);
    splitmix64_mix(new_state)
  }

  /// Snapshot the current internal state (primarily for debug / diagnostics).
  #[inline]
  pub fn current(&self) -> u64 {
    self.state.load(Ordering::Relaxed)
  }
}

impl Clone for Deterministic {
  fn clone(&self) -> Self {
    Self::new(self.current())
  }
}

impl SeedExt for Unseeded {
  #[inline(always)]
  fn rng(&self) -> SimdRng {
    SimdRng::new()
  }

  #[inline(always)]
  fn derive(&self) -> Self {
    Unseeded
  }

  #[inline(always)]
  fn rng_ext<R: SimdRngExt>(&self) -> R {
    R::new()
  }
}

impl SeedExt for Deterministic {
  #[inline(always)]
  fn rng(&self) -> SimdRng {
    SimdRng::from_seed(self.next_u64())
  }

  #[inline(always)]
  fn derive(&self) -> Self {
    Deterministic::new(self.next_u64())
  }

  #[inline(always)]
  fn rng_ext<R: SimdRngExt>(&self) -> R {
    R::from_seed(self.next_u64())
  }
}

/// Common interface for the SIMD RNG backends used by generic distributions.
///
/// `SimdNormal<T, N, R>` and friends are monomorphised against this trait so
/// the same struct definition serves both the single-stream [`SimdRng`] and
/// the experimental dual-stream `SimdRngDual` (gated behind the
/// `dual-stream-rng` feature). Implementations override
/// [`HAS_PAIR_ILP`](Self::HAS_PAIR_ILP) and [`next_i32x8_pair`](Self::next_i32x8_pair)
/// when they can usefully expose two independent batches per call —
/// consumers branch on the const to pick a 16-lane unrolled body, otherwise
/// they stay on the cheaper 8-lane body.
pub trait SimdRngExt: Sized + Send + 'static {
  /// `true` when [`next_i32x8_pair`](Self::next_i32x8_pair) returns two
  /// independent batches whose state updates can run in parallel. The
  /// single-stream impl leaves this at the default `false`; the dual-stream
  /// impl flips it to `true`. Consumers gate their loop unrolling on this
  /// const so single-stream codegen does not pay any unroll overhead.
  const HAS_PAIR_ILP: bool = false;

  /// Globally-unique auto-seeded constructor.
  fn new() -> Self;

  /// Deterministic constructor from a single `u64` seed.
  fn from_seed(seed: u64) -> Self;

  /// Returns 8 i32 lanes from the 32-bit engine. Used in Ziggurat fast paths.
  fn next_i32x8(&mut self) -> wide::i32x8;

  /// Returns two `i32x8` batches. Default impl is two back-to-back calls
  /// from the same engine — kept legal so any [`SimdRngExt`] can be
  /// consumed by a `pair`-shaped algorithm. Dual-stream impls override
  /// this to return batches from two **independent** engines so the
  /// surrounding code can hide table-lookup latency between them.
  #[inline(always)]
  fn next_i32x8_pair(&mut self) -> (wide::i32x8, wide::i32x8) {
    (self.next_i32x8(), self.next_i32x8())
  }

  /// Single random `i32`. Used by Ziggurat fallback paths.
  fn next_i32(&mut self) -> i32;

  /// Single uniform `f64` in `[0, 1)`. Used by Ziggurat tail / nfix paths.
  fn next_f64(&mut self) -> f64;

  /// Single uniform `f32` in `[0, 1)`.
  fn next_f32(&mut self) -> f32;

  /// Bulk-fill `out` with `U(0, 1)` `f64` values. Implementations should
  /// write through the slice directly without an intermediate `[f64; 8]`.
  fn fill_uniform_f64(&mut self, out: &mut [f64]);

  /// Bulk-fill `out` with `U(0, 1)` `f32` values.
  fn fill_uniform_f32(&mut self, out: &mut [f32]);
}

impl SimdRngExt for SimdRng {
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
    self.next_i32x8()
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
    self.next_f32()
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

/// Derives a child seed from a mutable parent seed.
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
