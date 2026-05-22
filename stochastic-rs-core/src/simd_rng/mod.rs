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

use std::sync::OnceLock;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;
use std::time::SystemTime;
use std::time::UNIX_EPOCH;

mod fill;
#[allow(clippy::module_inception)]
mod simd_rng;
#[cfg(test)]
mod tests;
mod xoshiro;

pub use simd_rng::SimdRng;
pub use xoshiro::Xoshiro128PP8;
pub use xoshiro::Xoshiro256PP4;
use xoshiro::splitmix64_mix;
use xoshiro::splitmix64_next;

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

/// Derives a child seed from a mutable parent seed.
#[inline]
pub fn derive_seed(state: &mut u64) -> u64 {
  splitmix64_next(state)
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

  /// Reset the internal seed state in place where meaningful.
  ///
  /// No-op for [`Unseeded`] — auto-seeded streams have no fixed point to
  /// reset to. For [`Deterministic`] this atomically replaces the current
  /// `state`, so a subsequent `rng()` / `rng_ext()` / `derive()` produces
  /// the stream rooted at the new `seed`. Lets a single
  /// `ProcessExt`-style instance replay or sweep different seeds without
  /// rebuilding the process — `fbm.seed.reseed(seed); fbm.sample();`.
  fn reseed(&self, _seed: u64) {}
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

  /// Atomically replace the internal `state` with `seed`. Subsequent
  /// stream-advancing calls (`rng`, `rng_ext`, `derive`) start from this
  /// seed, so the holder reproduces the same stream as
  /// `Deterministic::new(seed)` would.
  #[inline]
  pub fn reset(&self, seed: u64) {
    self.state.store(seed, Ordering::Relaxed);
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

  #[inline(always)]
  fn reseed(&self, seed: u64) {
    self.reset(seed);
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
pub trait SimdRngExt: rand::RngCore + Sized + Send + 'static {
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
