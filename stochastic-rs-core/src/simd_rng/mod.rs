//! # SIMD-accelerated random number generation
//!
//! $$
//! u_{k+1}=F(u_k),\quad x_k = \mathrm{transform}(u_k)
//! $$
//!
//! Xoshiro256++ lanes give `f64` and `u64`, Xoshiro128++ `f32` and `i32`, scalar draws buffered;
//! [`SimdRng::from_seed`] replays, [`SimdRng::new()`] is unique, [`SeedExt`] carries either.

use std::fmt::Debug;
use std::mem::MaybeUninit;
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
pub(crate) mod xoshiro;

pub use simd_rng::SimdRng;
use xoshiro::splitmix64_mix;
use xoshiro::splitmix64_next;

pub(crate) mod engine_seal {
  pub trait Sealed {}
}

/// Not part of the public API: `stochastic-rs-stochastic` implements it for its Python seed source.
#[doc(hidden)]
pub mod seed_seal {
  #[diagnostic::on_unimplemented(
    message = "`{Self}` cannot implement the sealed trait `SeedExt`",
    note = "use `Unseeded` or `Deterministic`"
  )]
  pub trait Sealed {}
}

/// Golden-ratio increment for the global seed counter.
const SEED_GAMMA: u64 = 0x9e37_79b9_7f4a_7c15;

#[inline]
fn global_seed_counter() -> &'static AtomicU64 {
  static SEED_COUNTER: OnceLock<AtomicU64> = OnceLock::new();
  SEED_COUNTER.get_or_init(|| AtomicU64::new(initial_seed()))
}

/// Seeds a thread takes per global reservation, so the atomic stays off every construction while
/// ordinary workloads still cross the block boundary.
const SEED_BLOCK_LEN: u64 = 1 << 18;

std::thread_local! {
  /// `(next_base, remaining)` of this thread's reserved gamma-sequence block.
  static SEED_BLOCK: std::cell::Cell<(u64, u64)> = const { std::cell::Cell::new((0, 0)) };
}

/// Distinct seeds: the global counter strides `SEED_BLOCK_LEN·γ` and each thread `γ` in its
/// reservation, odd `γ` injective mod 2⁶⁴, splitmix64 bijective; the counter is hit once a block.
#[inline(always)]
fn next_global_seed() -> u64 {
  SEED_BLOCK.with(|cell| {
    let (base, left) = cell.get();
    if left == 0 {
      let block =
        global_seed_counter().fetch_add(SEED_BLOCK_LEN.wrapping_mul(SEED_GAMMA), Ordering::Relaxed);
      cell.set((block.wrapping_add(SEED_GAMMA), SEED_BLOCK_LEN - 1));
      let mut seed = block;
      return splitmix64_next(&mut seed);
    }
    cell.set((base.wrapping_add(SEED_GAMMA), left - 1));
    let mut seed = base;
    splitmix64_next(&mut seed)
  })
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

/// The next child seed of a mutable parent seed, which it advances.
#[doc(hidden)]
#[inline]
pub fn derive_seed(state: &mut u64) -> u64 {
  splitmix64_next(state)
}

/// Child seed of worker `stream_idx` from one fork basis; pure in both arguments.
#[doc(hidden)]
#[inline]
pub fn derive_fork_seed(parent_seed: u64, stream_idx: u64) -> u64 {
  splitmix64_mix(parent_seed ^ stream_idx)
}

/// Sealed seed source: [`Unseeded`] draws fresh entropy, [`Deterministic`] replays a fixed stream.
/// State advances through `&self`, so `ProcessExt::sample(&self)` can draw without an outer `&mut`.
pub trait SeedExt: seed_seal::Sealed + Clone + Send + Sync + 'static {
  /// Create an independent [`SimdRng`] and advance internal state.
  fn rng(&self) -> SimdRng;

  /// Derive a child seed for sub-component propagation, advancing internal state.
  #[doc(hidden)]
  fn derive(&self) -> Self;

  /// Create any [`SimdRngExt`] engine from this seed source, advancing the internal state;
  /// the engine-generic streams use it (e.g. `Seeded<D, R>` in `stochastic-rs-distributions`).
  fn rng_ext<R: SimdRngExt>(&self) -> R;

  /// Restarts a [`Deterministic`] stream at `seed`, so `fbm.seed().reseed(s); fbm.sample();`
  /// replays without a rebuild; a no-op for [`Unseeded`].
  fn reseed(&self, _seed: u64) {}

  /// Draws the next `u64` seed, advancing the source exactly as [`rng`](Self::rng) and
  /// [`rng_ext`](Self::rng_ext) do; two reads give two values.
  fn next_seed(&self) -> u64;
}

/// No seed — each RNG is independently random. Zero overhead.
#[derive(Copy, Clone, Debug, Default)]
pub struct Unseeded;

/// Reproducible seed from a fixed `u64`, its state an [`AtomicU64`] that `derive`/`rng` advance
/// through `&self`; a clone snapshots it.
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

  /// Replaces the state with `seed` atomically, so later `rng`, `rng_ext` and `derive` calls follow
  /// `Deterministic::new(seed)`'s stream.
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

impl seed_seal::Sealed for Unseeded {}

impl SeedExt for Unseeded {
  #[inline(always)]
  fn rng(&self) -> SimdRng {
    SimdRng::from_seed(self.next_seed())
  }

  #[inline(always)]
  fn derive(&self) -> Self {
    Unseeded
  }

  #[inline(always)]
  fn rng_ext<R: SimdRngExt>(&self) -> R {
    R::from_seed(self.next_seed())
  }

  #[inline(always)]
  fn next_seed(&self) -> u64 {
    // What `SimdRng::new()` and `SimdRngDual::new()` feed `from_seed`, so `rng` and `rng_ext` draw
    // as those constructors.
    next_global_seed()
  }
}

impl seed_seal::Sealed for Deterministic {}

impl SeedExt for Deterministic {
  #[inline(always)]
  fn rng(&self) -> SimdRng {
    SimdRng::from_seed(self.next_seed())
  }

  #[inline(always)]
  fn derive(&self) -> Self {
    Deterministic::new(self.next_seed())
  }

  #[inline(always)]
  fn rng_ext<R: SimdRngExt>(&self) -> R {
    R::from_seed(self.next_seed())
  }

  #[inline(always)]
  fn reseed(&self, seed: u64) {
    self.reset(seed);
  }

  #[inline(always)]
  fn next_seed(&self) -> u64 {
    self.next_u64()
  }
}

/// SIMD RNG engine the distributions are generic over, sealed to [`SimdRng`] and the
/// experimental `SimdRngDual`.
pub trait SimdRngExt:
  engine_seal::Sealed + rand::Rng + Clone + Debug + Sized + Send + Sync + 'static
{
  /// `true` when [`next_i32x8_pair`](Self::next_i32x8_pair) draws its two batches from independent
  /// engines, so a consumer unrolls to 16 lanes only then.
  #[doc(hidden)]
  const HAS_PAIR_ILP: bool = false;

  /// Globally-unique auto-seeded constructor.
  fn new() -> Self;

  /// Deterministic constructor from a single `u64` seed.
  fn from_seed(seed: u64) -> Self;

  /// Returns 8 i32 lanes from the 32-bit engine. Used in Ziggurat fast paths.
  #[doc(hidden)]
  fn next_i32x8(&mut self) -> wide::i32x8;

  /// Two `i32x8` batches: two calls on one engine by default, one batch per engine in the
  /// dual-stream impl, which lets the caller overlap the two table lookups.
  #[doc(hidden)]
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

  /// `fill_uniform_f64` into uninitialised storage; writes every element and reads none, so a
  /// caller may `assume_init`.
  #[doc(hidden)]
  fn fill_uniform_f64_uninit(&mut self, out: &mut [MaybeUninit<f64>]);

  /// `fill_uniform_f32` into uninitialised storage; writes every element and reads none, so a
  /// caller may `assume_init`.
  #[doc(hidden)]
  fn fill_uniform_f32_uninit(&mut self, out: &mut [MaybeUninit<f32>]);
}

/// The caller stores a value into every slot of the view and never stores `MaybeUninit::uninit()`.
#[doc(hidden)]
#[inline]
pub unsafe fn as_uninit_mut<T>(slice: &mut [T]) -> &mut [MaybeUninit<T>] {
  // SAFETY: `MaybeUninit<T>` has `T`'s layout, and the caller's contract keeps `slice` initialised.
  unsafe { &mut *(slice as *mut [T] as *mut [MaybeUninit<T>]) }
}

impl engine_seal::Sealed for SimdRng {}

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

  #[inline(always)]
  fn fill_uniform_f64_uninit(&mut self, out: &mut [MaybeUninit<f64>]) {
    self.fill_uniform_f64_uninit(out);
  }

  #[inline(always)]
  fn fill_uniform_f32_uninit(&mut self, out: &mut [MaybeUninit<f32>]) {
    self.fill_uniform_f32_uninit(out);
  }
}
