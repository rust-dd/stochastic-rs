//! # Exp
//!
//! $$
//! f(x)=\lambda e^{-\lambda x},\ x\ge 0
//! $$
//!
use std::cell::Cell;
use std::cell::UnsafeCell;
use std::sync::OnceLock;

use rand::Rng;
use rand_distr::Distribution;
use stochastic_rs_core::simd_rng::Unseeded;
use wide::i32x8;

use super::SimdFloatExt;
use crate::simd_rng::SeedExt;
use crate::simd_rng::SimdRng;
use crate::simd_rng::SimdRngExt;

const ZIG_EXP_R: f64 = 7.697_117_470_131_487;
const ZIG_EXP_V: f64 = 3.949_659_822_581_572e-3;
const TABLE_SIZE: usize = 256;
const SMALL_EXP_THRESHOLD: usize = 16;

/// Precomputed lookup tables for the Ziggurat algorithm (exponential distribution).
/// `ke` holds threshold integers for the fast-accept test,
/// `we` holds the width of each rectangle,
/// `fe` holds the function values f(x)=exp(-x) at rectangle boundaries.
struct ExpZigTables {
  ke: [i32; TABLE_SIZE],
  we: [f64; TABLE_SIZE],
  fe: [f64; TABLE_SIZE],
}

static EXP_ZIG_TABLES: OnceLock<ExpZigTables> = OnceLock::new();

/// Returns a reference to the lazily-initialized exponential Ziggurat tables.
fn exp_zig_tables() -> &'static ExpZigTables {
  EXP_ZIG_TABLES.get_or_init(|| {
    let mut ke = [0i32; TABLE_SIZE];
    let mut we = [0.0f64; TABLE_SIZE];
    let mut fe = [0.0f64; TABLE_SIZE];

    let m2 = (1u64 << 31) as f64;

    let mut de = ZIG_EXP_R;
    let mut te = de;
    let q = ZIG_EXP_V / (-de).exp();

    let ke0 = (de / q) * m2;
    ke[0] = if ke0 > i32::MAX as f64 {
      i32::MAX
    } else {
      ke0 as i32
    };
    ke[1] = 0;

    we[0] = q / m2;
    we[TABLE_SIZE - 1] = de / m2;

    fe[0] = 1.0;
    fe[TABLE_SIZE - 1] = (-de).exp();

    for i in (2..TABLE_SIZE).rev() {
      de = -(ZIG_EXP_V / de + (-de).exp()).ln();
      let ke_val = (de / te) * m2;
      ke[i] = if ke_val > i32::MAX as f64 {
        i32::MAX
      } else {
        ke_val as i32
      };
      te = de;
      we[i - 1] = de / m2;
      fe[i - 1] = (-de).exp();
    }

    ExpZigTables { ke, we, fe }
  })
}

/// Scalar fallback for exponential samples that fall outside a Ziggurat rectangle.
/// For iz==0 (tail), uses the inversion method: R - ln(U).
/// Otherwise performs rejection sampling within the rectangle.
#[cold]
#[inline(never)]
fn efix<T: SimdFloatExt, R: SimdRngExt>(
  hz: i32,
  iz: usize,
  tables: &ExpZigTables,
  rng: &mut R,
) -> T {
  let mut hz = hz;
  let mut iz = iz;

  loop {
    if iz == 0 {
      return T::from_f64_fast(ZIG_EXP_R - (1.0f64 - rng.next_f64()).ln());
    }

    let x = (hz.unsigned_abs() as f64) * tables.we[iz];
    if tables.fe[iz] + rng.next_f64() * (tables.fe[iz - 1] - tables.fe[iz]) < (-x).exp() {
      return T::from_f64_fast(x);
    }

    hz = rng.next_i32();
    iz = (hz & 0xFF) as usize;
    let abs_hz = hz.unsigned_abs() as i64;
    if abs_hz < tables.ke[iz] as i64 {
      return T::from_f64_fast((abs_hz as f64) * tables.we[iz]);
    }
  }
}

/// SIMD-accelerated exponential distribution using the Ziggurat algorithm.
/// Generates Exp(1) samples internally, then scales by 1/lambda.
///
/// The const generic `N` controls the internal buffer size. The third
/// generic `R: SimdRngExt` picks the backing RNG: default
/// [`SimdRng`] (single-stream); the experimental
/// `SimdRngDual` (dual-stream) is reachable via the
/// `SimdExpZigDual` type alias when the
/// `dual-stream-rng` feature is enabled.
pub struct SimdExpZig<T: SimdFloatExt, const N: usize = 64, R: SimdRngExt = SimdRng> {
  lambda: T,
  buffer: UnsafeCell<[T; N]>,
  index: UnsafeCell<usize>,
  simd_rng: UnsafeCell<R>,
  pub(crate) stream_seed: Cell<u64>,
}

impl<T: SimdFloatExt, const N: usize, R: SimdRngExt> SimdExpZig<T, N, R> {
  /// Creates an exponential distribution with rate `lambda` and the given
  /// seed strategy (`Unseeded` for auto-seeded, `Deterministic::new(...)`
  /// for a reproducible stream). Single canonical constructor following
  /// the `Gbm::new(..., seed: S)` pattern used across the workspace.
  #[inline]
  pub fn new<S: SeedExt>(lambda: T, seed: &S) -> Self {
    let _ = exp_zig_tables();
    assert!(lambda > T::zero());
    assert!(N >= 8, "buffer size must be at least 8");
    let stream_seed = seed.seed_value();
    Self {
      lambda,
      buffer: UnsafeCell::new([T::zero(); N]),
      index: UnsafeCell::new(N),
      simd_rng: UnsafeCell::new(R::from_seed(stream_seed)),
      stream_seed: Cell::new(stream_seed),
    }
  }

  /// Builds an independent worker stream for the `stream_idx`-th chunk of a
  /// parallel `sample_matrix` fan-out; see
  /// [`DistributionSampler::fork`](crate::traits::DistributionSampler::fork).
  #[doc(hidden)]
  pub fn fork(&self, stream_idx: u64) -> Self {
    let mut basis = self.stream_seed.get();
    let call_basis = crate::simd_rng::derive_seed(&mut basis);
    self.stream_seed.set(basis);
    let child_seed = crate::simd_rng::derive_fork_seed(call_basis, stream_idx);
    Self::new(
      self.lambda,
      &crate::simd_rng::Deterministic::new(child_seed),
    )
  }

  /// Generates a single Exp(1) sample using the scalar Ziggurat path.
  #[inline]
  fn sample_exp1_one(rng: &mut R, tables: &ExpZigTables) -> T {
    let hz = rng.next_i32();
    let iz = (hz & 0xFF) as usize;
    let abs_hz = hz.unsigned_abs() as i64;
    if abs_hz < tables.ke[iz] as i64 {
      T::from_f64_fast((abs_hz as f64) * tables.we[iz])
    } else {
      efix::<T, R>(hz, iz, tables, rng)
    }
  }

  /// Processes one 8-lane exponential Ziggurat batch from `hz` into
  /// `out[..8]`, scaled by `factor` (= 1/λ).
  #[inline(always)]
  fn exp_batch8(
    hz: i32x8,
    tables: &ExpZigTables,
    factor: T,
    factor_simd: T::Simd,
    rng: &mut R,
    out: &mut [T],
  ) {
    let iz = hz & i32x8::splat(0xFF);
    let iz_arr = iz.to_array();
    let abs_hz = hz.abs();
    unsafe {
      let ke_vals = i32x8::new([
        *tables.ke.get_unchecked(iz_arr[0] as usize),
        *tables.ke.get_unchecked(iz_arr[1] as usize),
        *tables.ke.get_unchecked(iz_arr[2] as usize),
        *tables.ke.get_unchecked(iz_arr[3] as usize),
        *tables.ke.get_unchecked(iz_arr[4] as usize),
        *tables.ke.get_unchecked(iz_arr[5] as usize),
        *tables.ke.get_unchecked(iz_arr[6] as usize),
        *tables.ke.get_unchecked(iz_arr[7] as usize),
      ]);

      let accept = abs_hz.simd_lt(ke_vals);

      let we_arr: [T; 8] = [
        T::from_f64_fast(*tables.we.get_unchecked(iz_arr[0] as usize)),
        T::from_f64_fast(*tables.we.get_unchecked(iz_arr[1] as usize)),
        T::from_f64_fast(*tables.we.get_unchecked(iz_arr[2] as usize)),
        T::from_f64_fast(*tables.we.get_unchecked(iz_arr[3] as usize)),
        T::from_f64_fast(*tables.we.get_unchecked(iz_arr[4] as usize)),
        T::from_f64_fast(*tables.we.get_unchecked(iz_arr[5] as usize)),
        T::from_f64_fast(*tables.we.get_unchecked(iz_arr[6] as usize)),
        T::from_f64_fast(*tables.we.get_unchecked(iz_arr[7] as usize)),
      ];
      let hz_float = T::simd_from_i32x8(abs_hz);
      let result = hz_float * T::simd_from_array(we_arr) * factor_simd;

      if accept.all() {
        out[..8].copy_from_slice(&T::simd_to_array(result));
      } else {
        let hz_arr = hz.to_array();
        let accept_arr = accept.to_array();
        let result_arr = T::simd_to_array(result);
        for i in 0..8 {
          out[i] = if accept_arr[i] != 0 {
            result_arr[i]
          } else {
            efix::<T, R>(hz_arr[i], iz_arr[i] as usize, tables, rng) * factor
          };
        }
      }
    }
  }

  /// Core Ziggurat fill for Exp(λ) samples into `buf`. The internal Ziggurat
  /// path generates Exp(1) values which are then multiplied by `factor`
  /// (= 1/λ) at the SIMD store step, fusing the previous two-pass
  /// `fill_exp1` + `scale_in_place` pipeline into a single pass.
  ///
  /// Uses 8-wide SIMD for the fast-accept path; scalar fallback for edge
  /// cases multiplies by `factor` lane-by-lane. When `R` reports
  /// [`SimdRngExt::HAS_PAIR_ILP`] the main loop consumes
  /// [`next_i32x8_pair`](SimdRngExt::next_i32x8_pair) and processes 16
  /// lanes per iteration (const-folded away for single-stream engines).
  #[inline]
  fn fill_exp_scaled(buf: &mut [T], rng: &mut R, factor: T) {
    let tables = exp_zig_tables();
    let len = buf.len();
    if len < SMALL_EXP_THRESHOLD {
      for x in buf.iter_mut() {
        *x = Self::sample_exp1_one(rng, tables) * factor;
      }
      return;
    }
    let factor_simd = T::splat(factor);
    let mut filled = 0;

    if R::HAS_PAIR_ILP {
      while filled + 16 <= len {
        let (hz_a, hz_b) = rng.next_i32x8_pair();
        Self::exp_batch8(
          hz_a,
          tables,
          factor,
          factor_simd,
          rng,
          &mut buf[filled..filled + 8],
        );
        Self::exp_batch8(
          hz_b,
          tables,
          factor,
          factor_simd,
          rng,
          &mut buf[filled + 8..filled + 16],
        );
        filled += 16;
      }
    }
    while filled + 8 <= len {
      let hz = rng.next_i32x8();
      Self::exp_batch8(
        hz,
        tables,
        factor,
        factor_simd,
        rng,
        &mut buf[filled..filled + 8],
      );
      filled += 8;
    }
    while filled < len {
      buf[filled] = Self::sample_exp1_one(rng, tables) * factor;
      filled += 1;
    }
  }

  /// Returns a single Exp(lambda) sample using the internal SIMD RNG.
  #[inline]
  pub fn sample_fast(&self) -> T {
    let index = unsafe { &mut *self.index.get() };
    if *index >= N {
      self.refill_buffer();
    }
    let val = unsafe { (*self.buffer.get())[*index] };
    *index += 1;
    val
  }

  /// Fills a slice with Exp(λ) samples using the internal SIMD RNG stream —
  /// the only stream this sampler draws from (see the crate-level RNG
  /// policy: the seed goes to the constructor, not to sampling calls).
  #[inline]
  pub fn fill_slice(&self, out: &mut [T]) {
    let rng = unsafe { &mut *self.simd_rng.get() };
    Self::fill_exp_scaled(out, rng, T::one() / self.lambda);
  }

  /// Refills the internal sample buffer with Exp(λ) values.
  fn refill_buffer(&self) {
    let rng = unsafe { &mut *self.simd_rng.get() };
    let buf = unsafe { &mut *self.buffer.get() };
    Self::fill_exp_scaled(buf, rng, T::one() / self.lambda);
    unsafe {
      *self.index.get() = 0;
    }
  }
}

impl<T: SimdFloatExt, const N: usize, R: SimdRngExt> Clone for SimdExpZig<T, N, R> {
  fn clone(&self) -> Self {
    Self::new(self.lambda, &Unseeded)
  }
}

impl<T: SimdFloatExt, const N: usize, R: SimdRngExt> Distribution<T> for SimdExpZig<T, N, R> {
  /// Returns a single Exp(lambda) sample.
  /// Draws from a pre-filled buffer, refilling it when exhausted.
  ///
  /// The `rng` argument is intentionally unused — this type draws from its
  /// own internal SIMD stream seeded at construction. Use `Deterministic`
  /// in the constructor for reproducibility.
  #[inline(always)]
  fn sample<Rr: Rng + ?Sized>(&self, _rng: &mut Rr) -> T {
    let index = unsafe { &mut *self.index.get() };
    if *index >= N {
      self.refill_buffer();
    }
    let val = unsafe { (*self.buffer.get())[*index] };
    *index += 1;
    val
  }
}

impl<T: SimdFloatExt, const N: usize, R: SimdRngExt> crate::traits::DistributionExt
  for SimdExpZig<T, N, R>
{
  fn pdf(&self, x: f64) -> f64 {
    let lambda = self.lambda.to_f64().unwrap();
    if x < 0.0 {
      0.0
    } else {
      lambda * (-lambda * x).exp()
    }
  }

  fn cdf(&self, x: f64) -> f64 {
    let lambda = self.lambda.to_f64().unwrap();
    if x < 0.0 {
      0.0
    } else {
      1.0 - (-lambda * x).exp()
    }
  }

  fn inv_cdf(&self, p: f64) -> f64 {
    let lambda = self.lambda.to_f64().unwrap();
    -(1.0 - p).ln() / lambda
  }

  fn mean(&self) -> f64 {
    1.0 / self.lambda.to_f64().unwrap()
  }

  fn median(&self) -> f64 {
    std::f64::consts::LN_2 / self.lambda.to_f64().unwrap()
  }

  fn mode(&self) -> f64 {
    0.0
  }

  fn variance(&self) -> f64 {
    let l = self.lambda.to_f64().unwrap();
    1.0 / (l * l)
  }

  fn skewness(&self) -> f64 {
    2.0
  }

  fn kurtosis(&self) -> f64 {
    6.0
  }

  fn entropy(&self) -> f64 {
    1.0 - self.lambda.to_f64().unwrap().ln()
  }

  fn characteristic_function(&self, t: f64) -> num_complex::Complex64 {
    // φ(t) = λ / (λ - it)
    let lambda = self.lambda.to_f64().unwrap();
    let denom = num_complex::Complex64::new(lambda, -t);
    num_complex::Complex64::new(lambda, 0.0) / denom
  }

  fn moment_generating_function(&self, t: f64) -> f64 {
    let lambda = self.lambda.to_f64().unwrap();
    if t < lambda {
      lambda / (lambda - t)
    } else {
      f64::INFINITY
    }
  }
}

/// Convenience wrapper around [`SimdExpZig`] with a default buffer size.
/// Provides the same API with less generic noise. Inherits the `R` backing
/// RNG parameter from [`SimdExpZig`] so the dual-stream alias
/// `SimdExpDual` is just `SimdExp<T, SimdRngDual>`.
pub struct SimdExp<T: SimdFloatExt, R: SimdRngExt = SimdRng> {
  inner: SimdExpZig<T, 64, R>,
}

impl<T: SimdFloatExt, R: SimdRngExt> SimdExp<T, R> {
  /// Creates an exponential distribution with rate `lambda` and the given
  /// seed strategy. Single canonical constructor — pass
  /// [`Unseeded`](crate::simd_rng::Unseeded) for auto or
  /// [`Deterministic::new(seed)`](crate::simd_rng::Deterministic) for a
  /// reproducible stream.
  #[inline]
  pub fn new<S: SeedExt>(lambda: T, seed: &S) -> Self {
    Self {
      inner: SimdExpZig::new(lambda, seed),
    }
  }

  /// Builds an independent worker stream for the `stream_idx`-th chunk of a
  /// parallel `sample_matrix` fan-out; see
  /// [`DistributionSampler::fork`](crate::traits::DistributionSampler::fork).
  #[doc(hidden)]
  pub fn fork(&self, stream_idx: u64) -> Self {
    let mut basis = self.inner.stream_seed.get();
    let call_basis = crate::simd_rng::derive_seed(&mut basis);
    self.inner.stream_seed.set(basis);
    let child_seed = crate::simd_rng::derive_fork_seed(call_basis, stream_idx);
    Self::new(
      self.inner.lambda,
      &crate::simd_rng::Deterministic::new(child_seed),
    )
  }

  /// Returns a single Exp(lambda) sample using the internal SIMD RNG.
  #[inline]
  pub fn sample_fast(&self) -> T {
    self.inner.sample_fast()
  }

  /// Fills a slice with Exp(lambda) samples using the internal SIMD RNG
  /// stream — the only stream this sampler draws from (see the
  /// crate-level RNG policy). Delegates to the inner `SimdExpZig`.
  pub fn fill_slice(&self, out: &mut [T]) {
    self.inner.fill_slice(out);
  }
}

/// Exp(1) — the standard exponential; matches this file's own module doc
/// ("Generates Exp(1) samples internally, then scales by 1/lambda").
impl<T: SimdFloatExt, R: SimdRngExt> Default for SimdExp<T, R> {
  fn default() -> Self {
    Self::new(T::one(), &Unseeded)
  }
}

impl<T: SimdFloatExt, R: SimdRngExt> Clone for SimdExp<T, R> {
  fn clone(&self) -> Self {
    Self::new(self.inner.lambda, &Unseeded)
  }
}

impl<T: SimdFloatExt, R: SimdRngExt> Distribution<T> for SimdExp<T, R> {
  /// Returns a single Exp(lambda) sample. Delegates to the inner `SimdExpZig`.
  ///
  /// The `rng` argument is intentionally unused — this type draws from its
  /// own internal SIMD stream seeded at construction. Use `Deterministic`
  /// in the constructor for reproducibility.
  #[inline(always)]
  fn sample<Rr: Rng + ?Sized>(&self, rng: &mut Rr) -> T {
    self.inner.sample(rng)
  }
}

impl<T: SimdFloatExt, R: SimdRngExt> crate::traits::DistributionExt for SimdExp<T, R> {
  fn pdf(&self, x: f64) -> f64 {
    self.inner.pdf(x)
  }
  fn cdf(&self, x: f64) -> f64 {
    self.inner.cdf(x)
  }
  fn inv_cdf(&self, p: f64) -> f64 {
    self.inner.inv_cdf(p)
  }
  fn mean(&self) -> f64 {
    self.inner.mean()
  }
  fn median(&self) -> f64 {
    self.inner.median()
  }
  fn mode(&self) -> f64 {
    self.inner.mode()
  }
  fn variance(&self) -> f64 {
    self.inner.variance()
  }
  fn skewness(&self) -> f64 {
    self.inner.skewness()
  }
  fn kurtosis(&self) -> f64 {
    self.inner.kurtosis()
  }
  fn entropy(&self) -> f64 {
    self.inner.entropy()
  }
  fn characteristic_function(&self, t: f64) -> num_complex::Complex64 {
    self.inner.characteristic_function(t)
  }
  fn moment_generating_function(&self, t: f64) -> f64 {
    self.inner.moment_generating_function(t)
  }
}

py_distribution!(PyExp, SimdExp,
  sig: (lambda_, seed=None, dtype=None),
  params: (lambda_: f64)
);

#[cfg(test)]
mod tests;
