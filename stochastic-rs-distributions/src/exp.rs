//! # Exp
//!
//! $$
//! f(x)=\lambda e^{-\lambda x},\ x\ge 0
//! $$
//!
//! Sampling: Marsaglia, G., Tsang, W.W. (2000), "The Ziggurat Method for Generating Random Variables", *Journal of Statistical Software* 5(8), DOI 10.18637/jss.v005.i08.

use std::sync::OnceLock;

use rand::Rng;
use rand::distr::Distribution;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::SimdRngExt;
use wide::i32x8;

use crate::seeded::Buffered;
use crate::seeded::StreamState;
use crate::source::AnyRng;
use crate::source::Source;
use crate::traits::SimdFloatExt;
use crate::traits::distribution::Sealed;
use crate::traits::distribution::SimdDistribution;
use crate::traits::distribution::SimdKernel;

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
fn efix<T: SimdFloatExt, S: Source + ?Sized>(
  hz: i32,
  iz: usize,
  tables: &ExpZigTables,
  rng: &mut S,
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

/// One Exp(1) draw on the scalar Ziggurat path, from a SIMD engine or the caller's rng.
#[inline]
fn sample_exp1_one<T: SimdFloatExt, S: Source + ?Sized>(rng: &mut S, tables: &ExpZigTables) -> T {
  let hz = rng.next_i32();
  let iz = (hz & 0xFF) as usize;
  let abs_hz = hz.unsigned_abs() as i64;
  if abs_hz < tables.ke[iz] as i64 {
    T::from_f64_fast((abs_hz as f64) * tables.we[iz])
  } else {
    efix::<T, S>(hz, iz, tables, rng)
  }
}

/// One 8-lane exponential Ziggurat batch from `hz` into `out[..8]`, scaled by `factor` (= 1/λ).
#[inline(always)]
fn exp_batch8<T: SimdFloatExt, R: SimdRngExt>(
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

/// Exp(1) Ziggurat fill scaled by `factor` (= 1/λ) at the store; kept out of line because inlined into a
/// pop loop's refill this kernel slows the loop.
#[inline(never)]
fn fill_exp_scaled<T: SimdFloatExt, R: SimdRngExt>(buf: &mut [T], rng: &mut R, factor: T) {
  let tables = exp_zig_tables();
  let len = buf.len();
  if len < SMALL_EXP_THRESHOLD {
    for x in buf.iter_mut() {
      *x = sample_exp1_one::<T, R>(rng, tables) * factor;
    }
    return;
  }
  let factor_simd = T::splat(factor);
  let mut filled = 0;

  if R::HAS_PAIR_ILP {
    while filled + 16 <= len {
      let (hz_a, hz_b) = rng.next_i32x8_pair();
      exp_batch8::<T, R>(
        hz_a,
        tables,
        factor,
        factor_simd,
        rng,
        &mut buf[filled..filled + 8],
      );
      exp_batch8::<T, R>(
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
    exp_batch8::<T, R>(
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
    buf[filled] = sample_exp1_one::<T, R>(rng, tables) * factor;
    filled += 1;
  }
}

/// Exponential law `Exp(lambda)`: parameters only; a [`Seeded`](crate::Seeded) stream draws it in bulk.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SimdExp<T> {
  lambda: T,
}

impl<T: SimdFloatExt> SimdExp<T> {
  /// # Panics
  /// `lambda <= 0`.
  pub fn new(lambda: T) -> Self {
    let _ = exp_zig_tables();
    assert!(
      lambda > T::zero(),
      "lambda must satisfy `lambda > T::zero()`, got lambda = {lambda:?}"
    );
    Self { lambda }
  }

  /// The rate `λ`.
  pub fn lambda(&self) -> T {
    self.lambda
  }

  pub(crate) fn standard() -> Self {
    Self { lambda: T::one() }
  }

  pub(crate) fn draw_with<G: Rng + ?Sized>(&self, rng: &mut G) -> T {
    sample_exp1_one::<T, _>(&mut AnyRng(rng), exp_zig_tables()) * (T::one() / self.lambda)
  }
}

impl<T: SimdFloatExt> Default for SimdExp<T> {
  fn default() -> Self {
    Self::new(T::one())
  }
}

impl<T: SimdFloatExt> Sealed for SimdExp<T> {}

impl<T: SimdFloatExt> SimdDistribution for SimdExp<T> {
  type State<R: SimdRngExt> = StreamState<T, R, 64>;

  fn init<R: SimdRngExt, S: SeedExt>(&self, seed: &S) -> (StreamState<T, R, 64>, u64) {
    let stream_seed = seed.next_seed();
    (
      StreamState {
        rng: R::from_seed(stream_seed),
        buf: Buffered::new(),
      },
      stream_seed,
    )
  }
}

impl<T: SimdFloatExt> SimdKernel for SimdExp<T> {
  type Item = T;

  #[inline]
  fn fill<R: SimdRngExt>(&self, state: &mut StreamState<T, R, 64>, out: &mut [T]) {
    fill_exp_scaled(out, &mut state.rng, T::one() / self.lambda);
  }

  #[inline]
  fn next<R: SimdRngExt>(&self, state: &mut StreamState<T, R, 64>) -> T {
    let StreamState { rng, buf } = state;
    buf.pop(|b| fill_exp_scaled(b, rng, T::one() / self.lambda))
  }
}

impl<T: SimdFloatExt> Distribution<T> for SimdExp<T> {
  /// One scalar Ziggurat draw from the caller's rng.
  fn sample<G: Rng + ?Sized>(&self, rng: &mut G) -> T {
    self.draw_with(rng)
  }
}

impl<T: SimdFloatExt> crate::traits::DistributionExt for SimdExp<T> {
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

py_distribution!(PyExp, SimdExp,
  sig: (lambda_, seed=None, dtype=None),
  params: (lambda_: f64)
);

#[cfg(test)]
mod tests;
