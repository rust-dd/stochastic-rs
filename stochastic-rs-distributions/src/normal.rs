//! # Normal
//!
//! $$
//! f(x)=\frac{1}{\sigma\sqrt{2\pi}}\exp\!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)
//! $$
//!
//! Sampling: Marsaglia, G., Tsang, W.W. (2000), "The Ziggurat Method for Generating Random Variables", *Journal of Statistical Software* 5(8), DOI 10.18637/jss.v005.i08.

use std::sync::OnceLock;

use rand::Rng;
use rand::distr::Distribution;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::SimdRngExt;
use wide::i32x8;

use crate::seeded::StreamState;
use crate::source::AnyRng;
use crate::source::Source;
use crate::traits::SimdFloatExt;
use crate::traits::distribution::Sealed;
use crate::traits::distribution::SimdDistribution;
use crate::traits::distribution::SimdKernel;

/// Precomputed lookup tables for the Ziggurat algorithm (normal distribution).
/// `kn` holds threshold integers for the fast-accept test,
/// `wn`/`wn_f32` hold the width of each rectangle (f64 and f32),
/// `fn_tab` holds the function values f(x)=exp(-x²/2) at rectangle boundaries.
pub(crate) struct ZigTables {
  pub(crate) kn: [i32; 128],
  pub(crate) wn: [f64; 128],
  pub(crate) wn_f32: [f32; 128],
  pub(crate) fn_tab: [f64; 128],
}

static ZIG_TABLES: OnceLock<ZigTables> = OnceLock::new();
const SMALL_NORMAL_THRESHOLD: usize = 16;

/// Returns a reference to the lazily-initialized Ziggurat tables.
/// Uses `OnceLock` so the tables are computed only once per process.
pub(crate) fn zig_tables() -> &'static ZigTables {
  ZIG_TABLES.get_or_init(|| {
    let mut kn = [0i32; 128];
    let mut wn = [0.0f64; 128];
    let mut wn_f32 = [0.0f32; 128];
    let mut fn_tab = [0.0f64; 128];

    let mut dn = 3.442619855899f64;
    let vn = 9.91256303526217e-3f64;
    let m1 = 2147483648.0f64;

    let q = vn / (-0.5 * dn * dn).exp();

    let kn0 = (dn / q) * m1;
    kn[0] = if kn0 > i32::MAX as f64 {
      i32::MAX
    } else {
      kn0 as i32
    };
    kn[1] = 0;

    wn[0] = q / m1;
    wn[127] = dn / m1;

    fn_tab[0] = 1.0;
    fn_tab[127] = (-0.5 * dn * dn).exp();

    let mut tn = dn;
    for i in (1..=126).rev() {
      dn = (-2.0 * (vn / dn + (-0.5 * dn * dn).exp()).ln()).sqrt();
      let kn_val = (dn / tn) * m1;
      kn[i + 1] = if kn_val > i32::MAX as f64 {
        i32::MAX
      } else {
        kn_val as i32
      };
      tn = dn;
      fn_tab[i] = (-0.5 * dn * dn).exp();
      wn[i] = dn / m1;
    }

    for i in 0..128 {
      wn_f32[i] = wn[i] as f32;
    }

    ZigTables {
      kn,
      wn,
      wn_f32,
      fn_tab,
    }
  })
}

/// Scalar fallback for the ~3% of samples that fall outside a Ziggurat rectangle.
/// Handles tail sampling (iz==0) via the exponential-tail method and
/// rejection sampling for intermediate rectangles.
#[cold]
#[inline(never)]
fn nfix<T: SimdFloatExt, S: Source + ?Sized>(
  hz: i32,
  iz: usize,
  tables: &ZigTables,
  rng: &mut S,
) -> T {
  const R_TAIL: f64 = 3.442620;
  /// `1 / R_TAIL`, the scale of the tail's exponential draw.
  const R_TAIL_INV: f64 = 0.2904764;
  let mut hz = hz;
  let mut iz = iz;

  loop {
    let x = hz as f64 * tables.wn[iz];

    if iz == 0 {
      loop {
        let u1: f64 = rng.next_f64();
        let u2: f64 = rng.next_f64();
        // Marsaglia & Tsang's tail draw. `x_tail` is how far *past* the
        // ziggurat's boundary the sample falls, so it is positive: negated,
        // every tail draw lands back inside `R_TAIL` and the law loses
        // everything beyond 3.44σ — a kurtosis of 2.96, no four-sigma move
        // ever, and every Gaussian-driven process in the workspace short of
        // its tails.
        let x_tail = -u1.ln() * R_TAIL_INV;
        let y = -u2.ln();
        if y + y >= x_tail * x_tail {
          let val = if hz > 0 {
            R_TAIL + x_tail
          } else {
            -R_TAIL - x_tail
          };
          return T::from_f64_fast(val);
        }
      }
    }

    if tables.fn_tab[iz] + rng.next_f64() * (tables.fn_tab[iz - 1] - tables.fn_tab[iz])
      < (-0.5 * x * x).exp()
    {
      return T::from_f64_fast(x);
    }

    hz = rng.next_i32();
    iz = (hz & 127) as usize;
    if (hz.unsigned_abs() as i64) < tables.kn[iz] as i64 {
      return T::from_f64_fast(hz as f64 * tables.wn[iz]);
    }
  }
}

/// One standard normal draw on the scalar Ziggurat path, from a SIMD engine or the caller's rng.
#[inline]
fn sample_one_standard<T: SimdFloatExt, S: Source + ?Sized>(rng: &mut S, tables: &ZigTables) -> T {
  let hz = rng.next_i32();
  let iz = (hz & 127) as usize;
  if (hz.unsigned_abs() as i64) < tables.kn[iz] as i64 {
    T::from_f64_fast(hz as f64 * tables.wn[iz])
  } else {
    nfix::<T, S>(hz, iz, tables, rng)
  }
}

/// The 128-layer index mask, built at compile time: a run-time `splat` of a constant becomes
/// `memset_pattern16` calls on Darwin.
const LAYER_MASK: i32x8 = i32x8::splat(127);

/// One 8-lane Ziggurat batch from `hz` into `out[..8]`; `STANDARD` compiles the affine map out.
#[inline(always)]
fn zig_batch8<T: SimdFloatExt, R: SimdRngExt, const STANDARD: bool>(
  hz: i32x8,
  tables: &ZigTables,
  mean: T,
  std_dev: T,
  mean_simd: T::Simd,
  std_dev_simd: T::Simd,
  rng: &mut R,
  out: &mut [T],
) {
  let iz = hz & LAYER_MASK;
  let iz_arr = iz.to_array();
  unsafe {
    let kn_vals = i32x8::new([
      *tables.kn.get_unchecked(iz_arr[0] as usize),
      *tables.kn.get_unchecked(iz_arr[1] as usize),
      *tables.kn.get_unchecked(iz_arr[2] as usize),
      *tables.kn.get_unchecked(iz_arr[3] as usize),
      *tables.kn.get_unchecked(iz_arr[4] as usize),
      *tables.kn.get_unchecked(iz_arr[5] as usize),
      *tables.kn.get_unchecked(iz_arr[6] as usize),
      *tables.kn.get_unchecked(iz_arr[7] as usize),
    ]);
    let abs_hz = hz.abs();
    let accept = abs_hz.simd_lt(kn_vals);

    let wn_arr: [T; 8] = if T::PREFERS_F32_WN {
      [
        T::from_f32_fast(*tables.wn_f32.get_unchecked(iz_arr[0] as usize)),
        T::from_f32_fast(*tables.wn_f32.get_unchecked(iz_arr[1] as usize)),
        T::from_f32_fast(*tables.wn_f32.get_unchecked(iz_arr[2] as usize)),
        T::from_f32_fast(*tables.wn_f32.get_unchecked(iz_arr[3] as usize)),
        T::from_f32_fast(*tables.wn_f32.get_unchecked(iz_arr[4] as usize)),
        T::from_f32_fast(*tables.wn_f32.get_unchecked(iz_arr[5] as usize)),
        T::from_f32_fast(*tables.wn_f32.get_unchecked(iz_arr[6] as usize)),
        T::from_f32_fast(*tables.wn_f32.get_unchecked(iz_arr[7] as usize)),
      ]
    } else {
      [
        T::from_f64_fast(*tables.wn.get_unchecked(iz_arr[0] as usize)),
        T::from_f64_fast(*tables.wn.get_unchecked(iz_arr[1] as usize)),
        T::from_f64_fast(*tables.wn.get_unchecked(iz_arr[2] as usize)),
        T::from_f64_fast(*tables.wn.get_unchecked(iz_arr[3] as usize)),
        T::from_f64_fast(*tables.wn.get_unchecked(iz_arr[4] as usize)),
        T::from_f64_fast(*tables.wn.get_unchecked(iz_arr[5] as usize)),
        T::from_f64_fast(*tables.wn.get_unchecked(iz_arr[6] as usize)),
        T::from_f64_fast(*tables.wn.get_unchecked(iz_arr[7] as usize)),
      ]
    };
    let hz_float = T::simd_from_i32x8(hz);
    let result = hz_float * T::simd_from_array(wn_arr);

    if accept.all() {
      let scaled = if STANDARD {
        result
      } else {
        mean_simd + std_dev_simd * result
      };
      out[..8].copy_from_slice(&T::simd_to_array(scaled));
    } else {
      let hz_arr = hz.to_array();
      let accept_arr = accept.to_array();
      let result_arr = T::simd_to_array(result);
      for i in 0..8 {
        let z = if accept_arr[i] != 0 {
          result_arr[i]
        } else {
          nfix::<T, R>(hz_arr[i], iz_arr[i] as usize, tables, rng)
        };
        out[i] = if STANDARD { z } else { mean + std_dev * z };
      }
    }
  }
}

/// Core Ziggurat fill (16 lanes per step from two engines with `R::HAS_PAIR_ILP`); kept out of line because
/// inlined into a pop loop's refill this kernel slows the loop.
#[inline(never)]
fn fill_zig_impl<T: SimdFloatExt, R: SimdRngExt, const STANDARD: bool>(
  buf: &mut [T],
  rng: &mut R,
  mean: T,
  std_dev: T,
) {
  let len = buf.len();
  let tables = zig_tables();
  if len < SMALL_NORMAL_THRESHOLD {
    for x in buf.iter_mut() {
      let z = sample_one_standard::<T, R>(rng, tables);
      *x = if STANDARD { z } else { mean + std_dev * z };
    }
    return;
  }
  let mean_simd = T::splat(mean);
  let std_dev_simd = T::splat(std_dev);
  let mut filled = 0;

  if R::HAS_PAIR_ILP {
    while filled + 16 <= len {
      let (hz_a, hz_b) = rng.next_i32x8_pair();
      zig_batch8::<T, R, STANDARD>(
        hz_a,
        tables,
        mean,
        std_dev,
        mean_simd,
        std_dev_simd,
        rng,
        &mut buf[filled..filled + 8],
      );
      zig_batch8::<T, R, STANDARD>(
        hz_b,
        tables,
        mean,
        std_dev,
        mean_simd,
        std_dev_simd,
        rng,
        &mut buf[filled + 8..filled + 16],
      );
      filled += 16;
    }
  }
  while filled + 8 <= len {
    let hz = rng.next_i32x8();
    zig_batch8::<T, R, STANDARD>(
      hz,
      tables,
      mean,
      std_dev,
      mean_simd,
      std_dev_simd,
      rng,
      &mut buf[filled..filled + 8],
    );
    filled += 8;
  }
  while filled < len {
    let z = sample_one_standard::<T, R>(rng, tables);
    buf[filled] = if STANDARD { z } else { mean + std_dev * z };
    filled += 1;
  }
}

/// Normal law `N(mean, std_dev)`: parameters only; a [`Seeded`](crate::Seeded) stream draws it in bulk.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SimdNormal<T> {
  mean: T,
  std_dev: T,
}

impl<T: SimdFloatExt> SimdNormal<T> {
  /// # Panics
  /// `std_dev <= 0`.
  pub fn new(mean: T, std_dev: T) -> Self {
    let _ = zig_tables();
    assert!(
      std_dev > T::zero(),
      "std_dev must satisfy `std_dev > T::zero()`, got std_dev = {std_dev:?}"
    );
    Self { mean, std_dev }
  }

  /// The mean `μ`; on a concrete type it shadows the `f64` `DistributionExt::mean`.
  pub fn mean(&self) -> T {
    self.mean
  }

  /// The standard deviation `σ`.
  pub fn std_dev(&self) -> T {
    self.std_dev
  }

  pub(crate) fn standard() -> Self {
    Self {
      mean: T::zero(),
      std_dev: T::one(),
    }
  }

  pub(crate) fn fill_standard<R: SimdRngExt>(rng: &mut R, out: &mut [T]) {
    fill_zig_impl::<T, R, true>(out, rng, T::zero(), T::one());
  }

  pub(crate) fn draw_with<G: Rng + ?Sized>(&self, rng: &mut G) -> T {
    let z = sample_one_standard::<T, _>(&mut AnyRng(rng), zig_tables());
    self.mean + self.std_dev * z
  }
}

impl<T: SimdFloatExt> Default for SimdNormal<T> {
  fn default() -> Self {
    Self::new(T::zero(), T::one())
  }
}

impl<T: SimdFloatExt> Sealed for SimdNormal<T> {}

impl<T: SimdFloatExt> SimdDistribution for SimdNormal<T> {
  type State<R: SimdRngExt> = StreamState<T, R, 64>;

  fn init<R: SimdRngExt, S: SeedExt>(&self, seed: &S) -> (StreamState<T, R, 64>, u64) {
    StreamState::init(seed)
  }
}

impl<T: SimdFloatExt> SimdKernel for SimdNormal<T> {
  type Item = T;

  #[inline]
  fn fill<R: SimdRngExt>(&self, state: &mut StreamState<T, R, 64>, out: &mut [T]) {
    fill_zig_impl::<T, R, false>(out, &mut state.rng, self.mean, self.std_dev);
  }

  #[inline]
  fn next<R: SimdRngExt>(&self, state: &mut StreamState<T, R, 64>) -> T {
    let StreamState { rng, buf } = state;
    buf.pop(|b| fill_zig_impl::<T, R, false>(b, rng, self.mean, self.std_dev))
  }
}

impl<T: SimdFloatExt> Distribution<T> for SimdNormal<T> {
  /// One scalar Ziggurat draw from the caller's rng.
  fn sample<G: Rng + ?Sized>(&self, rng: &mut G) -> T {
    self.draw_with(rng)
  }
}

impl<T: SimdFloatExt> crate::traits::DistributionExt for SimdNormal<T> {
  fn characteristic_function(&self, t: f64) -> Option<num_complex::Complex64> {
    let mu = self.mean.to_f64().unwrap();
    let sigma = self.std_dev.to_f64().unwrap();
    Some(num_complex::Complex64::new(0.0, mu * t).exp() * (-0.5 * sigma * sigma * t * t).exp())
  }

  fn pdf(&self, x: f64) -> Option<f64> {
    let mu = self.mean.to_f64().unwrap();
    let sigma = self.std_dev.to_f64().unwrap();
    let z = (x - mu) / sigma;
    Some(crate::special::norm_pdf(z) / sigma)
  }

  fn cdf(&self, x: f64) -> Option<f64> {
    let mu = self.mean.to_f64().unwrap();
    let sigma = self.std_dev.to_f64().unwrap();
    Some(crate::special::norm_cdf((x - mu) / sigma))
  }

  fn quantile(&self, p: f64) -> Option<f64> {
    let mu = self.mean.to_f64().unwrap();
    let sigma = self.std_dev.to_f64().unwrap();
    Some(mu + sigma * crate::special::ndtri(p))
  }

  fn mean(&self) -> Option<f64> {
    Some(self.mean.to_f64().unwrap())
  }

  fn median(&self) -> Option<f64> {
    Some(self.mean.to_f64().unwrap())
  }

  fn mode(&self) -> Option<f64> {
    Some(self.mean.to_f64().unwrap())
  }

  fn variance(&self) -> Option<f64> {
    let s = self.std_dev.to_f64().unwrap();
    Some(s * s)
  }

  fn skewness(&self) -> Option<f64> {
    Some(0.0)
  }

  fn kurtosis(&self) -> Option<f64> {
    Some(0.0)
  }

  fn entropy(&self) -> Option<f64> {
    let s = self.std_dev.to_f64().unwrap();
    Some(0.5 * (2.0 * std::f64::consts::PI * std::f64::consts::E * s * s).ln())
  }

  fn moment_generating_function(&self, t: f64) -> Option<f64> {
    let mu = self.mean.to_f64().unwrap();
    let s = self.std_dev.to_f64().unwrap();
    Some((mu * t + 0.5 * s * s * t * t).exp())
  }
}

py_distribution!(PyNormal, SimdNormal,
  sig: (mean, std_dev, seed=None, dtype=None),
  params: (mean: f64, std_dev: f64)
);

#[cfg(test)]
mod tests;
