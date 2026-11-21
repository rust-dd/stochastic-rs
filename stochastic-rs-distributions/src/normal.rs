//! # Normal
//!
//! $$
//! f(x)=\frac{1}{\sigma\sqrt{2\pi}}\exp\!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)
//! $$
//!
//! Sampling: Marsaglia, G., Tsang, W.W. (2000), "The Ziggurat Method for Generating Random Variables", *Journal of Statistical Software* 5(8), DOI 10.18637/jss.v005.i08.

use std::mem::MaybeUninit;
use std::sync::OnceLock;

use rand::Rng;
use rand::distr::Distribution;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::SimdRngExt;
use stochastic_rs_core::simd_rng::as_uninit_mut;
use wide::i32x8;
use wide::u32x8;

use crate::seeded::StreamState;
use crate::source::AnyRng;
use crate::source::Source;
use crate::traits::SimdFloatExt;
use crate::traits::distribution::Sealed;
use crate::traits::distribution::SimdDistribution;
use crate::traits::distribution::SimdKernel;

/// Ziggurat tables of the normal: `kn` the fast-accept thresholds, `wn`/`wn_f32` the rectangle
/// widths in f64 and f32, `fn_tab` the values `e^{−x²/2}` at their edges.
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

/// The ~3 % of draws outside a ziggurat rectangle: the tail (`iz == 0`) by the exponential method,
/// the rest by rejection within the rectangle.
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
        // Marsaglia & Tsang's tail: `x_tail` is the distance *past* `R_TAIL`, so it adds; negated,
        // every tail draw would land inside and the law lose all mass beyond 3.44σ.
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

/// What every batch of one fill shares: the tables, and the affine map in scalar and lane form.
struct ZigFill<'a, T: SimdFloatExt> {
  tables: &'a ZigTables,
  mu: T,
  sigma: T,
  mu_simd: T::Simd,
  sigma_simd: T::Simd,
}

/// One 8-lane Ziggurat batch from `hz` into `out[..8]`; `STANDARD` compiles the affine map out.
#[inline(always)]
fn zig_batch8<T: SimdFloatExt, R: SimdRngExt, const STANDARD: bool>(
  hz: i32x8,
  fill: &ZigFill<'_, T>,
  rng: &mut R,
  out: &mut [MaybeUninit<T>],
) {
  let ZigFill {
    tables,
    mu,
    sigma,
    mu_simd,
    sigma_simd,
  } = *fill;
  let iz = hz & LAYER_MASK;
  let iz_arr = iz.to_array();
  unsafe {
    let kn_vals = u32x8::new([
      *tables.kn.get_unchecked(iz_arr[0] as usize) as u32,
      *tables.kn.get_unchecked(iz_arr[1] as usize) as u32,
      *tables.kn.get_unchecked(iz_arr[2] as usize) as u32,
      *tables.kn.get_unchecked(iz_arr[3] as usize) as u32,
      *tables.kn.get_unchecked(iz_arr[4] as usize) as u32,
      *tables.kn.get_unchecked(iz_arr[5] as usize) as u32,
      *tables.kn.get_unchecked(iz_arr[6] as usize) as u32,
      *tables.kn.get_unchecked(iz_arr[7] as usize) as u32,
    ]);
    // `abs` wraps `i32::MIN` to itself, which read unsigned is 2³¹, past every layer's bound.
    let abs_hz = u32x8::new(hz.abs().to_array().map(|v| v as u32));
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
        mu_simd + sigma_simd * result
      };
      out[..8].copy_from_slice(&T::simd_to_array(scaled).map(MaybeUninit::new));
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
        out[i].write(if STANDARD { z } else { mu + sigma * z });
      }
    }
  }
}

/// Core Ziggurat fill (16 lanes per step from two engines with `R::HAS_PAIR_ILP`); kept out of line
/// because inlined into a pop loop's refill this kernel slows the loop.
#[inline(never)]
fn fill_zig_impl<T: SimdFloatExt, R: SimdRngExt, const STANDARD: bool>(
  buf: &mut [MaybeUninit<T>],
  rng: &mut R,
  mu: T,
  sigma: T,
) {
  let len = buf.len();
  let tables = zig_tables();
  if len < SMALL_NORMAL_THRESHOLD {
    for x in buf.iter_mut() {
      let z = sample_one_standard::<T, R>(rng, tables);
      x.write(if STANDARD { z } else { mu + sigma * z });
    }
    return;
  }
  let fill = ZigFill {
    tables,
    mu,
    sigma,
    mu_simd: T::splat(mu),
    sigma_simd: T::splat(sigma),
  };
  let mut filled = 0;

  if R::HAS_PAIR_ILP {
    while filled + 16 <= len {
      let (hz_a, hz_b) = rng.next_i32x8_pair();
      zig_batch8::<T, R, STANDARD>(hz_a, &fill, rng, &mut buf[filled..filled + 8]);
      zig_batch8::<T, R, STANDARD>(hz_b, &fill, rng, &mut buf[filled + 8..filled + 16]);
      filled += 16;
    }
  }
  while filled + 8 <= len {
    let hz = rng.next_i32x8();
    zig_batch8::<T, R, STANDARD>(hz, &fill, rng, &mut buf[filled..filled + 8]);
    filled += 8;
  }
  while filled < len {
    let z = sample_one_standard::<T, R>(rng, tables);
    buf[filled].write(if STANDARD { z } else { mu + sigma * z });
    filled += 1;
  }
}

/// Normal law `N(mu, sigma²)`: parameters only; a [`Seeded`](crate::Seeded) stream draws it in bulk.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SimdNormal<T> {
  mu: T,
  sigma: T,
}

impl<T: SimdFloatExt> SimdNormal<T> {
  /// `N(mu, sigma²)`; panics unless `mu` is finite and `0 < sigma < ∞`.
  pub fn new(mu: T, sigma: T) -> Self {
    assert!(
      mu.is_finite(),
      "mu must satisfy `mu.is_finite()`, got mu = {mu:?}"
    );
    assert!(
      sigma.is_finite(),
      "sigma must satisfy `sigma.is_finite()`, got sigma = {sigma:?}"
    );
    let _ = zig_tables();
    assert!(
      sigma > T::zero(),
      "sigma must satisfy `sigma > T::zero()`, got sigma = {sigma:?}"
    );
    Self { mu, sigma }
  }

  /// The location `μ`.
  pub fn mu(&self) -> T {
    self.mu
  }

  /// The scale `σ`.
  pub fn sigma(&self) -> T {
    self.sigma
  }

  pub(crate) fn standard() -> Self {
    Self {
      mu: T::zero(),
      sigma: T::one(),
    }
  }

  pub(crate) fn fill_standard<R: SimdRngExt>(rng: &mut R, out: &mut [T]) {
    // SAFETY: `fill_zig_impl` writes every element of `out`.
    fill_zig_impl::<T, R, true>(unsafe { as_uninit_mut(out) }, rng, T::zero(), T::one());
  }

  pub(crate) fn draw_with<G: Rng + ?Sized>(&self, rng: &mut G) -> T {
    let z = sample_one_standard::<T, _>(&mut AnyRng(rng), zig_tables());
    self.mu + self.sigma * z
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
  fn fill<R: SimdRngExt>(&self, state: &mut StreamState<T, R, 64>, out: &mut [MaybeUninit<T>]) {
    fill_zig_impl::<T, R, false>(out, &mut state.rng, self.mu, self.sigma);
  }

  #[inline]
  fn next<R: SimdRngExt>(&self, state: &mut StreamState<T, R, 64>) -> T {
    let StreamState { rng, buf } = state;
    // SAFETY: `fill_zig_impl` writes every element of the buffer.
    buf.pop(|b| fill_zig_impl::<T, R, false>(unsafe { as_uninit_mut(b) }, rng, self.mu, self.sigma))
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
    let mu = self.mu.to_f64().unwrap();
    let sigma = self.sigma.to_f64().unwrap();
    Some(num_complex::Complex64::new(0.0, mu * t).exp() * (-0.5 * sigma * sigma * t * t).exp())
  }

  fn pdf(&self, x: f64) -> Option<f64> {
    let mu = self.mu.to_f64().unwrap();
    let sigma = self.sigma.to_f64().unwrap();
    let z = (x - mu) / sigma;
    Some(crate::special::norm_pdf(z) / sigma)
  }

  fn cdf(&self, x: f64) -> Option<f64> {
    let mu = self.mu.to_f64().unwrap();
    let sigma = self.sigma.to_f64().unwrap();
    Some(crate::special::norm_cdf((x - mu) / sigma))
  }

  fn quantile(&self, p: f64) -> Option<f64> {
    let mu = self.mu.to_f64().unwrap();
    let sigma = self.sigma.to_f64().unwrap();
    Some(mu + sigma * crate::special::ndtri(p))
  }

  fn mean(&self) -> Option<f64> {
    Some(self.mu.to_f64().unwrap())
  }

  fn median(&self) -> Option<f64> {
    Some(self.mu.to_f64().unwrap())
  }

  fn mode(&self) -> Option<f64> {
    Some(self.mu.to_f64().unwrap())
  }

  fn variance(&self) -> Option<f64> {
    let s = self.sigma.to_f64().unwrap();
    Some(s * s)
  }

  fn skewness(&self) -> Option<f64> {
    Some(0.0)
  }

  fn kurtosis(&self) -> Option<f64> {
    Some(0.0)
  }

  fn entropy(&self) -> Option<f64> {
    let s = self.sigma.to_f64().unwrap();
    Some(0.5 * (2.0 * std::f64::consts::PI * std::f64::consts::E * s * s).ln())
  }

  fn moment_generating_function(&self, t: f64) -> Option<f64> {
    let mu = self.mu.to_f64().unwrap();
    let s = self.sigma.to_f64().unwrap();
    Some((mu * t + 0.5 * s * s * t * t).exp())
  }
}

py_distribution!(PyNormal, SimdNormal,
  sig: (mu, sigma, seed=None, dtype=None),
  params: (mu: f64, sigma: f64)
);

#[cfg(test)]
mod tests;

#[cfg(test)]
mod zig_tests {
  use stochastic_rs_core::simd_rng::SimdRng;
  use wide::i32x8;

  use super::*;

  /// `i32::MIN` has no magnitude in `i32`: `abs` wraps, and a lane accepted on it lands on the
  /// layer-0 edge, a point mass of 2⁻³².
  #[test]
  fn minimum_hz_lane_is_rejected_into_the_fix_up() {
    let tables = zig_tables();
    let mut rng = SimdRng::from_seed(11);
    let mut out = [0.0_f64; 8];
    let hz = i32x8::new([i32::MIN, 1, 2, 3, 4, 5, 6, 7]);
    let fill = ZigFill {
      tables,
      mu: 0.0,
      sigma: 1.0,
      mu_simd: f64::splat(0.0),
      sigma_simd: f64::splat(1.0),
    };
    zig_batch8::<f64, SimdRng, true>(
      hz,
      &fill,
      &mut rng,
      // SAFETY: `zig_batch8` writes all eight lanes of `out`.
      unsafe { as_uninit_mut(&mut out) },
    );
    assert_ne!(
      out[0],
      i32::MIN as f64 * tables.wn[0],
      "lane 0 is the layer edge, not a draw"
    );
    assert!(
      out[0] <= -3.44,
      "lane 0 is a left-tail draw, got {}",
      out[0]
    );
  }
}
