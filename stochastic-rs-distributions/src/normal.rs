//! # Normal
//!
//! $$
//! f(x)=\frac{1}{\sigma\sqrt{2\pi}}\exp\!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)
//! $$
//!
use std::cell::UnsafeCell;
use std::sync::OnceLock;

use rand::Rng;
use rand_distr::Distribution;
use wide::i32x8;

use super::SimdFloatExt;
use crate::simd_rng::SeedExt;
use crate::simd_rng::SimdRng;
use crate::simd_rng::SimdRngExt;
use crate::simd_rng::Unseeded;

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
fn nfix<T: SimdFloatExt, R: SimdRngExt>(hz: i32, iz: usize, tables: &ZigTables, rng: &mut R) -> T {
  const R_TAIL: f64 = 3.442620;
  let mut hz = hz;
  let mut iz = iz;

  loop {
    let x = hz as f64 * tables.wn[iz];

    if iz == 0 {
      loop {
        let u1: f64 = rng.next_f64();
        let u2: f64 = rng.next_f64();
        let x_tail = -0.2904764 * (-u1.ln());
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

/// SIMD-accelerated normal (Gaussian) distribution using the Ziggurat algorithm.
/// ~97% of samples use a branchless SIMD fast path (integer compare + multiply),
/// with scalar fallback only for the rare ~3% edge cases.
///
/// The const generic `N` controls the internal buffer size for `sample()` calls.
/// Larger values reduce per-sample overhead but use more stack space.
///
/// The third generic, `R: SimdRngExt`, selects the backing RNG. The default
/// [`SimdRng`] is the single-stream production engine; the
/// `dual-stream-rng` cargo feature enables `R = SimdRngDual` via the
/// `SimdNormalDual` type alias, whose Ziggurat main loop consumes two
/// independent engine batches per iteration
/// ([`SimdRngExt::HAS_PAIR_ILP`]). Measured +3–10 % (≈5 % typical) on
/// 128-bit NEON Normal fills — modest because the table-gather chain
/// dominates; expected to matter more on wider-SIMD hardware.
pub struct SimdNormal<T: SimdFloatExt, const N: usize = 64, R: SimdRngExt = SimdRng> {
  mean: T,
  std_dev: T,
  buffer: UnsafeCell<[T; N]>,
  index: UnsafeCell<usize>,
  simd_rng: UnsafeCell<R>,
}

impl<T: SimdFloatExt, const N: usize, R: SimdRngExt> SimdNormal<T, N, R> {
  /// Creates a normal distribution with the given mean, standard deviation,
  /// and seed strategy.
  ///
  /// Mirrors the `Gbm::new(..., seed: S)`-style constructor used elsewhere
  /// in the workspace so seed handling is uniform across processes and
  /// distributions. Pass [`Unseeded`](crate::simd_rng::Unseeded) for an
  /// auto-seeded RNG, or [`Deterministic::new(seed)`](crate::simd_rng::Deterministic)
  /// for a reproducible stream.
  ///
  /// The seed source is taken by reference so a single seed can fan out
  /// independent RNGs to several sub-components (e.g. `SimdStudentT`
  /// builds both a `SimdNormal` and a `SimdChiSquared` from one seed),
  /// each call advancing the source's internal state.
  #[inline]
  pub fn new<S: SeedExt>(mean: T, std_dev: T, seed: &S) -> Self {
    let _ = zig_tables();
    assert!(std_dev > T::zero());
    assert!(N >= 8, "buffer size must be at least 8");
    Self {
      mean,
      std_dev,
      buffer: UnsafeCell::new([T::zero(); N]),
      index: UnsafeCell::new(N),
      simd_rng: UnsafeCell::new(seed.rng_ext::<R>()),
    }
  }

  /// Returns a single N(mean, std_dev) sample using the internal SIMD RNG.
  #[inline]
  pub fn sample_fast(&self) -> T {
    let index = unsafe { &mut *self.index.get() };
    if *index >= N {
      self.refill_buffer();
    }
    let buf = unsafe { &mut *self.buffer.get() };
    let z = buf[*index];
    *index += 1;
    z
  }

  /// Fills a slice with normally distributed samples.
  /// The `_rng` argument is accepted for API compatibility but ignored;
  /// the internal SIMD RNG is used instead.
  pub fn fill_slice<Rr: Rng + ?Sized>(&self, _rng: &mut Rr, out: &mut [T]) {
    self.fill_slice_fast(out);
  }

  /// Fills a slice with normally distributed samples using the internal SIMD RNG directly.
  #[inline]
  pub fn fill_slice_fast(&self, out: &mut [T]) {
    let rng = unsafe { &mut *self.simd_rng.get() };
    Self::fill_ziggurat(out, rng, self.mean, self.std_dev);
  }

  /// Fills exactly 16 elements with normally distributed samples.
  /// Optimized hot-path for small fixed-size buffers.
  #[inline]
  pub fn fill_16<Rr: Rng + ?Sized>(&self, _rng: &mut Rr, out16: &mut [T]) {
    debug_assert!(out16.len() >= 16);
    let rng = unsafe { &mut *self.simd_rng.get() };
    Self::fill_ziggurat(&mut out16[..16], rng, self.mean, self.std_dev);
  }

  /// Refills the internal sample buffer (used by the `Distribution::sample` trait impl).
  fn refill_buffer(&self) {
    let rng = unsafe { &mut *self.simd_rng.get() };
    let buf = unsafe { &mut *self.buffer.get() };
    Self::fill_ziggurat(buf.as_mut_slice(), rng, self.mean, self.std_dev);
    unsafe {
      *self.index.get() = 0;
    }
  }

  /// Processes one 8-lane Ziggurat batch from `hz` into `out[..8]`.
  /// `STANDARD` folds the mean / std_dev scaling away at compile time.
  #[inline(always)]
  fn zig_batch8<const STANDARD: bool>(
    hz: i32x8,
    tables: &ZigTables,
    mean: T,
    std_dev: T,
    mean_simd: T::Simd,
    std_dev_simd: T::Simd,
    rng: &mut R,
    out: &mut [T],
  ) {
    let iz = hz & i32x8::splat(127);
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

  /// Core Ziggurat fill: generates `N(mean, std_dev)` (or `N(0, 1)` when
  /// `STANDARD`) samples into `buf` — 8-wide SIMD on the fast-accept path
  /// (~97% of samples), scalar `nfix` for the rare edge cases.
  ///
  /// When the bound `R` reports [`SimdRngExt::HAS_PAIR_ILP`] the main loop
  /// consumes [`next_i32x8_pair`](SimdRngExt::next_i32x8_pair) and processes
  /// 16 lanes per iteration, so the two independent engine state updates
  /// can overlap the batches' table lookups on an out-of-order core. The
  /// branch is a monomorphised constant — single-stream codegen is
  /// unchanged. Measured +3–10 % on 128-bit NEON (the gather chain
  /// dominates the rest of the cost).
  fn fill_zig_impl<const STANDARD: bool>(buf: &mut [T], rng: &mut R, mean: T, std_dev: T) {
    let len = buf.len();
    let tables = zig_tables();
    if len < SMALL_NORMAL_THRESHOLD {
      for x in buf.iter_mut() {
        let z = Self::sample_one_standard(rng, tables);
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
        Self::zig_batch8::<STANDARD>(
          hz_a,
          tables,
          mean,
          std_dev,
          mean_simd,
          std_dev_simd,
          rng,
          &mut buf[filled..filled + 8],
        );
        Self::zig_batch8::<STANDARD>(
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
      Self::zig_batch8::<STANDARD>(
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
      let z = Self::sample_one_standard(rng, tables);
      buf[filled] = if STANDARD { z } else { mean + std_dev * z };
      filled += 1;
    }
  }

  /// `N(mean, std_dev)` fill — thin wrapper over [`Self::fill_zig_impl`].
  #[inline]
  fn fill_ziggurat(buf: &mut [T], rng: &mut R, mean: T, std_dev: T) {
    Self::fill_zig_impl::<false>(buf, rng, mean, std_dev);
  }
}

impl<T: SimdFloatExt, const N: usize, R: SimdRngExt> Clone for SimdNormal<T, N, R> {
  fn clone(&self) -> Self {
    // Cloning a stochastic source means "give me an independent stream", so
    // the clone is auto-seeded regardless of how the original was created.
    Self::new(self.mean, self.std_dev, &Unseeded)
  }
}

impl<T: SimdFloatExt, const N: usize, R: SimdRngExt> crate::traits::DistributionExt
  for SimdNormal<T, N, R>
{
  fn characteristic_function(&self, t: f64) -> num_complex::Complex64 {
    let mu = self.mean.to_f64().unwrap();
    let sigma = self.std_dev.to_f64().unwrap();
    num_complex::Complex64::new(0.0, mu * t).exp() * (-0.5 * sigma * sigma * t * t).exp()
  }

  fn pdf(&self, x: f64) -> f64 {
    let mu = self.mean.to_f64().unwrap();
    let sigma = self.std_dev.to_f64().unwrap();
    let z = (x - mu) / sigma;
    crate::special::norm_pdf(z) / sigma
  }

  fn cdf(&self, x: f64) -> f64 {
    let mu = self.mean.to_f64().unwrap();
    let sigma = self.std_dev.to_f64().unwrap();
    crate::special::norm_cdf((x - mu) / sigma)
  }

  fn inv_cdf(&self, p: f64) -> f64 {
    let mu = self.mean.to_f64().unwrap();
    let sigma = self.std_dev.to_f64().unwrap();
    mu + sigma * crate::special::ndtri(p)
  }

  fn mean(&self) -> f64 {
    self.mean.to_f64().unwrap()
  }

  fn median(&self) -> f64 {
    self.mean.to_f64().unwrap()
  }

  fn mode(&self) -> f64 {
    self.mean.to_f64().unwrap()
  }

  fn variance(&self) -> f64 {
    let s = self.std_dev.to_f64().unwrap();
    s * s
  }

  fn skewness(&self) -> f64 {
    0.0
  }

  fn kurtosis(&self) -> f64 {
    0.0
  }

  fn entropy(&self) -> f64 {
    let s = self.std_dev.to_f64().unwrap();
    0.5 * (2.0 * std::f64::consts::PI * std::f64::consts::E * s * s).ln()
  }

  fn moment_generating_function(&self, t: f64) -> f64 {
    let mu = self.mean.to_f64().unwrap();
    let s = self.std_dev.to_f64().unwrap();
    (mu * t + 0.5 * s * s * t * t).exp()
  }
}

impl<T: SimdFloatExt, const N: usize, R: SimdRngExt> SimdNormal<T, N, R> {
  /// Fills a slice with standard normal N(0,1) samples using the internal SIMD RNG.
  pub fn fill_standard_fast(&self, out: &mut [T]) {
    let rng = unsafe { &mut *self.simd_rng.get() };
    Self::fill_ziggurat_standard(out, rng);
  }

  /// Generates a single standard normal N(0,1) sample using the scalar Ziggurat path.
  #[inline]
  fn sample_one_standard(rng: &mut R, tables: &ZigTables) -> T {
    let hz = rng.next_i32();
    let iz = (hz & 127) as usize;
    if (hz.unsigned_abs() as i64) < tables.kn[iz] as i64 {
      T::from_f64_fast(hz as f64 * tables.wn[iz])
    } else {
      nfix::<T, R>(hz, iz, tables, rng)
    }
  }

  /// Core Ziggurat fill for standard normal N(0,1) samples — thin wrapper
  /// over [`Self::fill_zig_impl`] with the scaling compiled out.
  #[inline]
  fn fill_ziggurat_standard(buf: &mut [T], rng: &mut R) {
    Self::fill_zig_impl::<true>(buf, rng, T::zero(), T::one());
  }
}

impl<T: SimdFloatExt, const N: usize, R: SimdRngExt> Distribution<T> for SimdNormal<T, N, R> {
  /// Returns a single N(mean, std_dev) sample.
  /// Internally draws from a pre-filled buffer and refills it when exhausted.
  fn sample<Rr: Rng + ?Sized>(&self, _rng: &mut Rr) -> T {
    let index = unsafe { &mut *self.index.get() };
    if *index >= N {
      self.refill_buffer();
    }
    let buf = unsafe { &mut *self.buffer.get() };
    let z = buf[*index];
    *index += 1;
    z
  }
}

py_distribution!(PyNormal, SimdNormal,
  sig: (mean, std_dev, seed=None, dtype=None),
  params: (mean: f64, std_dev: f64)
);

#[cfg(test)]
mod tests {
  use rand_distr::Distribution;
  use stochastic_rs_core::simd_rng::Unseeded;

  use super::SimdNormal;
  use crate::special::erf;

  fn mean(samples: &[f64]) -> f64 {
    samples.iter().sum::<f64>() / samples.len() as f64
  }

  fn normal_cdf(x: f64, mean: f64, std_dev: f64) -> f64 {
    let z = (x - mean) / (std_dev * 2.0_f64.sqrt());
    0.5 * (1.0 + erf(z))
  }

  fn ks_statistic(samples: &mut [f64], mut cdf: impl FnMut(f64) -> f64) -> f64 {
    samples.sort_by(f64::total_cmp);
    let n = samples.len() as f64;
    let mut d = 0.0_f64;
    for (i, &x) in samples.iter().enumerate() {
      let f = cdf(x).clamp(0.0, 1.0);
      let i_f = i as f64;
      let d_plus = ((i_f + 1.0) / n - f).abs();
      let d_minus = (f - i_f / n).abs();
      d = d.max(d_plus.max(d_minus));
    }
    d
  }

  /// The dual-engine pair path interleaves batches from engines A and B —
  /// every lane (including B's) must still be N(0, 1).
  #[cfg(feature = "dual-stream-rng")]
  #[test]
  fn simd_normal_dual_pair_path_matches_theoretical_distribution() {
    const N: usize = 40_000;
    let dist = crate::SimdNormalDual::<f64>::new(0.0, 1.0, &Unseeded);
    let mut samples = vec![0.0_f64; N];
    dist.fill_standard_fast(&mut samples);
    assert!(samples.iter().all(|x| x.is_finite()));
    let d = ks_statistic(&mut samples, |x| normal_cdf(x, 0.0, 1.0));
    let ks_critical = 2.0 / (N as f64).sqrt();
    assert!(
      d < ks_critical,
      "dual-path normal KS statistic too large: D={d}, critical={ks_critical}"
    );
  }

  #[test]
  fn simd_normal_matches_theoretical_distribution() {
    const N: usize = 40_000;
    let mu = -0.75_f64;
    let sigma = 1.35_f64;

    let dist = SimdNormal::<f64>::new(mu, sigma, &Unseeded);
    let mut rng = rand::rng();
    let mut samples: Vec<f64> = (0..N).map(|_| dist.sample(&mut rng)).collect();

    assert!(
      samples.iter().all(|x| x.is_finite()),
      "non-finite normal sample encountered"
    );

    let mean_emp = mean(&samples);
    let mean_se = sigma / (N as f64).sqrt();
    assert!(
      (mean_emp - mu).abs() < 6.0 * mean_se,
      "normal mean mismatch: emp={mean_emp}, target={mu}, se={mean_se}"
    );

    let d = ks_statistic(&mut samples, |x| normal_cdf(x, mu, sigma));
    let ks_critical = 2.0 / (N as f64).sqrt();
    assert!(
      d < ks_critical,
      "normal KS statistic too large: D={d}, critical={ks_critical}"
    );
  }
}
