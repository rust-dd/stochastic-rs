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
use wide::CmpLt;

use super::SimdFloatExt;
use crate::simd_rng::SimdRng;

struct ZigTables {
  kn: [i32; 128],
  wn: [f64; 128],
  fn_tab: [f64; 128],
}

static ZIG_TABLES: OnceLock<ZigTables> = OnceLock::new();

fn zig_tables() -> &'static ZigTables {
  ZIG_TABLES.get_or_init(|| {
    let mut kn = [0i32; 128];
    let mut wn = [0.0f64; 128];
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

    ZigTables { kn, wn, fn_tab }
  })
}

#[cold]
#[inline(never)]
fn nfix<T: SimdFloatExt>(hz: i32, iz: usize, tables: &ZigTables, rng: &mut SimdRng) -> T {
  const R_TAIL: f64 = 3.442620;
  let mut hz = hz;
  let mut iz = iz;

  loop {
    let x = hz as f64 * tables.wn[iz];

    if iz == 0 {
      loop {
        let u1: f64 = rng.random();
        let u2: f64 = rng.random();
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

    if tables.fn_tab[iz] + rng.random::<f64>() * (tables.fn_tab[iz - 1] - tables.fn_tab[iz])
      < (-0.5 * x * x).exp()
    {
      return T::from_f64_fast(x);
    }

    hz = rng.random::<i32>();
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
/// `fill_slice()` bypasses the buffer entirely.
pub struct SimdNormal<T: SimdFloatExt, const N: usize = 64> {
  mean: T,
  std_dev: T,
  buffer: UnsafeCell<[T; N]>,
  index: UnsafeCell<usize>,
  simd_rng: UnsafeCell<SimdRng>,
}

impl<T: SimdFloatExt, const N: usize> SimdNormal<T, N> {
  pub fn new(mean: T, std_dev: T) -> Self {
    let _ = zig_tables();
    assert!(std_dev > T::zero());
    assert!(N >= 8, "buffer size must be at least 8");
    Self {
      mean,
      std_dev,
      buffer: UnsafeCell::new([T::zero(); N]),
      index: UnsafeCell::new(N),
      simd_rng: UnsafeCell::new(SimdRng::new()),
    }
  }

  pub fn fill_slice<R: Rng + ?Sized>(&self, _rng: &mut R, out: &mut [T]) {
    self.fill_slice_fast(out);
  }

  #[inline]
  pub fn fill_slice_fast(&self, out: &mut [T]) {
    let rng = unsafe { &mut *self.simd_rng.get() };
    Self::fill_ziggurat(out, rng, self.mean, self.std_dev);
  }

  #[inline]
  pub fn fill_16<R: Rng + ?Sized>(&self, _rng: &mut R, out16: &mut [T]) {
    debug_assert!(out16.len() >= 16);
    let rng = unsafe { &mut *self.simd_rng.get() };
    Self::fill_ziggurat(&mut out16[..16], rng, self.mean, self.std_dev);
  }

  fn refill_buffer(&self) {
    let rng = unsafe { &mut *self.simd_rng.get() };
    let buf = unsafe { &mut *self.buffer.get() };
    Self::fill_ziggurat(buf.as_mut_slice(), rng, self.mean, self.std_dev);
    unsafe {
      *self.index.get() = 0;
    }
  }

  #[inline]
  fn fill_ziggurat(buf: &mut [T], rng: &mut SimdRng, mean: T, std_dev: T) {
    let len = buf.len();
    let tables = zig_tables();
    let mean_simd = T::splat(mean);
    let std_dev_simd = T::splat(std_dev);
    let mask127 = i32x8::splat(127);
    let mut filled = 0;

    while filled < len {
      let hz = rng.next_i32x8();
      let iz = hz & mask127;
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

        let wn_arr: [T; 8] = [
          T::from_f64_fast(*tables.wn.get_unchecked(iz_arr[0] as usize)),
          T::from_f64_fast(*tables.wn.get_unchecked(iz_arr[1] as usize)),
          T::from_f64_fast(*tables.wn.get_unchecked(iz_arr[2] as usize)),
          T::from_f64_fast(*tables.wn.get_unchecked(iz_arr[3] as usize)),
          T::from_f64_fast(*tables.wn.get_unchecked(iz_arr[4] as usize)),
          T::from_f64_fast(*tables.wn.get_unchecked(iz_arr[5] as usize)),
          T::from_f64_fast(*tables.wn.get_unchecked(iz_arr[6] as usize)),
          T::from_f64_fast(*tables.wn.get_unchecked(iz_arr[7] as usize)),
        ];
        let hz_float = T::simd_from_i32x8(hz);
        let wn_simd = T::simd_from_array(wn_arr);
        let result = hz_float * wn_simd;

        if accept.all() {
          let scaled = mean_simd + std_dev_simd * result;
          let scaled_arr = T::simd_to_array(scaled);
          let take = (len - filled).min(8);
          buf[filled..filled + take].copy_from_slice(&scaled_arr[..take]);
          filled += take;
        } else {
          let hz_arr = hz.to_array();
          let accept_arr = accept.to_array();
          let result_arr = T::simd_to_array(result);
          for i in 0..8 {
            if filled >= len {
              break;
            }
            if accept_arr[i] != 0 {
              buf[filled] = mean + std_dev * result_arr[i];
              filled += 1;
            } else {
              let x = nfix::<T>(hz_arr[i], iz_arr[i] as usize, tables, rng);
              buf[filled] = mean + std_dev * x;
              filled += 1;
            }
          }
        }
      }
    }
  }
}

impl<T: SimdFloatExt, const N: usize> SimdNormal<T, N> {
  pub fn fill_standard_fast(&self, out: &mut [T]) {
    let rng = unsafe { &mut *self.simd_rng.get() };
    Self::fill_ziggurat_standard(out, rng);
  }

  #[inline]
  pub fn sample_pair<R: Rng + ?Sized>(&self, _rng: &mut R) -> (T, T) {
    let index = unsafe { &mut *self.index.get() };
    if *index + 1 >= N {
      self.refill_buffer();
    }
    let buf = unsafe { &*self.buffer.get() };
    let a = buf[*index];
    let b = buf[*index + 1];
    *index += 2;
    (a, b)
  }

  #[inline]
  fn refill_buffer_standard(&self) {
    let rng = unsafe { &mut *self.simd_rng.get() };
    let buf = unsafe { &mut *self.buffer.get() };
    Self::fill_ziggurat_standard(buf.as_mut_slice(), rng);
    unsafe {
      *self.index.get() = 0;
    }
  }

  #[inline]
  pub fn sample_pair_standard<R: Rng + ?Sized>(&self, _rng: &mut R) -> (T, T) {
    let index = unsafe { &mut *self.index.get() };
    if *index + 1 >= N {
      self.refill_buffer_standard();
    }
    let buf = unsafe { &*self.buffer.get() };
    let a = buf[*index];
    let b = buf[*index + 1];
    *index += 2;
    (a, b)
  }

  #[inline]
  fn fill_ziggurat_standard(buf: &mut [T], rng: &mut SimdRng) {
    let len = buf.len();
    let tables = zig_tables();
    let mask127 = i32x8::splat(127);
    let mut filled = 0;

    while filled < len {
      let hz = rng.next_i32x8();
      let iz = hz & mask127;
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

        let wn_arr: [T; 8] = [
          T::from_f64_fast(*tables.wn.get_unchecked(iz_arr[0] as usize)),
          T::from_f64_fast(*tables.wn.get_unchecked(iz_arr[1] as usize)),
          T::from_f64_fast(*tables.wn.get_unchecked(iz_arr[2] as usize)),
          T::from_f64_fast(*tables.wn.get_unchecked(iz_arr[3] as usize)),
          T::from_f64_fast(*tables.wn.get_unchecked(iz_arr[4] as usize)),
          T::from_f64_fast(*tables.wn.get_unchecked(iz_arr[5] as usize)),
          T::from_f64_fast(*tables.wn.get_unchecked(iz_arr[6] as usize)),
          T::from_f64_fast(*tables.wn.get_unchecked(iz_arr[7] as usize)),
        ];
        let hz_float = T::simd_from_i32x8(hz);
        let wn_simd = T::simd_from_array(wn_arr);
        let result = hz_float * wn_simd;

        if accept.all() {
          let result_arr = T::simd_to_array(result);
          let take = (len - filled).min(8);
          buf[filled..filled + take].copy_from_slice(&result_arr[..take]);
          filled += take;
        } else {
          let hz_arr = hz.to_array();
          let accept_arr = accept.to_array();
          let result_arr = T::simd_to_array(result);
          for i in 0..8 {
            if filled >= len {
              break;
            }
            if accept_arr[i] != 0 {
              buf[filled] = result_arr[i];
              filled += 1;
            } else {
              buf[filled] = nfix::<T>(hz_arr[i], iz_arr[i] as usize, tables, rng);
              filled += 1;
            }
          }
        }
      }
    }
  }
}

impl<T: SimdFloatExt, const N: usize> Distribution<T> for SimdNormal<T, N> {
  fn sample<R: Rng + ?Sized>(&self, _rng: &mut R) -> T {
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
  sig: (mean, std_dev, dtype=None),
  params: (mean: f64, std_dev: f64)
);