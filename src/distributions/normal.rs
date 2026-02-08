use std::cell::UnsafeCell;
use std::sync::OnceLock;

use rand::Rng;
use rand_distr::Distribution;
use wide::CmpLt;
use wide::i32x8;

use super::SimdFloat;

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

fn nfix<T: SimdFloat, R: Rng + ?Sized>(hz: i32, iz: usize, tables: &ZigTables, rng: &mut R) -> T {
  const R_TAIL: f64 = 3.442620;
  let mut hz = hz;
  let mut iz = iz;

  loop {
    let x = hz as f64 * tables.wn[iz];

    if iz == 0 {
      loop {
        let u1: f64 = rng.random_range(0.0f64..1.0f64);
        let u2: f64 = rng.random_range(0.0f64..1.0f64);
        let x_tail = -0.2904764 * (-u1.ln());
        let y = -u2.ln();
        if y + y >= x_tail * x_tail {
          let val = if hz > 0 {
            R_TAIL + x_tail
          } else {
            -R_TAIL - x_tail
          };
          return T::from(val).unwrap();
        }
      }
    }

    if tables.fn_tab[iz]
      + rng.random_range(0.0f64..1.0f64) * (tables.fn_tab[iz - 1] - tables.fn_tab[iz])
      < (-0.5 * x * x).exp()
    {
      return T::from(x).unwrap();
    }

    hz = rng.random::<i32>();
    iz = (hz & 127) as usize;
    if (hz.unsigned_abs() as i64) < tables.kn[iz] as i64 {
      return T::from(hz as f64 * tables.wn[iz]).unwrap();
    }
  }
}

/// SIMD-accelerated normal (Gaussian) distribution using the Ziggurat algorithm.
/// Generates 16 samples at a time. ~97% of samples use a branchless SIMD fast path
/// (integer compare + multiply), with scalar fallback only for the rare ~3% edge cases.
pub struct SimdNormal<T: SimdFloat> {
  mean: T,
  std_dev: T,
  buffer: UnsafeCell<[T; 16]>,
  index: UnsafeCell<usize>,
}

impl<T: SimdFloat> SimdNormal<T> {
  pub fn new(mean: T, std_dev: T) -> Self {
    let _ = zig_tables();
    assert!(std_dev > T::zero());
    Self {
      mean,
      std_dev,
      buffer: UnsafeCell::new([T::zero(); 16]),
      index: UnsafeCell::new(16),
    }
  }

  pub fn fill_slice<R: Rng + ?Sized>(&self, rng: &mut R, out: &mut [T]) {
    let mut tmp = [T::zero(); 16];
    let mut chunks = out.chunks_exact_mut(16);
    for chunk in &mut chunks {
      Self::fill_ziggurat(&mut tmp, rng, self.mean, self.std_dev);
      chunk.copy_from_slice(&tmp);
    }
    let rem = chunks.into_remainder();
    if !rem.is_empty() {
      Self::fill_ziggurat(&mut tmp, rng, self.mean, self.std_dev);
      rem.copy_from_slice(&tmp[..rem.len()]);
    }
  }

  #[inline]
  pub fn fill_16<R: Rng + ?Sized>(&self, rng: &mut R, out16: &mut [T]) {
    debug_assert!(out16.len() >= 16);
    let mut tmp = [T::zero(); 16];
    Self::fill_ziggurat(&mut tmp, rng, self.mean, self.std_dev);
    out16[..16].copy_from_slice(&tmp);
  }

  fn refill_buffer<R: Rng + ?Sized>(&self, rng: &mut R) {
    let buf = unsafe { &mut *self.buffer.get() };
    Self::fill_ziggurat(buf, rng, self.mean, self.std_dev);
    unsafe {
      *self.index.get() = 0;
    }
  }

  fn fill_ziggurat<R: Rng + ?Sized>(buf: &mut [T; 16], rng: &mut R, mean: T, std_dev: T) {
    let tables = zig_tables();
    let mean_simd = T::splat(mean);
    let std_dev_simd = T::splat(std_dev);
    let mask127 = i32x8::splat(127);
    let mut filled = 0;

    while filled < 16 {
      let hz_arr: [i32; 8] = std::array::from_fn(|_| rng.random::<i32>());
      let hz = i32x8::new(hz_arr);
      let iz = hz & mask127;
      let iz_arr = iz.to_array();

      let kn_vals = i32x8::new(std::array::from_fn(|i| tables.kn[iz_arr[i] as usize]));
      let abs_hz = hz.abs();
      let accept = abs_hz.simd_lt(kn_vals);

      let wn_arr: [T; 8] =
        std::array::from_fn(|i| T::from(tables.wn[iz_arr[i] as usize]).unwrap());
      let hz_float = T::simd_from_i32x8(hz);
      let wn_simd = T::simd_from_array(wn_arr);
      let result = hz_float * wn_simd;

      if accept.all() {
        let scaled = mean_simd + std_dev_simd * result;
        let scaled_arr = T::simd_to_array(scaled);
        let take = (16 - filled).min(8);
        buf[filled..filled + take].copy_from_slice(&scaled_arr[..take]);
        filled += take;
      } else {
        let accept_arr = accept.to_array();
        let result_arr = T::simd_to_array(result);
        for i in 0..8 {
          if filled >= 16 {
            break;
          }
          if accept_arr[i] != 0 {
            buf[filled] = mean + std_dev * result_arr[i];
            filled += 1;
          } else {
            let x = nfix::<T, R>(hz_arr[i], iz_arr[i] as usize, tables, rng);
            buf[filled] = mean + std_dev * x;
            filled += 1;
          }
        }
      }
    }
  }
}

impl<T: SimdFloat> Distribution<T> for SimdNormal<T> {
  fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> T {
    let index = unsafe { &mut *self.index.get() };
    if *index >= 16 {
      self.refill_buffer(rng);
    }
    let buf = unsafe { &mut *self.buffer.get() };
    let z = buf[*index];
    *index += 1;
    z
  }
}
