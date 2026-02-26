//! # Exp
//!
//! $$
//! f(x)=\lambda e^{-\lambda x},\ x\ge 0
//! $$
//!
use std::cell::UnsafeCell;
use std::sync::OnceLock;

use rand::Rng;
use rand_distr::Distribution;
use wide::CmpLt;
use wide::i32x8;

use super::SimdFloatExt;
use crate::simd_rng::SimdRng;

const ZIG_EXP_R: f64 = 7.697_117_470_131_487;
const ZIG_EXP_V: f64 = 3.949_659_822_581_572e-3;
const TABLE_SIZE: usize = 256;
const SMALL_EXP_THRESHOLD: usize = 16;

struct ExpZigTables {
  ke: [i32; TABLE_SIZE],
  we: [f64; TABLE_SIZE],
  fe: [f64; TABLE_SIZE],
}

static EXP_ZIG_TABLES: OnceLock<ExpZigTables> = OnceLock::new();

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

#[cold]
#[inline(never)]
fn efix<T: SimdFloatExt>(hz: i32, iz: usize, tables: &ExpZigTables, rng: &mut SimdRng) -> T {
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

pub struct SimdExpZig<T: SimdFloatExt, const N: usize = 64> {
  lambda: T,
  buffer: UnsafeCell<[T; N]>,
  index: UnsafeCell<usize>,
  simd_rng: UnsafeCell<SimdRng>,
}

impl<T: SimdFloatExt, const N: usize> SimdExpZig<T, N> {
  pub fn new(lambda: T) -> Self {
    let _ = exp_zig_tables();
    assert!(lambda > T::zero());
    assert!(N >= 8, "buffer size must be at least 8");
    Self {
      lambda,
      buffer: UnsafeCell::new([T::zero(); N]),
      index: UnsafeCell::new(N),
      simd_rng: UnsafeCell::new(SimdRng::new()),
    }
  }

  #[inline]
  fn sample_exp1_one(rng: &mut SimdRng, tables: &ExpZigTables) -> T {
    let hz = rng.next_i32();
    let iz = (hz & 0xFF) as usize;
    let abs_hz = hz.unsigned_abs() as i64;
    if abs_hz < tables.ke[iz] as i64 {
      T::from_f64_fast((abs_hz as f64) * tables.we[iz])
    } else {
      efix::<T>(hz, iz, tables, rng)
    }
  }

  #[inline]
  fn fill_exp1(buf: &mut [T], rng: &mut SimdRng) {
    let tables = exp_zig_tables();
    let len = buf.len();
    if len < SMALL_EXP_THRESHOLD {
      for x in buf.iter_mut() {
        *x = Self::sample_exp1_one(rng, tables);
      }
      return;
    }
    let mask255 = i32x8::splat(0xFF);
    let mut filled = 0;

    while filled < len {
      let hz = rng.next_i32x8();
      let iz = hz & mask255;
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
        let we_simd = T::simd_from_array(we_arr);
        let result = hz_float * we_simd;

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
              buf[filled] = efix::<T>(hz_arr[i], iz_arr[i] as usize, tables, rng);
              filled += 1;
            }
          }
        }
      }
    }
  }

  #[inline]
  fn scale_in_place(buf: &mut [T], factor: T) {
    let factor_simd = T::splat(factor);
    let mut chunks = buf.chunks_exact_mut(8);
    for chunk in &mut chunks {
      let mut tmp = [T::zero(); 8];
      tmp.copy_from_slice(chunk);
      let scaled = T::simd_to_array(T::simd_from_array(tmp) * factor_simd);
      chunk.copy_from_slice(&scaled);
    }
    for x in chunks.into_remainder().iter_mut() {
      *x = *x * factor;
    }
  }

  pub fn fill_slice<R: Rng + ?Sized>(&self, _rng: &mut R, out: &mut [T]) {
    let rng = unsafe { &mut *self.simd_rng.get() };
    Self::fill_exp1(out, rng);
    if self.lambda != T::one() {
      Self::scale_in_place(out, T::one() / self.lambda);
    }
  }

  fn refill_buffer(&self) {
    let rng = unsafe { &mut *self.simd_rng.get() };
    let buf = unsafe { &mut *self.buffer.get() };
    Self::fill_exp1(buf, rng);
    if self.lambda != T::one() {
      Self::scale_in_place(buf, T::one() / self.lambda);
    }
    unsafe {
      *self.index.get() = 0;
    }
  }
}

impl<T: SimdFloatExt, const N: usize> Clone for SimdExpZig<T, N> {
  fn clone(&self) -> Self {
    Self::new(self.lambda)
  }
}

impl<T: SimdFloatExt, const N: usize> Distribution<T> for SimdExpZig<T, N> {
  #[inline(always)]
  fn sample<R: Rng + ?Sized>(&self, _rng: &mut R) -> T {
    let index = unsafe { &mut *self.index.get() };
    if *index >= N {
      self.refill_buffer();
    }
    let val = unsafe { (*self.buffer.get())[*index] };
    *index += 1;
    val
  }
}

pub struct SimdExp<T: SimdFloatExt> {
  inner: SimdExpZig<T>,
}

impl<T: SimdFloatExt> SimdExp<T> {
  pub fn new(lambda: T) -> Self {
    Self {
      inner: SimdExpZig::new(lambda),
    }
  }

  pub fn fill_slice<R: Rng + ?Sized>(&self, rng: &mut R, out: &mut [T]) {
    self.inner.fill_slice(rng, out);
  }
}

impl<T: SimdFloatExt> Clone for SimdExp<T> {
  fn clone(&self) -> Self {
    Self::new(self.inner.lambda)
  }
}

impl<T: SimdFloatExt> Distribution<T> for SimdExp<T> {
  #[inline(always)]
  fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> T {
    self.inner.sample(rng)
  }
}

py_distribution!(PyExp, SimdExp,
  sig: (lambda_, dtype=None),
  params: (lambda_: f64)
);

#[cfg(test)]
mod tests {
  use rand_distr::Distribution;

  use super::SimdExp;
  use super::SimdExpZig;

  fn mean(samples: &[f64]) -> f64 {
    samples.iter().sum::<f64>() / samples.len() as f64
  }

  fn exp_cdf(x: f64, lambda: f64) -> f64 {
    if x <= 0.0 {
      0.0
    } else {
      1.0 - (-lambda * x).exp()
    }
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

  #[test]
  fn simd_exp_matches_theoretical_distribution() {
    const N: usize = 40_000;
    let lambda = 1.8_f64;
    let mean_target = 1.0 / lambda;

    let dist = SimdExp::<f64>::new(lambda);
    let mut rng = rand::rng();
    let mut samples: Vec<f64> = (0..N).map(|_| dist.sample(&mut rng)).collect();

    assert!(
      samples.iter().all(|x| x.is_finite() && *x >= 0.0),
      "invalid exponential sample encountered"
    );

    let mean_emp = mean(&samples);
    let mean_se = mean_target / (N as f64).sqrt();
    assert!(
      (mean_emp - mean_target).abs() < 6.0 * mean_se,
      "exp mean mismatch: emp={mean_emp}, target={mean_target}, se={mean_se}"
    );

    let d = ks_statistic(&mut samples, |x| exp_cdf(x, lambda));
    let ks_critical = 2.0 / (N as f64).sqrt();
    assert!(
      d < ks_critical,
      "exp KS statistic too large: D={d}, critical={ks_critical}"
    );
  }

  #[test]
  fn simd_exp_zig_fill_slice_matches_theoretical_distribution() {
    const N: usize = 32_000;
    let lambda = 0.65_f64;

    let dist = SimdExpZig::<f64>::new(lambda);
    let mut rng = rand::rng();
    let mut samples = vec![0.0_f64; N];
    dist.fill_slice(&mut rng, &mut samples);

    assert!(
      samples.iter().all(|x| x.is_finite() && *x >= 0.0),
      "invalid exponential sample encountered"
    );

    let d = ks_statistic(&mut samples, |x| exp_cdf(x, lambda));
    let ks_critical = 2.0 / (N as f64).sqrt();
    assert!(
      d < ks_critical,
      "exp-zig KS statistic too large: D={d}, critical={ks_critical}"
    );
  }
}
