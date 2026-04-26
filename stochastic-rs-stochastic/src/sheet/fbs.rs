//! # Fbs
//!
//! $$
//! \mathbb E[B^H(t_1,t_2)B^H(s_1,s_2)]=\prod_{j=1}^2\tfrac12\left(t_j^{2H_j}+s_j^{2H_j}-|t_j-s_j|^{2H_j}\right)
//! $$
//!
use ndarray::Array1;
use ndarray::Array2;
use ndarray::Axis;
use ndarray::linalg::kron;
use ndarray::s;
use ndrustfft::FftHandler;
use ndrustfft::ndfft;
use num_complex::Complex;

use stochastic_rs_distributions::normal::SimdNormal;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use crate::traits::FloatExt;
use crate::traits::ProcessExt;

#[derive(Debug, Clone)]
pub struct FBS<T: FloatExt, S: SeedExt = Unseeded> {
  /// Hurst exponent controlling roughness and long-memory.
  pub hurst: T,
  /// Number of parallel paths / first-grid resolution.
  pub m: usize,
  /// Number of discrete simulation points (or samples).
  pub n: usize,
  /// Risk-free rate / drift adjustment parameter.
  pub r: T,
  /// Seed strategy (compile-time: [`Unseeded`] or [`Deterministic`]).
  pub seed: S,
}

impl<T: FloatExt> FBS<T> {
  pub fn new(hurst: T, m: usize, n: usize, r: T) -> Self {
    Self {
      hurst,
      m,
      n,
      r,
      seed: Unseeded,
    }
  }
}

impl<T: FloatExt> FBS<T, Deterministic> {
  /// Create a new FBS model with a deterministic seed for reproducible output.
  pub fn seeded(hurst: T, m: usize, n: usize, r: T, seed: u64) -> Self {
    Self {
      hurst,
      m,
      n,
      r,
      seed: Deterministic(seed),
    }
  }
}

impl<T: FloatExt, S: SeedExt> ProcessExt<T> for FBS<T, S> {
  type Output = Array2<T>;

  fn sample(&self) -> Array2<T> {
    let (m, n, r) = (self.m, self.n, self.r);
    let alpha = T::from_usize_(2) * self.hurst;

    let tx = Array1::linspace(r / T::from_usize_(n), r, n);
    let ty = Array1::linspace(r / T::from_usize_(m), r, m);

    let mut cov = Array2::<T>::zeros((m, n));
    for i in 0..n {
      for j in 0..m {
        cov[[j, i]] = Self::rho((tx[i], ty[j]), (tx[0], ty[0]), r, alpha).0;
      }
    }

    let big_m = 2 * (m - 1);
    let big_n = 2 * (n - 1);
    let mut blk = Array2::<T>::zeros((big_m, big_n));

    blk.slice_mut(s![..m, ..n]).assign(&cov);

    blk
      .slice_mut(s![..m, n..])
      .assign(&cov.slice(s![.., 1..n - 1;-1]));

    blk
      .slice_mut(s![m.., ..n])
      .assign(&cov.slice(s![1..m - 1;-1, ..]));

    blk
      .slice_mut(s![m.., n..])
      .assign(&cov.slice(s![1..m - 1, 1..n - 1]).slice(s![..;-1, ..;-1]));

    let scale = T::from_usize_(4) * T::from_usize_(m - 1) * T::from_usize_(n - 1);
    let fft_handler0 = FftHandler::<T>::new(big_m);
    let fft_handler1 = FftHandler::<T>::new(big_n);

    let blk_c = blk.mapv(|v| Complex::new(v, T::zero()));
    let mut fft_tmp = Array2::<Complex<T>>::zeros((big_m, big_n));
    ndfft(&blk_c, &mut fft_tmp, &fft_handler0, 0);
    let mut fft_freq = Array2::<Complex<T>>::zeros((big_m, big_n));
    ndfft(&fft_tmp, &mut fft_freq, &fft_handler1, 1);

    let lam = fft_freq.mapv(|c| (c.re / scale).max(T::zero()).sqrt());

    let mut seed = self.seed;
    let normal = SimdNormal::<T, 64>::from_seed_source(T::zero(), T::one(), &mut seed);
    let z = Array2::from_shape_fn((big_m, big_n), |_| {
      Complex::new(normal.sample_fast(), normal.sample_fast())
    });

    let prod = lam.mapv(|v| Complex::new(v, T::zero())) * z;
    let mut fft_tmp2 = Array2::<Complex<T>>::zeros((big_m, big_n));
    ndfft(&prod, &mut fft_tmp2, &fft_handler0, 0);
    let mut result = Array2::<Complex<T>>::zeros((big_m, big_n));
    ndfft(&fft_tmp2, &mut result, &fft_handler1, 1);

    let mut field = Array2::<T>::zeros((m, n));
    for i in 0..m {
      for j in 0..n {
        field[[i, j]] = result[[i, j]].re;
      }
    }

    let (_, _, c2) = Self::rho((T::zero(), T::zero()), (T::zero(), T::zero()), r, alpha);

    let shift = field[[0, 0]];
    field.mapv_inplace(|v| v - shift);

    let normal_scalar = SimdNormal::<T>::from_seed_source(T::zero(), T::one(), &mut seed);
    let mut z_buf = [T::zero(); 2];
    normal_scalar.fill_slice_fast(&mut z_buf);
    let z1 = z_buf[0];
    let z2 = z_buf[1];

    let ty_scaled = &ty * z1;
    let tx_scaled = &tx * z2;
    let ty_mat = ty_scaled.insert_axis(Axis(1));
    let tx_mat = tx_scaled.insert_axis(Axis(0));

    let correction = kron(&ty_mat, &tx_mat) * (T::from_usize_(2) * c2).sqrt();
    field = &field + &correction;

    field
  }
}

impl<T: FloatExt, S: SeedExt> FBS<T, S> {
  fn rho(x: (T, T), y: (T, T), r: T, alpha: T) -> (T, T, T) {
    let one = T::one();
    let two = T::from_usize_(2);
    let three = T::from_usize_(3);
    let half = one / two;
    let one_point_five = three * half;

    let (beta, c2, c0) = if alpha <= one_point_five {
      let c2 = alpha * half;
      let c0 = one - alpha * half;
      (T::zero(), c2, c0)
    } else {
      let beta = alpha * (two - alpha) / (three * r * (r * r - one));
      let c2 = (alpha - beta * (r - one).powi(2) * (r + two)) * half;
      let c0 = beta * (r - one).powi(3) + one - c2;
      (beta, c2, c0)
    };

    let dx = x.0 - y.0;
    let dy = x.1 - y.1;
    let dist = (dx * dx + dy * dy).sqrt();
    let out = if dist <= one {
      c0 - dist.powf(alpha) + c2 * dist * dist
    } else if dist <= r {
      beta * (r - dist).powi(3) / dist
    } else {
      T::zero()
    };
    (out, c0, c2)
  }
}

py_process_2d!(PyFBS, FBS,
  sig: (hurst, m, n, r, seed=None, dtype=None),
  params: (hurst: f64, m: usize, n: usize, r: f64)
);
