//! # Core
//!
//! $$
//! \operatorname{Cov}(\Delta B_i^H,\Delta B_j^H)=\tfrac12\left(|k+1|^{2H}-2|k|^{2H}+|k-1|^{2H}\right),\ k=i-j
//! $$
//!
use std::sync::Arc;

use ndarray::concatenate;
use ndarray::prelude::*;
use ndrustfft::ndfft_inplace_par;
use ndrustfft::ndfft_par;
use ndrustfft::FftHandler;
use num_complex::Complex;

use crate::traits::FloatExt;

pub struct FGN<T: FloatExt> {
  /// Hurst exponent controlling roughness and long-memory.
  pub hurst: T,
  /// Number of discrete simulation points (or samples).
  pub n: usize,
  /// Total simulation horizon (defaults to 1 when omitted).
  pub t: Option<T>,
  /// Model parameter controlling process dynamics.
  pub offset: usize,
  pub(crate) out_len: usize,
  pub(crate) scale: T,
  /// Model parameter controlling process dynamics.
  pub sqrt_eigenvalues: Arc<Array1<T>>,
  /// Model parameter controlling process dynamics.
  pub fft_handler: Arc<FftHandler<T>>,
}

impl<T: FloatExt> FGN<T> {
  pub fn dt(&self) -> T {
    self.t.unwrap_or(T::one()) / T::from_usize_(self.n)
  }

  #[must_use]
  pub fn new(hurst: T, n: usize, t: Option<T>) -> Self {
    if !(T::zero()..=T::one()).contains(&hurst) {
      panic!("Hurst parameter must be between 0 and 1");
    }

    let offset = n.next_power_of_two() - n;
    let out_len = n;
    let n = n.next_power_of_two();
    let mut r = Array1::linspace(T::zero(), T::from_usize_(n), n + 1);
    let f2 = T::from_usize_(2);
    r.mapv_inplace(|x| {
      if x == T::zero() {
        T::one()
      } else {
        T::from_f64_fast(0.5)
          * ((x + T::one()).powf(f2 * hurst) - f2 * x.powf(f2 * hurst)
            + (x - T::one()).powf(f2 * hurst))
      }
    });
    let r = concatenate(
      Axis(0),
      #[allow(clippy::reversed_empty_ranges)]
      &[r.view(), r.slice(s![..;-1]).slice(s![1..-1]).view()],
    )
    .unwrap();
    let data = r.mapv(|v| Complex::new(v, T::zero()));
    let r_fft = FftHandler::new(r.len());
    let mut eig_fft = Array1::<Complex<T>>::zeros(r.len());
    ndfft_par(&data, &mut eig_fft, &r_fft, 0);
    let norm = T::from_usize_(2 * n);
    let mut sqrt_eigenvalues = Array1::<T>::zeros(r.len());
    for (dst, eig) in sqrt_eigenvalues.iter_mut().zip(eig_fft.iter()) {
      let lambda = eig.re / norm;
      *dst = if lambda > T::zero() {
        lambda.sqrt()
      } else {
        T::zero()
      };
    }

    Self {
      hurst,
      n,
      offset,
      out_len,
      t,
      scale: T::from_usize_(n).powf(-hurst) * t.unwrap_or(T::one()).powf(hurst),
      sqrt_eigenvalues: Arc::new(sqrt_eigenvalues),
      fft_handler: Arc::new(FftHandler::new(2 * n)),
    }
  }

  pub(crate) fn sample_cpu(&self) -> Array1<T> {
    let len = 2 * self.n;
    let mut fgn = Array1::<T>::zeros(self.out_len);

    T::with_fgn_complex_scratch(len, |rnd| {
      // SAFETY: Complex<T> is repr(C) with layout {re: T, im: T}, identical to [T; 2]
      let flat = unsafe { std::slice::from_raw_parts_mut(rnd.as_mut_ptr() as *mut T, 2 * len) };
      T::fill_standard_normal_slice(flat);
      for (z, &w) in rnd.iter_mut().zip(self.sqrt_eigenvalues.iter()) {
        z.re = z.re * w;
        z.im = z.im * w;
      }

      let mut rnd_view = ArrayViewMut1::from(rnd);
      ndfft_inplace_par(&mut rnd_view, &*self.fft_handler, 0);
      let src = rnd_view.slice(s![1..self.out_len + 1]);
      for (dst, c) in fgn.iter_mut().zip(src.iter()) {
        *dst = c.re * self.scale;
      }
    });

    fgn
  }
}