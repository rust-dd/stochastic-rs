//! Core Fourier coefficient computation for the Fourier-Malliavin volatility method.
//!
//! See Sanfelici & Toscano (2024), arXiv:2402.00172, §2 for the mathematical details.

use ndarray::Array1;
use ndrustfft::FftHandler;
use ndrustfft::ndfft;
use num_complex::Complex;

use crate::traits::FloatExt;

/// Compute discrete Fourier coefficients of price increments (general irregular grid).
///
/// $$c_k(dx_n) = \frac{1}{T}\sum_{l=0}^{n-1} e^{-i\frac{2\pi}{T}k\,t_l}\,\delta_l(x)$$
///
/// Returns an `Array1` of length `2*max_freq+1`.
/// Index mapping: frequency `k` → index `k + max_freq`.
pub fn fourier_coefficients_dx<T: FloatExt>(
  prices: &[T],
  times: &[T],
  period: T,
  max_freq: usize,
) -> Array1<Complex<T>> {
  assert_eq!(
    prices.len(),
    times.len(),
    "prices and times must have the same length"
  );
  let n = prices.len() - 1;
  assert!(n > 0, "need at least 2 price observations");
  assert!(max_freq < n, "max_freq must be smaller than n");

  let const_ = T::from_f64_fast(std::f64::consts::TAU) / period;
  let inv_t = T::one() / period;

  // Increments
  let r: Array1<T> = Array1::from_vec((0..n).map(|l| prices[l + 1] - prices[l]).collect());

  // Phases: neg_phases[l] = -const_ * t[l]
  let neg_phases: Array1<T> = Array1::from_vec((0..n).map(|l| -const_ * times[l]).collect());

  // Positive frequency coefficients (unnormalised)
  let mut c_pos = Array1::<Complex<T>>::zeros(max_freq);
  for k in 1..=max_freq {
    let k_t = T::from_usize_(k);
    let mut re = T::zero();
    let mut im = T::zero();
    for l in 0..n {
      let phase = k_t * neg_phases[l];
      re += phase.cos() * r[l];
      im += phase.sin() * r[l];
    }
    c_pos[k - 1] = Complex::new(re, im);
  }

  // c_0 = sum(r)
  let c_0: T = r.sum();

  // Assemble: [c_{-max_freq}, …, c_0, …, c_{max_freq}] with 1/T scaling
  let total = 2 * max_freq + 1;
  let mut coeffs = Array1::<Complex<T>>::zeros(total);

  for k in 1..=max_freq {
    coeffs[max_freq - k] = c_pos[k - 1].conj() * inv_t;
    coeffs[max_freq + k] = c_pos[k - 1] * inv_t;
  }
  coeffs[max_freq] = Complex::new(c_0 * inv_t, T::zero());

  coeffs
}

/// FFT-accelerated Fourier coefficients for **uniformly spaced** observations.
///
/// Assumes `t_l = l · T / n`. Runs in O(n log n) instead of O(n · max_freq).
pub fn fourier_coefficients_dx_uniform<T: FloatExt>(
  prices: &[T],
  period: T,
  max_freq: usize,
) -> Array1<Complex<T>> {
  let n = prices.len() - 1;
  assert!(n > 0, "need at least 2 price observations");
  assert!(max_freq < n, "max_freq must be smaller than n");

  let inv_t = T::one() / period;

  // Increments
  let mut input = Array1::<Complex<T>>::zeros(n);
  for l in 0..n {
    input[l] = Complex::new(prices[l + 1] - prices[l], T::zero());
  }

  // Forward FFT: DFT(r)[k] = Σ_{l} r[l] · exp(-i·2π·k·l/n)
  let mut fft_out = Array1::<Complex<T>>::zeros(n);
  let handler = FftHandler::<T>::new(n);
  ndfft(&input, &mut fft_out, &handler, 0);

  // Assemble two-sided spectrum: c_k = DFT[k] / T
  let total = 2 * max_freq + 1;
  let mut coeffs = Array1::<Complex<T>>::zeros(total);

  // k = 0
  coeffs[max_freq] = fft_out[0] * inv_t;

  // k = 1..max_freq
  for k in 1..=max_freq {
    coeffs[max_freq + k] = fft_out[k] * inv_t;
    coeffs[max_freq - k] = fft_out[k].conj() * inv_t; // c_{-k} = conj(c_k)
  }

  coeffs
}

/// Compute volatility / covariance Fourier coefficients via convolution.
///
/// $$c_k(\Sigma^{ij}_{n,N})=\frac{T}{2N+1}\sum_{|s|\le N}c_s(dx^i)\,c_{k-s}(dx^j)$$
///
/// For *variance*, pass the same array for both `dx_a` and `dx_b`.
///
/// Both inputs must satisfy `max_freq ≥ n_freq + m_freq`.
/// Returns coefficients for `k = −m_freq, …, m_freq` (length `2*m_freq+1`).
pub fn convolution_coefficients<T: FloatExt>(
  dx_a: &Array1<Complex<T>>,
  dx_b: &Array1<Complex<T>>,
  period: T,
  n_freq: usize,
  m_freq: usize,
) -> Array1<Complex<T>> {
  let center_a = (dx_a.len() - 1) / 2;
  let center_b = (dx_b.len() - 1) / 2;

  let total_out = 2 * m_freq + 1;
  let mut result = Array1::<Complex<T>>::zeros(total_out);
  let scale = period / T::from_usize_(2 * n_freq + 1);

  for (j, k) in (-(m_freq as i64)..=(m_freq as i64)).enumerate() {
    let mut sum = Complex::<T>::new(T::zero(), T::zero());
    for s in -(n_freq as i64)..=(n_freq as i64) {
      let idx_a = (center_a as i64 + s) as usize;
      let idx_b = (center_b as i64 + (k - s)) as usize;
      sum = sum + dx_a[idx_a] * dx_b[idx_b];
    }
    result[j] = sum * scale;
  }

  result
}
