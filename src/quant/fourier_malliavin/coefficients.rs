//! Core Fourier coefficient computation for the Fourier-Malliavin volatility method.
//!
//! See Sanfelici & Toscano (2024), arXiv:2402.00172, §2 for the mathematical details.

use num_complex::Complex;

use crate::traits::FloatExt;

/// Compute discrete Fourier coefficients of price increments.
///
/// $$c_k(dx_n) = \frac{1}{T}\sum_{l=0}^{n-1} e^{-i\frac{2\pi}{T}k\,t_l}\,\delta_l(x)$$
///
/// where $\delta_l(x)=x(t_{l+1})-x(t_l)$.
///
/// Returns a vector of length `2*max_freq+1` with index mapping: frequency `k` → index `k + max_freq`.
pub fn fourier_coefficients_dx<T: FloatExt>(
  prices: &[T],
  times: &[T],
  period: T,
  max_freq: usize,
) -> Vec<Complex<T>> {
  assert_eq!(prices.len(), times.len(), "prices and times must have the same length");
  let n = prices.len() - 1;
  assert!(n > 0, "need at least 2 price observations");
  assert!(max_freq < n, "max_freq must be smaller than n");

  let const_ = T::from_f64_fast(std::f64::consts::TAU) / period;

  // Increments
  let r: Vec<T> = (0..n).map(|l| prices[l + 1] - prices[l]).collect();

  // Phase per unit frequency at each observation time: neg_phase[l] = -const_ * t[l]
  let neg_phases: Vec<T> = (0..n).map(|l| -const_ * times[l]).collect();

  // Positive frequency coefficients (unnormalised): sum_l exp(k * neg_phase[l]) * r[l]
  let mut c_pos = vec![Complex::<T>::new(T::zero(), T::zero()); max_freq];
  for k in 1..=max_freq {
    let k_t = T::from_usize_(k);
    let mut re = T::zero();
    let mut im = T::zero();
    for l in 0..n {
      let phase = k_t * neg_phases[l];
      re = re + phase.cos() * r[l];
      im = im + phase.sin() * r[l];
    }
    c_pos[k - 1] = Complex::new(re, im);
  }

  // c_0 = sum(r)
  let c_0: T = r.iter().copied().sum();

  // Assemble full array with 1/T scaling:
  //   [c_{-max_freq}, ..., c_{-1}, c_0, c_1, ..., c_{max_freq}]
  // For real increments c_{-k} = conj(c_k).
  let total = 2 * max_freq + 1;
  let inv_t = T::one() / period;
  let mut coeffs = vec![Complex::<T>::new(T::zero(), T::zero()); total];

  for k in 1..=max_freq {
    coeffs[max_freq - k] = c_pos[k - 1].conj() * inv_t;
    coeffs[max_freq + k] = c_pos[k - 1] * inv_t;
  }
  coeffs[max_freq] = Complex::new(c_0 * inv_t, T::zero());

  coeffs
}

/// Compute volatility / covariance Fourier coefficients via convolution.
///
/// $$c_k(\Sigma^{ij}_{n,N})=\frac{T}{2N+1}\sum_{|s|\le N}c_s(dx^i)\,c_{k-s}(dx^j)$$
///
/// For *variance*, pass the same slice for both `dx_a` and `dx_b`.
///
/// Both input arrays must satisfy `max_freq ≥ n_freq + m_freq`.
/// Returns coefficients for `k = −m_freq, …, m_freq` (length `2*m_freq+1`).
pub fn convolution_coefficients<T: FloatExt>(
  dx_a: &[Complex<T>],
  dx_b: &[Complex<T>],
  period: T,
  n_freq: usize,
  m_freq: usize,
) -> Vec<Complex<T>> {
  let center_a = (dx_a.len() - 1) / 2;
  let center_b = (dx_b.len() - 1) / 2;

  let total_out = 2 * m_freq + 1;
  let mut result = vec![Complex::<T>::new(T::zero(), T::zero()); total_out];
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
