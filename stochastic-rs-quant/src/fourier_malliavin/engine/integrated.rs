//! Integrated volatility / covariance / leverage / vol-of-vol / quarticity estimators.

use ndarray::Array1;
use num_complex::Complex;

use super::FMVol;
use crate::traits::FloatExt;

impl<T: FloatExt> FMVol<T> {
  /// Integrated variance.
  ///
  /// $$\widehat{IV}=\frac{T^2}{2N+1}\sum_{|k|\le N}c_k(dx)\,c_{-k}(dx)$$
  pub fn integrated_variance(&self) -> T {
    let c = self.center();
    let big_n = self.n_freq;
    let mut sum = Complex::<T>::new(T::zero(), T::zero());
    for k in -(big_n as i64)..=(big_n as i64) {
      let idx = (c as i64 + k) as usize;
      let idx_neg = (c as i64 - k) as usize;
      sum = sum + self.dx[idx] * self.dx[idx_neg];
    }
    self.period * self.period * sum.re / T::from_usize_(2 * big_n + 1)
  }

  /// Integrated covariance with another process.
  ///
  /// Uses Fejér-weighted coefficients. The two engines must share the same period.
  pub fn integrated_covariance(&self, other: &Self) -> T {
    let big_n = self.n_freq.min(other.n_freq);
    let big_n_plus_1 = T::from_usize_(big_n + 1);
    let c_self = self.center();
    let c_other = other.center();

    let mut sum = Complex::<T>::new(T::zero(), T::zero());
    for k in -(big_n as i64)..=(big_n as i64) {
      let fejer = T::one() - T::from_usize_(k.unsigned_abs() as usize) / big_n_plus_1;
      let cj = other.dx[(c_other as i64 + k) as usize];
      let ci_neg = self.dx[(c_self as i64 - k) as usize];
      sum = sum + cj * fejer * ci_neg;
    }
    self.period * self.period * sum.re / big_n_plus_1
  }

  /// Integrated leverage.
  pub fn integrated_leverage(&self, m_freq: Option<usize>) -> T {
    let big_m = self.resolve_m(m_freq);
    let const_ = self.const_();
    let big_m_plus_1 = T::from_usize_(big_m + 1);
    let c_v = self.vol_coeffs(big_m);
    let c = self.center();

    let len = 2 * big_m + 1;
    let mut c_dv = Array1::<Complex<T>>::zeros(len);
    for (j, k) in (-(big_m as i64)..=(big_m as i64)).enumerate() {
      let k_t = T::from_f64_fast(k as f64);
      let fejer = T::one() - T::from_usize_(k.unsigned_abs() as usize) / big_m_plus_1;
      c_dv[j] = Complex::<T>::new(T::zero(), k_t * const_) * c_v[j] * fejer;
    }

    let mut sum = Complex::<T>::new(T::zero(), T::zero());
    for (j, k) in (-(big_m as i64)..=(big_m as i64)).enumerate() {
      sum = sum + c_dv[j] * self.dx[(c as i64 - k) as usize];
    }
    self.period * self.period * sum.re / big_m_plus_1
  }

  /// Integrated volatility of volatility.
  pub fn integrated_volvol(&self, m_freq: Option<usize>) -> T {
    let big_m = self.resolve_m_volvol(m_freq);
    let const_ = self.const_();
    let big_m_plus_1 = T::from_usize_(big_m + 1);
    let c_v = self.vol_coeffs(big_m);

    let len = 2 * big_m + 1;
    let mut c_dv = Array1::<Complex<T>>::zeros(len);
    let mut c_dv2 = Array1::<Complex<T>>::zeros(len);
    for (j, k) in (-(big_m as i64)..=(big_m as i64)).enumerate() {
      let diff = Complex::<T>::new(T::zero(), T::from_f64_fast(k as f64) * const_);
      let fejer = T::one() - T::from_usize_(k.unsigned_abs() as usize) / big_m_plus_1;
      c_dv[j] = diff * c_v[j] * fejer;
      c_dv2[j] = diff * c_v[j];
    }

    let mut sum = Complex::<T>::new(T::zero(), T::zero());
    for j in 0..len {
      sum = sum + c_dv[j] * c_dv2[len - 1 - j];
    }
    self.period * self.period * sum.re / big_m_plus_1
  }

  /// Bias-corrected integrated volatility of volatility (Toscano-Livieri-
  /// Mancino-Marmi 2024 eq.4, arXiv:2112.14529v3).
  ///
  /// Subtracts `K · σ̂⁴_{n,N,M}` from the raw `integrated_volvol`, where
  /// the constant `K` is given by eq.(51) of the paper — see
  /// `compute_bias_correction_constant`.  Achieves the optimal
  /// `n^{1/4}` convergence rate (Theorem 3.1) at the price of
  /// **destroying the positivity of the estimator** (paper, p.5 / Sec.2);
  /// finite-sample realizations may therefore be negative.
  ///
  /// The default smoothing window is `M ≈ N^{0.25}` rather than the `N^{0.4}`
  /// used by [`integrated_volvol`]; the larger window applicable to the
  /// non-rate-optimal eq.(3) estimator (Theorem 3.6) leaves us outside the
  /// regime where the eq.(51) bias term dominates, and the correction
  /// under-shoots empirically.
  pub fn integrated_volvol_bias_corrected(&self, m_freq: Option<usize>) -> T {
    let big_m = self.resolve_m_volvol_bc(m_freq);
    let raw = self.integrated_volvol(Some(big_m));
    let quarticity = self.integrated_quarticity(Some(big_m));
    let k_const = self.compute_bias_correction_constant(big_m);
    raw - k_const * quarticity
  }

  /// Integrated quarticity.
  pub fn integrated_quarticity(&self, m_freq: Option<usize>) -> T {
    let big_m = self.resolve_m(m_freq);
    let c_v = self.vol_coeffs(big_m);
    let total = c_v.len();
    let mut sum = Complex::<T>::new(T::zero(), T::zero());
    for i in 0..total {
      sum = sum + c_v[i] * c_v[total - 1 - i];
    }
    self.period * sum.re
  }
}
