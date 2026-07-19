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
  /// Uses Fejér-weighted coefficients. The engines must share a period and
  /// Fourier time origin.
  pub fn integrated_covariance(&self, other: &Self) -> T {
    self
      .try_integrated_covariance(other)
      .expect("invalid integrated covariance configuration")
  }

  /// Fallible variant of [`Self::integrated_covariance`].
  pub fn try_integrated_covariance(&self, other: &Self) -> anyhow::Result<T> {
    self.validate_compatible_period(other)?;
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
    Ok(self.period * self.period * sum.re / big_n_plus_1)
  }

  /// Integrated leverage.
  pub fn integrated_leverage(&self, m_freq: Option<usize>) -> T {
    self
      .try_integrated_leverage(m_freq)
      .expect("invalid integrated leverage window")
  }

  /// Fallible variant of [`Self::integrated_leverage`].
  pub fn try_integrated_leverage(&self, m_freq: Option<usize>) -> anyhow::Result<T> {
    let big_m = self.resolve_m(m_freq);
    self.validate_m_window(big_m)?;
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
    Ok(self.period * self.period * sum.re / big_m_plus_1)
  }

  /// Integrated volatility of volatility.
  pub fn integrated_volvol(&self, m_freq: Option<usize>) -> T {
    self
      .try_integrated_volvol(m_freq)
      .expect("invalid integrated volatility-of-volatility window")
  }

  /// Fallible variant of [`Self::integrated_volvol`].
  pub fn try_integrated_volvol(&self, m_freq: Option<usize>) -> anyhow::Result<T> {
    let big_m = self.resolve_m_volvol(m_freq);
    self.validate_m_window(big_m)?;
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
    Ok(self.period * self.period * sum.re / big_m_plus_1)
  }

  /// Bias-corrected integrated volatility of volatility (Toscano-Livieri-
  /// Mancino-Marmi 2024 eq.4, arXiv:2112.14529v3).
  ///
  /// Applies equation (4) coefficient by coefficient. For a general period,
  /// the integrated quarticity multiplier is `(2π)² K / T`; at `T = 2π`
  /// this becomes the paper's `2π K`. The correction can make finite-sample
  /// estimates negative.
  pub fn integrated_volvol_bias_corrected(&self, m_freq: Option<usize>) -> T {
    self
      .try_integrated_volvol_bias_corrected(m_freq)
      .expect("invalid bias-corrected integrated volatility-of-volatility window")
  }

  /// Fallible variant of [`Self::integrated_volvol_bias_corrected`].
  pub fn try_integrated_volvol_bias_corrected(&self, m_freq: Option<usize>) -> anyhow::Result<T> {
    let big_m = self.resolve_m_volvol_bc(m_freq);
    self.validate_m_window(big_m)?;
    let coefficient = self.bias_corrected_volvol_coefficients(big_m, 0)[0];
    Ok(self.period * coefficient.re)
  }

  /// Integrated quarticity.
  pub fn integrated_quarticity(&self, m_freq: Option<usize>) -> T {
    self
      .try_integrated_quarticity(m_freq)
      .expect("invalid integrated quarticity window")
  }

  /// Fallible variant of [`Self::integrated_quarticity`].
  pub fn try_integrated_quarticity(&self, m_freq: Option<usize>) -> anyhow::Result<T> {
    let big_m = self.resolve_m(m_freq);
    self.validate_m_window(big_m)?;
    let c_v = self.vol_coeffs(big_m);
    let total = c_v.len();
    let mut sum = Complex::<T>::new(T::zero(), T::zero());
    for i in 0..total {
      sum = sum + c_v[i] * c_v[total - 1 - i];
    }
    Ok(self.period * sum.re)
  }
}
