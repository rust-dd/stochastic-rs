//! Spot volatility / covariance / leverage / vol-of-vol / quarticity estimators.

use ndarray::Array1;
use num_complex::Complex;

use super::FmVol;
use super::helpers::fejer_inversion;
use super::validation::window_overflow;
use crate::QuantError;
use crate::fourier_malliavin::coefficients::convolution_coefficients;
use crate::traits::RealExt;

impl<T: RealExt> FmVol<T> {
  /// Spot variance at evaluation times `tau`.
  pub fn spot_variance(&self, tau: &[T], m_freq: Option<usize>) -> Array1<T> {
    self
      .try_spot_variance(tau, m_freq)
      .unwrap_or_else(|e| panic!("{e}"))
  }

  /// Fallible variant of [`Self::spot_variance`].
  pub fn try_spot_variance(
    &self,
    tau: &[T],
    m_freq: Option<usize>,
  ) -> Result<Array1<T>, QuantError> {
    let big_m = self.resolve_m(m_freq);
    self.validate_tau(tau)?;
    self.validate_m_window(big_m)?;
    let c_v = self.vol_coeffs(big_m);
    Ok(fejer_inversion(
      &c_v,
      big_m,
      self.period,
      tau,
      T::from_usize_(big_m + 1),
    ))
  }

  /// Spot variance with the FE Cesàro weight `1 − |k|/M` (MATLAB FSDA's `FE_spot_vol`), where FM
  /// uses `1 − |k|/(M+1)`; panics where [`Self::try_spot_variance_fe`] returns an error.
  pub fn spot_variance_fe(&self, tau: &[T], m_freq: Option<usize>) -> Array1<T> {
    self
      .try_spot_variance_fe(tau, m_freq)
      .unwrap_or_else(|e| panic!("{e}"))
  }

  /// Fallible variant of [`Self::spot_variance_fe`].
  pub fn try_spot_variance_fe(
    &self,
    tau: &[T],
    m_freq: Option<usize>,
  ) -> Result<Array1<T>, QuantError> {
    let big_m = m_freq.unwrap_or((self.n_freq as f64).sqrt() as usize);
    if big_m < 2 {
      return Err(QuantError::invalid(
        "M",
        format!("must be at least 2 for the Fejér kernel, got {big_m}"),
      ));
    }
    self.validate_tau(tau)?;
    self.validate_m_window(big_m)?;
    let c_v = self.vol_coeffs(big_m);
    Ok(fejer_inversion(
      &c_v,
      big_m,
      self.period,
      tau,
      T::from_usize_(big_m),
    ))
  }

  /// Spot covariance with another process at evaluation times `tau`.
  pub fn spot_covariance(&self, other: &Self, tau: &[T], m_freq: Option<usize>) -> Array1<T> {
    self
      .try_spot_covariance(other, tau, m_freq)
      .unwrap_or_else(|e| panic!("{e}"))
  }

  /// Fallible variant of [`Self::spot_covariance`].
  pub fn try_spot_covariance(
    &self,
    other: &Self,
    tau: &[T],
    m_freq: Option<usize>,
  ) -> Result<Array1<T>, QuantError> {
    self.validate_compatible_period(other)?;
    self.validate_tau(tau)?;
    let big_n = self.n_freq.min(other.n_freq);
    let big_m = m_freq.unwrap_or((big_n as f64).sqrt() as usize);
    if big_m == 0 {
      return Err(QuantError::invalid("M", "must be positive, got 0"));
    }
    let required = big_n
      .checked_add(big_m)
      .ok_or_else(|| window_overflow("N + M"))?;
    self.validate_stored_frequency(required)?;
    other.validate_stored_frequency(big_n)?;
    let c_c = convolution_coefficients(&other.dx, &self.dx, self.period, big_n, big_m);
    Ok(fejer_inversion(
      &c_c,
      big_m,
      self.period,
      tau,
      T::from_usize_(big_m + 1),
    ))
  }

  /// Spot leverage at evaluation times `tau`.
  pub fn spot_leverage(
    &self,
    tau: &[T],
    m_freq: Option<usize>,
    l_freq: Option<usize>,
  ) -> Array1<T> {
    self
      .try_spot_leverage(tau, m_freq, l_freq)
      .unwrap_or_else(|e| panic!("{e}"))
  }

  /// Fallible variant of [`Self::spot_leverage`].
  pub fn try_spot_leverage(
    &self,
    tau: &[T],
    m_freq: Option<usize>,
    l_freq: Option<usize>,
  ) -> Result<Array1<T>, QuantError> {
    let big_m = self.resolve_m(m_freq);
    let big_l = l_freq.unwrap_or((self.n_freq as f64).powf(0.25) as usize);
    self.validate_tau(tau)?;
    self.validate_leverage_window(big_m, big_l)?;
    let const_ = self.const_();
    let c = self.center();

    let c_v = self.vol_coeffs(big_m);

    let len_m = 2 * big_m + 1;
    let mut c_dv = Array1::<Complex<T>>::zeros(len_m);
    for (j, k) in (-(big_m as i64)..=(big_m as i64)).enumerate() {
      c_dv[j] = Complex::<T>::new(T::zero(), T::from_f64_fast(k as f64) * const_) * c_v[j];
    }

    let len_l = 2 * big_l + 1;
    let mut c_lev = Array1::<Complex<T>>::zeros(len_l);
    let scale = self.period / T::from_usize_(2 * big_m + 1);

    for (j_lev, k) in (-(big_l as i64)..=(big_l as i64)).enumerate() {
      let mut sum = Complex::<T>::new(T::zero(), T::zero());
      for (j_dv, s) in (-(big_m as i64)..=(big_m as i64)).enumerate() {
        let dx_idx = (c as i64 + k - s) as usize;
        sum = sum + c_dv[j_dv] * self.dx[dx_idx];
      }
      c_lev[j_lev] = sum * scale;
    }

    Ok(fejer_inversion(
      &c_lev,
      big_l,
      self.period,
      tau,
      T::from_usize_(big_l + 1),
    ))
  }

  /// Spot volatility of volatility at evaluation times `tau`.
  pub fn spot_volvol(&self, tau: &[T], m_freq: Option<usize>, l_freq: Option<usize>) -> Array1<T> {
    self
      .try_spot_volvol(tau, m_freq, l_freq)
      .unwrap_or_else(|e| panic!("{e}"))
  }

  /// Fallible variant of [`Self::spot_volvol`].
  pub fn try_spot_volvol(
    &self,
    tau: &[T],
    m_freq: Option<usize>,
    l_freq: Option<usize>,
  ) -> Result<Array1<T>, QuantError> {
    let big_m = self.resolve_m_volvol(m_freq);
    let big_l = l_freq.unwrap_or((self.n_freq as f64).powf(0.2) as usize);
    self.validate_tau(tau)?;
    self.validate_ml_window(big_m, big_l)?;
    let const_ = self.const_();
    let mm = big_m + big_l;

    let c_v = self.vol_coeffs(big_m);
    let len_m = 2 * big_m + 1;
    let mut c_dv = Array1::<Complex<T>>::zeros(len_m);
    for (j, k) in (-(big_m as i64)..=(big_m as i64)).enumerate() {
      c_dv[j] = Complex::<T>::new(T::zero(), T::from_f64_fast(k as f64) * const_) * c_v[j];
    }

    let c_v2 = convolution_coefficients(&self.dx, &self.dx, self.period, self.n_freq, mm);
    let len_mm = 2 * mm + 1;
    let mut c_dv2 = Array1::<Complex<T>>::zeros(len_mm);
    for (j, k) in (-(mm as i64)..=(mm as i64)).enumerate() {
      c_dv2[j] = Complex::<T>::new(T::zero(), T::from_f64_fast(k as f64) * const_) * c_v2[j];
    }

    let center_dv2 = mm;
    let len_l = 2 * big_l + 1;
    let mut c_w = Array1::<Complex<T>>::zeros(len_l);
    let scale = self.period / T::from_usize_(2 * big_m + 1);

    for (j_w, k) in (-(big_l as i64)..=(big_l as i64)).enumerate() {
      let mut sum = Complex::<T>::new(T::zero(), T::zero());
      for (j_dv, s) in (-(big_m as i64)..=(big_m as i64)).enumerate() {
        let idx = (center_dv2 as i64 + k - s) as usize;
        sum = sum + c_dv[j_dv] * c_dv2[idx];
      }
      c_w[j_w] = sum * scale;
    }

    Ok(fejer_inversion(
      &c_w,
      big_l,
      self.period,
      tau,
      T::from_usize_(big_l + 1),
    ))
  }

  /// Bias-corrected spot vol of vol, eq. (11) of Toscano et al.: the inner `M`-Fejér derivative
  /// product less its unweighted quarticity coefficient, before the outer `L`-Fejér inversion.
  pub fn spot_volvol_bias_corrected(
    &self,
    tau: &[T],
    m_freq: Option<usize>,
    l_freq: Option<usize>,
  ) -> Array1<T> {
    self
      .try_spot_volvol_bias_corrected(tau, m_freq, l_freq)
      .unwrap_or_else(|e| panic!("{e}"))
  }

  /// Fallible variant of [`Self::spot_volvol_bias_corrected`].
  pub fn try_spot_volvol_bias_corrected(
    &self,
    tau: &[T],
    m_freq: Option<usize>,
    l_freq: Option<usize>,
  ) -> Result<Array1<T>, QuantError> {
    let big_m = self.resolve_m_volvol_bc(m_freq);
    let big_l = self.resolve_l_volvol_bc(l_freq, big_m);
    if big_l > big_m.saturating_mul(2) {
      return Err(QuantError::invalid(
        "L",
        format!("equation (11) requires L <= 2M, got L = {big_l}, M = {big_m}"),
      ));
    }
    self.validate_tau(tau)?;
    self.validate_ml_window(big_m, big_l)?;
    let corrected = self.bias_corrected_volvol_coefficients(big_m, big_l);
    Ok(fejer_inversion(
      &corrected,
      big_l,
      self.period,
      tau,
      T::from_usize_(big_l + 1),
    ))
  }

  /// Spot quarticity at evaluation times `tau`.
  pub fn spot_quarticity(
    &self,
    tau: &[T],
    m_freq: Option<usize>,
    l_freq: Option<usize>,
  ) -> Array1<T> {
    self
      .try_spot_quarticity(tau, m_freq, l_freq)
      .unwrap_or_else(|e| panic!("{e}"))
  }

  /// Fallible variant of [`Self::spot_quarticity`].
  pub fn try_spot_quarticity(
    &self,
    tau: &[T],
    m_freq: Option<usize>,
    l_freq: Option<usize>,
  ) -> Result<Array1<T>, QuantError> {
    let big_m = self.resolve_m(m_freq);
    let big_l = l_freq.unwrap_or(((self.n_freq as f64).sqrt()).sqrt() as usize);
    self.validate_tau(tau)?;
    self.validate_ml_window(big_m, big_l)?;
    let mm = big_m + big_l;

    let c_v = self.vol_coeffs(big_m);

    let c_v2 = convolution_coefficients(&self.dx, &self.dx, self.period, self.n_freq, mm);
    let center_v2 = mm;
    let len_l = 2 * big_l + 1;
    let mut c_q = Array1::<Complex<T>>::zeros(len_l);

    for (j_q, k) in (-(big_l as i64)..=(big_l as i64)).enumerate() {
      let mut sum = Complex::<T>::new(T::zero(), T::zero());
      for (j_v, s) in (-(big_m as i64)..=(big_m as i64)).enumerate() {
        let idx = (center_v2 as i64 + k - s) as usize;
        sum = sum + c_v[j_v] * c_v2[idx];
      }
      c_q[j_q] = sum;
    }

    Ok(fejer_inversion(
      &c_q,
      big_l,
      self.period,
      tau,
      T::from_usize_(big_l + 1),
    ))
  }
}
