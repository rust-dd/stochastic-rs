//! Spot volatility / covariance / leverage / vol-of-vol / quarticity estimators.

use ndarray::Array1;
use num_complex::Complex;

use super::FMVol;
use super::helpers::fejer_inversion;
use crate::fourier_malliavin::coefficients::convolution_coefficients;
use crate::traits::FloatExt;

impl<T: FloatExt> FMVol<T> {
  /// Spot variance at evaluation times `tau`.
  pub fn spot_variance(&self, tau: &[T], m_freq: Option<usize>) -> Array1<T> {
    self
      .try_spot_variance(tau, m_freq)
      .expect("invalid spot variance window or evaluation time")
  }

  /// Fallible variant of [`Self::spot_variance`].
  pub fn try_spot_variance(&self, tau: &[T], m_freq: Option<usize>) -> anyhow::Result<Array1<T>> {
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

  /// Spot variance under the **FE** (Fourier-Estimator) Cesàro-kernel
  /// convention: weight `1 − |k|/M`, i.e. effective bandwidth `M − 1`,
  /// instead of the FM-convention weight `1 − |k|/(M+1)`.
  ///
  /// Matches the MATLAB FSDA `FE_spot_vol` / `FE_spot_vol_FFT` output
  /// element-for-element. Use [`super::super::default_cutting_freq_fe`] for the
  /// FE-style `(N, M)` defaults and pair with
  /// [`FMVol::with_freq`] so that `max_freq ≥ N + M`.
  ///
  /// Panics if the window is invalid. Use [`Self::try_spot_variance_fe`] to
  /// handle invalid input without panicking.
  pub fn spot_variance_fe(&self, tau: &[T], m_freq: Option<usize>) -> Array1<T> {
    self
      .try_spot_variance_fe(tau, m_freq)
      .expect("invalid FE spot variance window or evaluation time")
  }

  /// Fallible variant of [`Self::spot_variance_fe`].
  pub fn try_spot_variance_fe(
    &self,
    tau: &[T],
    m_freq: Option<usize>,
  ) -> anyhow::Result<Array1<T>> {
    let big_m = m_freq.unwrap_or((self.n_freq as f64).sqrt() as usize);
    if big_m < 2 {
      anyhow::bail!("FE kernel requires M >= 2, got {big_m}");
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
      .expect("invalid spot covariance configuration")
  }

  /// Fallible variant of [`Self::spot_covariance`].
  pub fn try_spot_covariance(
    &self,
    other: &Self,
    tau: &[T],
    m_freq: Option<usize>,
  ) -> anyhow::Result<Array1<T>> {
    self.validate_compatible_period(other)?;
    self.validate_tau(tau)?;
    let big_n = self.n_freq.min(other.n_freq);
    let big_m = m_freq.unwrap_or((big_n as f64).sqrt() as usize);
    if big_m == 0 {
      anyhow::bail!("M must be positive");
    }
    let required = big_n
      .checked_add(big_m)
      .ok_or_else(|| anyhow::anyhow!("N + M overflows usize"))?;
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
      .expect("invalid spot leverage window or evaluation time")
  }

  /// Fallible variant of [`Self::spot_leverage`].
  pub fn try_spot_leverage(
    &self,
    tau: &[T],
    m_freq: Option<usize>,
    l_freq: Option<usize>,
  ) -> anyhow::Result<Array1<T>> {
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
      .expect("invalid spot volatility-of-volatility window or evaluation time")
  }

  /// Fallible variant of [`Self::spot_volvol`].
  pub fn try_spot_volvol(
    &self,
    tau: &[T],
    m_freq: Option<usize>,
    l_freq: Option<usize>,
  ) -> anyhow::Result<Array1<T>> {
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

  /// Bias-corrected spot volatility of volatility.
  ///
  /// Implements equation (11) of Toscano et al.: every Fourier coefficient
  /// uses the inner `M`-Fejér derivative product and its matching unweighted
  /// quarticity coefficient before the outer `L`-Fejér inversion.
  pub fn spot_volvol_bias_corrected(
    &self,
    tau: &[T],
    m_freq: Option<usize>,
    l_freq: Option<usize>,
  ) -> Array1<T> {
    self
      .try_spot_volvol_bias_corrected(tau, m_freq, l_freq)
      .expect("invalid bias-corrected spot volatility-of-volatility window or evaluation time")
  }

  /// Fallible variant of [`Self::spot_volvol_bias_corrected`].
  pub fn try_spot_volvol_bias_corrected(
    &self,
    tau: &[T],
    m_freq: Option<usize>,
    l_freq: Option<usize>,
  ) -> anyhow::Result<Array1<T>> {
    let big_m = self.resolve_m_volvol_bc(m_freq);
    let big_l = self.resolve_l_volvol_bc(l_freq, big_m);
    if big_l > big_m.saturating_mul(2) {
      anyhow::bail!("equation (11) requires L <= 2M");
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
      .expect("invalid spot quarticity window or evaluation time")
  }

  /// Fallible variant of [`Self::spot_quarticity`].
  pub fn try_spot_quarticity(
    &self,
    tau: &[T],
    m_freq: Option<usize>,
    l_freq: Option<usize>,
  ) -> anyhow::Result<Array1<T>> {
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
