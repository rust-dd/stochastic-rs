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
    let big_m = self.resolve_m(m_freq);
    let c_v = self.vol_coeffs(big_m);
    fejer_inversion(&c_v, big_m, self.period, tau, T::from_usize_(big_m + 1))
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
  /// Panics if `m_freq.unwrap_or(default) < 2`.
  pub fn spot_variance_fe(&self, tau: &[T], m_freq: Option<usize>) -> Array1<T> {
    let big_m = m_freq.unwrap_or((self.n_freq as f64).sqrt() as usize);
    assert!(big_m >= 2, "FE kernel requires M >= 2, got {big_m}");
    let c_v = self.vol_coeffs(big_m);
    fejer_inversion(&c_v, big_m, self.period, tau, T::from_usize_(big_m))
  }

  /// Spot covariance with another process at evaluation times `tau`.
  pub fn spot_covariance(&self, other: &Self, tau: &[T], m_freq: Option<usize>) -> Array1<T> {
    let big_n = self.n_freq.min(other.n_freq);
    let big_m = m_freq.unwrap_or((big_n as f64).sqrt() as usize);
    let c_c = convolution_coefficients(&other.dx, &self.dx, self.period, big_n, big_m);
    fejer_inversion(&c_c, big_m, self.period, tau, T::from_usize_(big_m + 1))
  }

  /// Spot leverage at evaluation times `tau`.
  pub fn spot_leverage(
    &self,
    tau: &[T],
    m_freq: Option<usize>,
    l_freq: Option<usize>,
  ) -> Array1<T> {
    let big_m = self.resolve_m(m_freq);
    let big_l = l_freq.unwrap_or((self.n_freq as f64).powf(0.25) as usize);
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

    fejer_inversion(&c_lev, big_l, self.period, tau, T::from_usize_(big_l + 1))
  }

  /// Spot volatility of volatility at evaluation times `tau`.
  pub fn spot_volvol(&self, tau: &[T], m_freq: Option<usize>, l_freq: Option<usize>) -> Array1<T> {
    let big_m = self.resolve_m_volvol(m_freq);
    let big_l = l_freq.unwrap_or((self.n_freq as f64).powf(0.2) as usize);
    let const_ = self.const_();
    let mm = big_m + big_l;

    let c_v = self.vol_coeffs(big_m);
    let len_m = 2 * big_m + 1;
    let mut c_dv = Array1::<Complex<T>>::zeros(len_m);
    for (j, k) in (-(big_m as i64)..=(big_m as i64)).enumerate() {
      c_dv[j] = Complex::<T>::new(T::zero(), T::from_f64_fast(k as f64) * const_) * c_v[j];
    }

    assert!(
      self.n_freq + mm <= self.max_freq,
      "need max_freq ≥ N+M+L = {} but have {}",
      self.n_freq + mm,
      self.max_freq
    );
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

    fejer_inversion(&c_w, big_l, self.period, tau, T::from_usize_(big_l + 1))
  }

  /// Bias-corrected spot volatility-of-volatility (analogous to `integrated_volvol_bias_corrected`).
  /// Subtracts `K · spot_quarticity(τ)` per evaluation point, where `K = M²/(3n)` per
  /// Toscano-Livieri-Mancino-Marmi (2024) eq.51 under uniform sampling at the default `N = n/2`.
  ///
  /// Uses `M ≈ N^{0.25}` default for the volvol window (matching `integrated_volvol_bias_corrected`),
  /// distinct from the legacy `spot_volvol`'s `N^{0.2}` window. The smaller window is needed for the
  /// bias correction's rate-`n^{1/4}` convergence regime.
  pub fn spot_volvol_bias_corrected(
    &self,
    tau: &[T],
    m_freq: Option<usize>,
    l_freq: Option<usize>,
  ) -> Array1<T> {
    let big_m = self.resolve_m_volvol_bc(m_freq);
    let raw = self.spot_volvol(tau, Some(big_m), l_freq);
    let quart = self.spot_quarticity(tau, Some(big_m), l_freq);
    let k_const: T = self.compute_bias_correction_constant(big_m);
    let mut out = Array1::<T>::zeros(tau.len());
    for i in 0..tau.len() {
      out[i] = raw[i] - k_const * quart[i];
    }
    out
  }

  /// Spot quarticity at evaluation times `tau`.
  pub fn spot_quarticity(
    &self,
    tau: &[T],
    m_freq: Option<usize>,
    l_freq: Option<usize>,
  ) -> Array1<T> {
    let big_m = self.resolve_m(m_freq);
    let big_l = l_freq.unwrap_or(((self.n_freq as f64).sqrt()).sqrt() as usize);
    let mm = big_m + big_l;

    let c_v = self.vol_coeffs(big_m);

    assert!(
      self.n_freq + mm <= self.max_freq,
      "need max_freq ≥ N+M+L = {} but have {}",
      self.n_freq + mm,
      self.max_freq
    );
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

    fejer_inversion(&c_q, big_l, self.period, tau, T::from_usize_(big_l + 1))
  }
}
