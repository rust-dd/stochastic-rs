//! Struct-based Fourier-Malliavin volatility estimation engine.
//!
//! See Sanfelici & Toscano (2024), arXiv:2402.00172.

use ndarray::Array1;
use num_complex::Complex;

use super::coefficients::convolution_coefficients;
use super::coefficients::fourier_coefficients_dx;
use super::coefficients::fourier_coefficients_dx_uniform;
use crate::traits::FloatExt;

/// Fourier-Malliavin volatility estimation engine.
///
/// Pre-computes the Fourier coefficients of price increments once, then
/// exposes integrated and spot estimators as cheap method calls.
///
/// Generic over `T: FloatExt` (`f32` / `f64`).
///
/// # Example
/// ```ignore
/// let engine = FMVol::new(&log_prices, &times, 1.0);
/// let iv   = engine.integrated_variance();
/// let spot = engine.spot_variance(&tau, None);
/// ```
pub struct FMVol<T: FloatExt> {
  /// Precomputed Fourier coefficients of price increments.
  dx: Array1<Complex<T>>,
  /// Time period *T*.
  period: T,
  /// Number of price increments (*n*).
  n: usize,
  /// Primary cutting frequency *N*.
  n_freq: usize,
  /// Maximum frequency stored in `dx`.
  max_freq: usize,
}

impl<T: FloatExt> FMVol<T> {
  /// Build an engine from irregularly spaced observations.
  ///
  /// Sets `N = floor(n/2)` and pre-computes Fourier coefficients up to
  /// `N + M_max + L_max` where `M_max = floor(N^0.5)` and `L_max = floor(N^0.25)`.
  ///
  /// Panics if `prices.len() < 2` or `times.len() != prices.len()`.
  pub fn new(prices: &[T], times: &[T], period: T) -> Self {
    assert!(
      prices.len() >= 2,
      "FMVol::new requires at least 2 price observations to form increments, got {}",
      prices.len()
    );
    assert_eq!(
      prices.len(),
      times.len(),
      "FMVol::new: prices.len()={} must equal times.len()={}",
      prices.len(),
      times.len()
    );
    let n = prices.len() - 1;
    let big_n = n / 2;
    let m_max = (big_n as f64).sqrt() as usize;
    let l_max = (big_n as f64).powf(0.25) as usize;
    let max_freq = big_n + m_max + l_max;
    let dx = fourier_coefficients_dx(prices, times, period, max_freq);
    Self {
      dx,
      period,
      n,
      n_freq: big_n,
      max_freq,
    }
  }

  /// Build an engine from **uniformly spaced** observations (FFT-accelerated, O(n log n)).
  ///
  /// Assumes `t_l = l · T / n`; no explicit times array needed.
  ///
  /// Panics if `prices.len() < 2`.
  pub fn new_uniform(prices: &[T], period: T) -> Self {
    assert!(
      prices.len() >= 2,
      "FMVol::new_uniform requires at least 2 price observations to form increments, got {}",
      prices.len()
    );
    let n = prices.len() - 1;
    let big_n = n / 2;
    let m_max = (big_n as f64).sqrt() as usize;
    let l_max = (big_n as f64).powf(0.25) as usize;
    let max_freq = big_n + m_max + l_max;
    let dx = fourier_coefficients_dx_uniform(prices, period, max_freq);
    Self {
      dx,
      period,
      n,
      n_freq: big_n,
      max_freq,
    }
  }

  /// Build an engine with explicit cutting frequency *N* and maximum frequency.
  ///
  /// `max_freq` controls how high the Fourier coefficients are computed.
  /// Must satisfy `max_freq ≥ n_freq`.
  /// For spot leverage / volvol / quarticity you need `max_freq ≥ N + M + L`.
  ///
  /// Panics if `prices.len() < 2`, `times.len() != prices.len()`, or `max_freq < n_freq`.
  pub fn with_freq(prices: &[T], times: &[T], period: T, n_freq: usize, max_freq: usize) -> Self {
    assert!(
      prices.len() >= 2,
      "FMVol::with_freq requires at least 2 price observations to form increments, got {}",
      prices.len()
    );
    assert_eq!(
      prices.len(),
      times.len(),
      "FMVol::with_freq: prices.len()={} must equal times.len()={}",
      prices.len(),
      times.len()
    );
    let n = prices.len() - 1;
    assert!(max_freq >= n_freq, "max_freq={max_freq} must be ≥ n_freq={n_freq}");
    let dx = fourier_coefficients_dx(prices, times, period, max_freq);
    Self {
      dx,
      period,
      n,
      n_freq,
      max_freq,
    }
  }

  /// Primary cutting frequency *N*.
  pub fn n_freq(&self) -> usize {
    self.n_freq
  }

  /// Number of price increments.
  pub fn n(&self) -> usize {
    self.n
  }

  /// Time period.
  pub fn period(&self) -> T {
    self.period
  }

  fn center(&self) -> usize {
    self.max_freq
  }

  fn resolve_m(&self, m: Option<usize>) -> usize {
    m.unwrap_or((self.n_freq as f64).sqrt() as usize)
  }

  fn resolve_m_volvol(&self, m: Option<usize>) -> usize {
    m.unwrap_or((self.n_freq as f64).powf(0.4) as usize)
  }

  fn const_(&self) -> T {
    T::from_f64_fast(std::f64::consts::TAU) / self.period
  }

  fn vol_coeffs(&self, m: usize) -> Array1<Complex<T>> {
    assert!(
      self.n_freq + m <= self.max_freq,
      "need max_freq ≥ N + M = {} but have {}",
      self.n_freq + m,
      self.max_freq
    );
    convolution_coefficients(&self.dx, &self.dx, self.period, self.n_freq, m)
  }

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

  /// Spot variance at evaluation times `tau`.
  pub fn spot_variance(&self, tau: &[T], m_freq: Option<usize>) -> Array1<T> {
    let big_m = self.resolve_m(m_freq);
    let c_v = self.vol_coeffs(big_m);
    fejer_inversion(&c_v, big_m, self.period, tau, T::from_usize_(big_m + 1))
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

/// Fejér kernel inversion (internal helper).
fn fejer_inversion<T: FloatExt>(
  coeffs: &Array1<Complex<T>>,
  m_freq: usize,
  period: T,
  tau: &[T],
  fejer_denom: T,
) -> Array1<T> {
  let const_ = T::from_f64_fast(std::f64::consts::TAU) / period;
  let mut result = Array1::<T>::zeros(tau.len());
  for (i, &t) in tau.iter().enumerate() {
    let mut sum = Complex::<T>::new(T::zero(), T::zero());
    for (j, k) in (-(m_freq as i64)..=(m_freq as i64)).enumerate() {
      let k_t = T::from_f64_fast(k as f64);
      let fejer = T::one() - T::from_usize_(k.unsigned_abs() as usize) / fejer_denom;
      let phase = const_ * k_t * t;
      sum = sum + coeffs[j] * Complex::new(phase.cos(), phase.sin()) * fejer;
    }
    result[i] = sum.re;
  }
  result
}

#[cfg(test)]
mod tests {
  use stochastic_rs_stochastic::volatility::HestonPow;
  use stochastic_rs_stochastic::volatility::heston::Heston;

  use super::*;
  use crate::traits::ProcessExt;

  fn heston_paths() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = 23401_usize;
    let t = 1.0_f64;
    let heston = Heston::seeded(
      Some(100.0),
      Some(0.4),
      2.0,
      0.4,
      1.0,
      -0.5,
      0.0,
      n,
      Some(t),
      HestonPow::Sqrt,
      Some(false),
      42,
    );
    let [s, v] = heston.sample();
    let dt = t / (n as f64 - 1.0);
    let times: Vec<f64> = (0..n).map(|i| i as f64 * dt).collect();
    let log_prices: Vec<f64> = s.iter().map(|&si| si.ln()).collect();
    (log_prices, v.to_vec(), times)
  }

  fn true_integrated_variance(v: &[f64], dt: f64) -> f64 {
    (0..v.len() - 1).map(|i| (v[i] + v[i + 1]) * 0.5 * dt).sum()
  }

  #[test]
  fn test_integrated_variance() {
    let (lp, v, times) = heston_paths();
    let dt = 1.0 / (lp.len() - 1) as f64;
    let engine = FMVol::new(&lp, &times, 1.0);

    let true_iv = true_integrated_variance(&v, dt);
    let est_iv = engine.integrated_variance();
    let rel_err = (est_iv - true_iv).abs() / true_iv;
    assert!(
      rel_err < 0.15,
      "est={est_iv:.6}, true={true_iv:.6}, rel_err={rel_err:.4}"
    );
  }

  #[test]
  fn test_integrated_variance_f32() {
    let (lp64, v, times64) = heston_paths();
    let dt = 1.0 / (lp64.len() - 1) as f64;
    let lp: Vec<f32> = lp64.iter().map(|&x| x as f32).collect();
    let times: Vec<f32> = times64.iter().map(|&x| x as f32).collect();

    let engine = FMVol::new(&lp, &times, 1.0_f32);
    let true_iv = true_integrated_variance(&v, dt);
    let est_iv = engine.integrated_variance() as f64;
    let rel_err = (est_iv - true_iv).abs() / true_iv;
    assert!(
      rel_err < 0.15,
      "f32 est={est_iv:.6}, true={true_iv:.6}, rel_err={rel_err:.4}"
    );
  }

  #[test]
  fn test_uniform_fft_matches_direct() {
    let (lp, _, times) = heston_paths();
    let engine_direct = FMVol::new(&lp, &times, 1.0);
    let engine_fft = FMVol::new_uniform(&lp, 1.0);

    let iv_direct = engine_direct.integrated_variance();
    let iv_fft = engine_fft.integrated_variance();
    let rel_err = (iv_fft - iv_direct).abs() / iv_direct.abs();
    assert!(
      rel_err < 1e-6,
      "FFT vs direct mismatch: fft={iv_fft:.8}, direct={iv_direct:.8}, rel_err={rel_err:.2e}"
    );
  }

  #[test]
  fn test_uniform_fft_spot_matches_direct() {
    let (lp, _, times) = heston_paths();
    let engine_direct = FMVol::new(&lp, &times, 1.0);
    let engine_fft = FMVol::new_uniform(&lp, 1.0);
    let tau: Vec<f64> = (0..11).map(|i| i as f64 / 10.0).collect();

    let spot_direct = engine_direct.spot_variance(&tau, None);
    let spot_fft = engine_fft.spot_variance(&tau, None);

    let max_diff = spot_direct
      .iter()
      .zip(spot_fft.iter())
      .map(|(a, b)| (a - b).abs())
      .fold(0.0_f64, f64::max);
    assert!(
      max_diff < 1e-6,
      "FFT vs direct spot max_diff = {max_diff:.2e}"
    );
  }

  #[test]
  fn test_covariance_self_equals_variance() {
    let (lp, _, times) = heston_paths();
    let engine = FMVol::new(&lp, &times, 1.0);
    let iv = engine.integrated_variance();
    let icov = engine.integrated_covariance(&engine);
    let rel_err = (icov - iv).abs() / iv;
    assert!(
      rel_err < 0.05,
      "cov(x,x)={icov:.6} ≠ var(x)={iv:.6}, rel_err={rel_err:.4}"
    );
  }

  #[test]
  fn test_integrated_leverage_negative() {
    let (lp, _, times) = heston_paths();
    let engine = FMVol::new(&lp, &times, 1.0);
    let est = engine.integrated_leverage(None);
    assert!(est < 0.0, "leverage should be < 0 for ρ<0, got {est}");
  }

  #[test]
  fn test_integrated_volvol_positive() {
    let (lp, _, times) = heston_paths();
    let engine = FMVol::new(&lp, &times, 1.0);
    let est = engine.integrated_volvol(None);
    assert!(est > 0.0, "vol-of-vol should be > 0, got {est}");
  }

  #[test]
  fn test_integrated_quarticity_positive() {
    let (lp, _, times) = heston_paths();
    let engine = FMVol::new(&lp, &times, 1.0);
    let est = engine.integrated_quarticity(None);
    assert!(est > 0.0, "quarticity should be > 0, got {est}");
  }

  #[test]
  fn test_spot_variance_vs_true() {
    let (lp, v, times) = heston_paths();
    let engine = FMVol::new(&lp, &times, 1.0);
    let n_tau = 21;
    let tau: Vec<f64> = (0..n_tau).map(|i| i as f64 / (n_tau - 1) as f64).collect();
    let spot = engine.spot_variance(&tau, None);

    let step = (lp.len() - 1) / (n_tau - 1);
    let mae: f64 = (0..n_tau)
      .map(|i| (spot[i] - v[i * step]).abs())
      .sum::<f64>()
      / n_tau as f64;
    assert!(mae < 0.25, "spot vol MAE = {mae:.4} too large");
  }

  #[test]
  fn test_spot_covariance_self_equals_spot_vol() {
    let (lp, _, times) = heston_paths();
    let engine = FMVol::new(&lp, &times, 1.0);
    let tau: Vec<f64> = (0..11).map(|i| i as f64 / 10.0).collect();

    let sv = engine.spot_variance(&tau, None);
    let sc = engine.spot_covariance(&engine, &tau, None);

    let max_diff = sv
      .iter()
      .zip(sc.iter())
      .map(|(a, b)| (a - b).abs())
      .fold(0.0_f64, f64::max);
    assert!(max_diff < 0.05, "max_diff = {max_diff:.6}");
  }

  #[test]
  fn test_spot_leverage_negative() {
    let (lp, _, times) = heston_paths();
    let engine = FMVol::new(&lp, &times, 1.0);
    let tau: Vec<f64> = (0..11).map(|i| i as f64 / 10.0).collect();
    let spot = engine.spot_leverage(&tau, None, None);
    let mean: f64 = spot.iter().copied().sum::<f64>() / spot.len() as f64;
    assert!(mean < 0.0, "mean spot leverage should be < 0, got {mean}");
  }

  #[test]
  fn test_spot_volvol_positive() {
    let (lp, _, times) = heston_paths();
    let engine = FMVol::new(&lp, &times, 1.0);
    let tau: Vec<f64> = (0..11).map(|i| i as f64 / 10.0).collect();
    let spot = engine.spot_volvol(&tau, None, None);
    let mean: f64 = spot.iter().copied().sum::<f64>() / spot.len() as f64;
    assert!(mean > 0.0, "mean spot volvol should be > 0, got {mean}");
  }

  #[test]
  fn test_spot_quarticity_positive() {
    let (lp, _, times) = heston_paths();
    let engine = FMVol::new(&lp, &times, 1.0);
    let tau: Vec<f64> = (0..11).map(|i| i as f64 / 10.0).collect();
    let spot = engine.spot_quarticity(&tau, None, None);
    let mean: f64 = spot.iter().copied().sum::<f64>() / spot.len() as f64;
    assert!(mean > 0.0, "mean spot quarticity should be > 0, got {mean}");
  }

  #[test]
  fn test_spot_variance_f32() {
    let (lp64, _, times64) = heston_paths();
    let lp: Vec<f32> = lp64.iter().map(|&x| x as f32).collect();
    let times: Vec<f32> = times64.iter().map(|&x| x as f32).collect();
    let tau: Vec<f32> = (0..11).map(|i| i as f32 / 10.0).collect();

    let engine = FMVol::new(&lp, &times, 1.0_f32);
    let spot = engine.spot_variance(&tau, None);
    let mean: f32 = spot.iter().copied().sum::<f32>() / spot.len() as f32;
    assert!(
      mean > 0.1 && mean < 0.8,
      "f32 mean spot vol {mean} out of range"
    );
  }

  #[test]
  fn test_optimal_cutting_frequency_noisy() {
    let (lp, v, times) = heston_paths();
    let dt = 1.0 / (lp.len() - 1) as f64;
    let true_iv: f64 = (0..v.len() - 1).map(|i| (v[i] + v[i + 1]) * 0.5 * dt).sum();

    // Add i.i.d. noise: η ~ N(0, σ²_η) with noise-to-signal ≈ 0.5
    let sigma_eta = 0.005;
    let noisy: Vec<f64> = lp
      .iter()
      .enumerate()
      .map(|(i, &p)| {
        // Deterministic pseudo-noise for reproducibility
        let noise = sigma_eta * (((i * 7919 + 104729) % 10000) as f64 / 5000.0 - 1.0);
        p + noise
      })
      .collect();

    // Optimal N
    let result = super::super::optimal_cutting_frequency(&noisy, &times);
    let (n_opt, m_opt, _l_opt) = result.cutting_freqs();

    // Fixed-rule N (heuristic)
    let n = lp.len() - 1;
    let (n_heur, m_heur, _) = super::super::default_cutting_freq_noisy(n);

    // Estimate with optimal N
    let engine_opt = FMVol::with_freq(&noisy, &times, 1.0, n_opt, n_opt + m_opt + 10);
    let iv_opt = engine_opt.integrated_variance();

    // Estimate with heuristic N
    let engine_heur = FMVol::with_freq(&noisy, &times, 1.0, n_heur, n_heur + m_heur + 10);
    let iv_heur = engine_heur.integrated_variance();

    // Estimate with naive N = n/2 (no noise correction)
    let engine_naive = FMVol::new(&noisy, &times, 1.0);
    let iv_naive = engine_naive.integrated_variance();

    let err_opt = (iv_opt - true_iv).abs() / true_iv;
    let _err_heur = (iv_heur - true_iv).abs() / true_iv;
    let err_naive = (iv_naive - true_iv).abs() / true_iv;

    // Optimal N should give smaller error than naive (no noise correction)
    assert!(
      err_opt < err_naive,
      "optimal N should beat naive: err_opt={err_opt:.4}, err_naive={err_naive:.4}"
    );

    // Optimal N should be much smaller than n/2
    assert!(
      n_opt < n / 4,
      "optimal N={n_opt} should be << n/2={} for noisy data",
      n / 2
    );

    // Estimated noise variance should be in reasonable range
    assert!(
      result.noise_variance > 0.0,
      "noise variance should be positive, got {}",
      result.noise_variance
    );
  }
}
