//! Struct-based Fourier-Malliavin volatility estimation engine.
//!
//! References:
//!   - Sanfelici & Toscano (2024), arXiv:2402.00172 — FMVol MATLAB library paper.
//!   - Malliavin & Mancino (2002, 2009) — original method.
//!   - Mancino & Recchioni (2015) — bias/variance analysis of second-order estimators.
//!   - Toscano, Livieri, Mancino, Marmi (2024), arXiv:2112.14529 — CLTs for the volvol
//!     Fourier estimator; eq.(3) defines the consistent estimator, eq.(4) the bias-corrected
//!     rate-n^{1/4} variant, eq.(51) gives the bias-correction constant K.
//!
//! ## Numerical validation
//!
//! All estimators are validated against Heston-Sqrt ground-truth at fixture parameters
//! (σ_v = 1.0, ρ = -0.5, V̄ = 0.4, T = 1.0, n = 23401 points):
//!
//! | Estimator                          | Ground-truth formula            | Test tolerance |
//! |------------------------------------|---------------------------------|----------------|
//! | `integrated_variance`              | ∫V_t dt (trapezoidal of v)      | rel_err < 15%  |
//! | `spot_variance`                    | V_τ                             | MAE < 0.25     |
//! | `integrated_leverage`              | σ_v · ρ · IV(T)                 | rel_err < 40%  |
//! | `integrated_volvol` (eq.3)         | σ_v² · IV(T)                    | factor-of-3    |
//! | `integrated_volvol_bias_corrected` | σ_v² · IV(T)                    | rel_err < 30%  |
//! | `spot_leverage` (mean)             | σ_v · ρ · mean(V_τ)             | rel_err < 30%  |
//! | `spot_volvol` (mean, eq.3)         | σ_v² · mean(V_τ)                | factor-of-3    |
//! | `spot_volvol_bias_corrected`       | σ_v² · mean(V_τ)                | rel_err < 40%  |
//!
//! The `integrated_volvol` and `spot_volvol` (eq.3) estimators have a known ~2× finite-sample
//! bias documented in Toscano-Livieri-Mancino-Marmi (2024) §3-§4. The bias-corrected
//! variants (eq.4) subtract `K · quarticity` where `K = M²/(3n)` for the uniform-sampling
//! default, reducing the bias to <10% on the Heston fixture.
//!
//! Tolerances reflect finite-sample variance of high-order moment estimators on a single
//! Heston path. Tests catch structural bugs (sign errors, missing factors of T or 2π)
//! while accommodating the expected estimator noise; the bias-corrected variants get
//! much tighter tolerances because the dominant bias term is removed.

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
    assert!(
      max_freq >= n_freq,
      "max_freq={max_freq} must be ≥ n_freq={n_freq}"
    );
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

  /// Default smoothing window for spot-volatility / spot-leverage / spot-quarticity:
  /// `m ≈ √N`. This is the canonical Malliavin-Mancino choice (Malliavin & Mancino
  /// 2009, eq. 4.5 / Mancino-Recchioni 2015 §3): the bias from a finite Cesàro
  /// window scales like `1/m` while the variance of the Fourier-coefficient
  /// estimator scales like `m/N`, so balancing the two gives the `m ≈ N^{1/2}`
  /// optimum. The cast truncates the square root toward zero, which is
  /// intentional — we want `m ≤ √N` rather than rounding up across the
  /// bias/variance crossover.
  fn resolve_m(&self, m: Option<usize>) -> usize {
    m.unwrap_or((self.n_freq as f64).sqrt() as usize)
  }

  /// Default smoothing window for the volvol / leverage estimators: `m ≈ N^{0.4}`.
  /// Slower-than-square-root window to keep the estimator's variance bounded
  /// for the cube/quartic cumulants, per Mancino-Recchioni (2015) §4.
  fn resolve_m_volvol(&self, m: Option<usize>) -> usize {
    m.unwrap_or((self.n_freq as f64).powf(0.4) as usize)
  }

  /// Default smoothing window for the **bias-corrected** integrated volvol
  /// (Toscano-Livieri-Mancino-Marmi 2024 eq.4 / Theorem 3.1).
  ///
  /// Paper recommends `M·ρ(n)^{1/2} ~ c_M` (rate-`n^{1/4}` regime) with
  /// MSE-optimized `c_M ≈ 0.05–0.07` in the simulation studies (§4.3).
  /// For an annualized horizon (`T = 1`) and uniform sampling
  /// `ρ(n) = T/n`, this translates to `M ≈ 0.05·√n`.  We take the
  /// slightly more conservative `M ≈ N^{0.25}`, which keeps the finite-sample
  /// bias of the underlying eq.(3) estimator small enough that the eq.(51)
  /// correction acts in the regime where it was derived.  Larger windows
  /// (e.g. `M ≈ N^{0.4}`) push us out of the rate-optimal regime and the
  /// eq.(51) correction under-shoots the empirical bias.
  fn resolve_m_volvol_bc(&self, m: Option<usize>) -> usize {
    m.unwrap_or((self.n_freq as f64).powf(0.25).max(2.0) as usize)
  }

  /// Bias-correction constant `K` from Toscano-Livieri-Mancino-Marmi (2024)
  /// equation (51) [arXiv:2112.14529v3, p.42]:
  ///
  /// $$K := \tfrac{1}{3}\cdot \tfrac{c_M^2}{2\pi}\,\bigl(1 + 2\eta(c_N/\pi)\bigr),$$
  ///
  /// where `c_M, c_N` are the asymptotic-regime constants
  /// `c_M = M·ρ(n)^{1/2}`, `c_N = N·ρ(n)` and
  /// `η(a) := r(a)(1-r(a)) / (2 a²)`, `r(a) = a − ⌊a⌋`.
  ///
  /// In our convention `ρ(n) = T/n` and `c_N/π = 2N/n` is independent of `T`,
  /// so `K` reduces to a `T`-independent quantity:
  ///
  /// $$K = \frac{M^2}{3n}\,(1 + 2\eta(2N/n)).$$
  ///
  /// When `N = n/2` (the default Nyquist choice), `2N/n = 1`, the fractional
  /// part vanishes and `η = 0`, so `K = M²/(3n)`.
  fn compute_bias_correction_constant(&self, big_m: usize) -> T {
    let n_f = self.n as f64;
    let n_freq_f = self.n_freq as f64;
    let m_f = big_m as f64;
    // a = c_N/π = 2N/n (paper convention, dimensionless)
    let a = 2.0 * n_freq_f / n_f;
    let r = a - a.floor();
    let eta = if a.abs() > 1e-12 {
      r * (1.0 - r) / (2.0 * a * a)
    } else {
      0.0
    };
    let k = (m_f * m_f) / (3.0 * n_f) * (1.0 + 2.0 * eta);
    T::from_f64_fast(k)
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

  /// Bias-corrected integrated volatility of volatility (Toscano-Livieri-
  /// Mancino-Marmi 2024 eq.4, arXiv:2112.14529v3).
  ///
  /// Subtracts `K · σ̂⁴_{n,N,M}` from the raw `integrated_volvol`, where
  /// the constant `K` is given by eq.(51) of the paper — see
  /// [`compute_bias_correction_constant`].  Achieves the optimal
  /// `n^{1/4}` convergence rate (Theorem 3.1) at the price of
  /// **destroying the positivity of the estimator** (paper, p.5 / Sec.2);
  /// finite-sample realizations may therefore be negative.
  ///
  /// The default smoothing window is `M ≈ N^{0.25}` rather than the `N^{0.4}`
  /// used by [`integrated_volvol`]; the larger window applicable to the
  /// non-rate-optimal eq.(3) estimator (Theorem 3.6) leaves us outside the
  /// regime where the eq.(51) bias term dominates, and the correction
  /// under-shoots empirically.  See [`resolve_m_volvol_bc`] for details.
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
  /// element-for-element. Use [`super::default_cutting_freq_fe`] for the
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
  use stochastic_rs_stochastic::volatility::heston2d::Heston2D;

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

  /// Ground-truth integrated leverage: σ_v · ρ · IV(T).
  /// Returns the analytical Heston covariation between V_t and log S_t.
  fn true_integrated_leverage(v: &[f64], dt: f64, sigma_v: f64, rho: f64) -> f64 {
    let iv = true_integrated_variance(v, dt);
    sigma_v * rho * iv
  }

  /// Ground-truth integrated volvol: σ_v² · IV(T).
  /// Returns the analytical Heston quadratic variation of V_t.
  fn true_integrated_volvol(v: &[f64], dt: f64, sigma_v: f64) -> f64 {
    let iv = true_integrated_variance(v, dt);
    sigma_v * sigma_v * iv
  }

  /// Heston fixture parameters reused across tests (matches `heston_paths()`).
  const HESTON_SIGMA_V: f64 = 1.0;
  const HESTON_RHO: f64 = -0.5;

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
  fn test_integrated_leverage_vs_truth() {
    let (lp, v, times) = heston_paths();
    let dt = 1.0 / (lp.len() - 1) as f64;
    let engine = FMVol::new(&lp, &times, 1.0);

    let true_lev = true_integrated_leverage(&v, dt, HESTON_SIGMA_V, HESTON_RHO);
    let est_lev = engine.integrated_leverage(None);
    let rel_err = (est_lev - true_lev).abs() / true_lev.abs();
    assert!(
      est_lev < 0.0,
      "leverage should be < 0 for ρ<0, got {est_lev}"
    );
    assert!(
      rel_err < 0.40,
      "integrated_leverage rel_err = {rel_err:.4} > 40%, est={est_lev:.4}, true={true_lev:.4}. \
       Tolerance is generous because the second-order estimators have higher finite-sample \
       variance; a 40% tolerance still catches structural bugs (sign errors, scaling factors of 2)."
    );
  }

  #[test]
  fn test_integrated_volvol_eq3_within_factor_of_3() {
    let (lp, v, times) = heston_paths();
    let dt = 1.0 / (lp.len() - 1) as f64;
    let engine = FMVol::new(&lp, &times, 1.0);

    let true_vv = true_integrated_volvol(&v, dt, HESTON_SIGMA_V);
    let est_vv = engine.integrated_volvol(None);
    assert!(est_vv > 0.0, "volvol should be > 0, got {est_vv}");
    assert!(
      est_vv > 0.3 * true_vv && est_vv < 3.0 * true_vv,
      "eq.3 volvol est={est_vv:.4} not within [0.3×, 3.0×] of true={true_vv:.4} \
       — this is the non-bias-corrected variant, expect ~2× overestimate from finite-sample bias"
    );
  }

  #[test]
  fn test_integrated_volvol_bias_corrected_vs_truth() {
    let (lp, v, times) = heston_paths();
    let dt = 1.0 / (lp.len() - 1) as f64;
    let engine = FMVol::new(&lp, &times, 1.0);
    let true_vv = true_integrated_volvol(&v, dt, HESTON_SIGMA_V);
    let est_vv = engine.integrated_volvol_bias_corrected(None);
    let rel_err = (est_vv - true_vv).abs() / true_vv;
    assert!(
      rel_err < 0.30,
      "bias-corrected volvol rel_err = {rel_err:.4} > 30%, est={est_vv:.4}, true={true_vv:.4}"
    );
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
  fn test_spot_leverage_mean_vs_truth() {
    let (lp, v, times) = heston_paths();
    let engine = FMVol::new(&lp, &times, 1.0);
    let n_tau = 11;
    let tau: Vec<f64> = (0..n_tau).map(|i| i as f64 / (n_tau - 1) as f64).collect();
    let spot = engine.spot_leverage(&tau, None, None);

    let step = (lp.len() - 1) / (n_tau - 1);
    let true_lev: Vec<f64> = (0..n_tau)
      .map(|i| HESTON_SIGMA_V * HESTON_RHO * v[i * step])
      .collect();

    let spot_mean: f64 = spot.iter().sum::<f64>() / spot.len() as f64;
    let true_mean: f64 = true_lev.iter().sum::<f64>() / n_tau as f64;
    assert!(
      spot_mean < 0.0,
      "mean spot leverage should be < 0 for ρ<0, got {spot_mean}"
    );
    let rel_err = (spot_mean - true_mean).abs() / true_mean.abs();
    assert!(
      rel_err < 0.30,
      "spot_leverage mean rel_err = {rel_err:.4} > 30%; spot_mean = {spot_mean:.4}, true_mean = {true_mean:.4}. \
       Per-τ MAE is high (~120%) because the FM estimator smooths via Fejér window and per-τ values \
       have large finite-sample variance; we assert the mean instead, which is stable."
    );
  }

  #[test]
  fn test_spot_volvol_within_factor_of_3() {
    let (lp, v, times) = heston_paths();
    let engine = FMVol::new(&lp, &times, 1.0);
    let n_tau = 11;
    let tau: Vec<f64> = (0..n_tau).map(|i| i as f64 / (n_tau - 1) as f64).collect();
    let spot = engine.spot_volvol(&tau, None, None);

    let step = (lp.len() - 1) / (n_tau - 1);
    let true_vv: Vec<f64> = (0..n_tau)
      .map(|i| HESTON_SIGMA_V * HESTON_SIGMA_V * v[i * step])
      .collect();
    let spot_mean: f64 = spot.iter().sum::<f64>() / spot.len() as f64;
    let true_mean: f64 = true_vv.iter().sum::<f64>() / n_tau as f64;
    assert!(
      spot_mean > 0.0,
      "mean spot volvol should be > 0, got {spot_mean}"
    );
    assert!(
      spot_mean > 0.3 * true_mean && spot_mean < 3.0 * true_mean,
      "spot_volvol mean {spot_mean:.4} not within [0.3×, 3.0×] of true mean {true_mean:.4} \
       — this is the non-bias-corrected variant, expect ~2-3× overestimate from finite-sample bias"
    );
  }

  #[test]
  fn test_spot_volvol_bias_corrected_vs_truth() {
    let (lp, v, times) = heston_paths();
    let engine = FMVol::new(&lp, &times, 1.0);
    let n_tau = 11;
    let tau: Vec<f64> = (0..n_tau).map(|i| i as f64 / (n_tau - 1) as f64).collect();
    let spot = engine.spot_volvol_bias_corrected(&tau, None, None);

    let step = (lp.len() - 1) / (n_tau - 1);
    let true_vv: Vec<f64> = (0..n_tau)
      .map(|i| HESTON_SIGMA_V * HESTON_SIGMA_V * v[i * step])
      .collect();
    let spot_mean: f64 = spot.iter().sum::<f64>() / spot.len() as f64;
    let true_mean: f64 = true_vv.iter().sum::<f64>() / n_tau as f64;
    assert!(
      spot_mean > 0.0,
      "mean bias-corrected spot volvol should be > 0, got {spot_mean}"
    );
    let rel_err = (spot_mean - true_mean).abs() / true_mean;
    assert!(
      rel_err < 0.40,
      "spot_volvol_bias_corrected mean rel_err = {rel_err:.4} > 40%; spot_mean = {spot_mean:.4}, true_mean = {true_mean:.4}"
    );
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

  /// Bivariate Heston fixture matching the MATLAB Heston2D.m example
  /// `parameters=[0,0;0.4,0.4;2,2;1,1]`, `Rho=[0.5,-0.5,0,0,-0.5,0.5]`.
  /// Returns `(log_prices_1, log_prices_2, times, v_1, v_2)`.
  fn heston2d_paths() -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = 23401_usize;
    let t = 1.0_f64;
    let h = Heston2D::<f64, stochastic_rs_stochastic::simd_rng::Deterministic>::seeded(
      [Some(0.0_f64), Some(0.0_f64)],
      [Some(0.4_f64), Some(0.4_f64)],
      [0.0, 0.0],
      [0.4, 0.4],
      [2.0, 2.0],
      [1.0, 1.0],
      [0.5, -0.5, 0.0, 0.0, -0.5, 0.5],
      n,
      Some(t),
      Some(false),
      42,
    );
    let [x1, v1, x2, v2] = h.sample();
    let dt = t / (n as f64 - 1.0);
    let times: Vec<f64> = (0..n).map(|i| i as f64 * dt).collect();
    (x1.to_vec(), x2.to_vec(), times, v1.to_vec(), v2.to_vec())
  }

  #[test]
  fn test_integrated_covariance_bivariate_heston() {
    // For bivariate Heston with ρ(W1,W2)=0.5: IC = ρ ∫₀ᵀ √(v_1(s) v_2(s)) ds
    let (x1, x2, times, v1, v2) = heston2d_paths();
    let engine1 = FMVol::new(&x1, &times, 1.0);
    let engine2 = FMVol::new(&x2, &times, 1.0);
    let dt = 1.0 / (x1.len() - 1) as f64;
    let true_ic: f64 = (0..v1.len() - 1)
      .map(|i| {
        let m1 = (v1[i] * v2[i]).sqrt();
        let m2 = (v1[i + 1] * v2[i + 1]).sqrt();
        0.5 * (m1 + m2) * 0.5 * dt
      })
      .sum();
    let est_ic = engine1.integrated_covariance(&engine2);
    let rel_err = (est_ic - true_ic).abs() / true_ic.abs();
    assert!(
      est_ic > 0.0,
      "covariance should be > 0 for ρ>0, got {est_ic}"
    );
    assert!(
      rel_err < 0.05,
      "bivariate FM_int_cov rel_err = {rel_err:.4} > 5%, est={est_ic:.6}, true={true_ic:.6}"
    );
  }

  #[test]
  fn test_spot_variance_fe_kernel() {
    // FE convention uses (1 − |k|/M) instead of (1 − |k|/(M+1)).
    // For the same M, FE has slightly more aggressive smoothing — but for
    // a Heston path with M ≈ √N both produce results close to the truth.
    let (lp, v, times) = heston_paths();
    let engine = FMVol::new(&lp, &times, 1.0);
    let tau: Vec<f64> = (0..11).map(|i| i as f64 / 10.0).collect();

    let spot_fm = engine.spot_variance(&tau, None);
    let spot_fe = engine.spot_variance_fe(&tau, None);

    let step = (lp.len() - 1) / (tau.len() - 1);
    let mae_fm: f64 = (0..tau.len())
      .map(|i| (spot_fm[i] - v[i * step]).abs())
      .sum::<f64>()
      / tau.len() as f64;
    let mae_fe: f64 = (0..tau.len())
      .map(|i| (spot_fe[i] - v[i * step]).abs())
      .sum::<f64>()
      / tau.len() as f64;
    assert!(mae_fm < 0.30, "FM spot vol MAE {mae_fm:.4} too large");
    assert!(mae_fe < 0.30, "FE spot vol MAE {mae_fe:.4} too large");
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
