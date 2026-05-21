//! Internal helpers: window resolution, bias-correction constant, and Fejér inversion.

use ndarray::Array1;
use num_complex::Complex;

use super::FMVol;
use crate::fourier_malliavin::coefficients::convolution_coefficients;
use crate::traits::FloatExt;

impl<T: FloatExt> FMVol<T> {
  /// Default smoothing window for spot-volatility / spot-leverage / spot-quarticity:
  /// `m ≈ √N`. This is the canonical Malliavin-Mancino choice (Malliavin & Mancino
  /// 2009, eq. 4.5 / Mancino-Recchioni 2015 §3): the bias from a finite Cesàro
  /// window scales like `1/m` while the variance of the Fourier-coefficient
  /// estimator scales like `m/N`, so balancing the two gives the `m ≈ N^{1/2}`
  /// optimum. The cast truncates the square root toward zero, which is
  /// intentional — we want `m ≤ √N` rather than rounding up across the
  /// bias/variance crossover.
  pub(super) fn resolve_m(&self, m: Option<usize>) -> usize {
    m.unwrap_or((self.n_freq as f64).sqrt() as usize)
  }

  /// Default smoothing window for the volvol / leverage estimators: `m ≈ N^{0.4}`.
  /// Slower-than-square-root window to keep the estimator's variance bounded
  /// for the cube/quartic cumulants, per Mancino-Recchioni (2015) §4.
  pub(super) fn resolve_m_volvol(&self, m: Option<usize>) -> usize {
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
  pub(super) fn resolve_m_volvol_bc(&self, m: Option<usize>) -> usize {
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
  pub(super) fn compute_bias_correction_constant(&self, big_m: usize) -> T {
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

  pub(super) fn center(&self) -> usize {
    self.max_freq
  }

  pub(super) fn const_(&self) -> T {
    T::from_f64_fast(std::f64::consts::TAU) / self.period
  }

  pub(super) fn vol_coeffs(&self, m: usize) -> Array1<Complex<T>> {
    assert!(
      self.n_freq + m <= self.max_freq,
      "need max_freq ≥ N + M = {} but have {}",
      self.n_freq + m,
      self.max_freq
    );
    convolution_coefficients(&self.dx, &self.dx, self.period, self.n_freq, m)
  }
}

/// Fejér kernel inversion (internal helper).
pub(super) fn fejer_inversion<T: FloatExt>(
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
