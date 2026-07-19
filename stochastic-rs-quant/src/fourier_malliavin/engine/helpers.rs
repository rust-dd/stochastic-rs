//! Internal helpers: window resolution, bias-correction constant, and Fejér inversion.

use ndarray::Array1;
use num_complex::Complex;

use super::FMVol;
use crate::fourier_malliavin::coefficients::convolution_coefficients;
use crate::traits::FloatExt;

const BIAS_CORRECTED_C_M: f64 = 0.05;

/// Default `M = floor(c_M / sqrt(mesh))` for the rate-efficient estimator.
///
/// Toscano et al. (2022), Section 4.3, report `c_M = 0.05` as the
/// MSE-optimal value for their Heston experiment. The constant is expressed
/// in the square root of the caller's time unit, exactly as the paper's
/// `M rho(n)^(1/2) -> c_M` condition requires.
pub(super) fn default_bias_corrected_m<T: FloatExt>(mesh: T) -> usize {
  let scaled = T::from_f64_fast(BIAS_CORRECTED_C_M) / mesh.sqrt();
  scaled.floor().to_usize().unwrap_or(usize::MAX).max(1)
}

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

  /// Default smoothing window for the bias-corrected estimator.
  ///
  /// Theorem 3.1 of Toscano et al. requires
  /// `M rho(n)^(1/2) -> c_M > 0`. The actual largest observation gap is
  /// retained by the engine, so irregular grids use their own mesh rather
  /// than the uniform-grid proxy `period / n`.
  pub(super) fn resolve_m_volvol_bc(&self, m: Option<usize>) -> usize {
    m.unwrap_or_else(|| default_bias_corrected_m(self.mesh))
  }

  /// Default outer Fejér window for corrected spot coefficients.
  pub(super) fn resolve_l_volvol_bc(&self, l: Option<usize>, big_m: usize) -> usize {
    l.unwrap_or_else(|| {
      let asymptotic = (self.n_freq as f64).powf(0.25).floor().max(1.0) as usize;
      asymptotic.min(big_m.saturating_mul(2))
    })
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
  /// For a general period, rescaling the grid to `[0, 2π]` gives
  ///
  /// $$K = \frac{M^2\rho(n)}{3T}\,
  ///   \left(1 + 2\eta\left(\frac{2N\rho(n)}{T}\right)\right).$$
  ///
  /// This reduces to `M²/(3n)` at the uniform-grid Nyquist frequency. The
  /// integrated correction is `(2π)² K / T` times integrated quarticity;
  /// at `T = 2π` this is the paper's `2π K` multiplier.
  pub(super) fn compute_bias_correction_constant(&self, big_m: usize) -> T {
    let two = T::from_f64_fast(2.0);
    let three = T::from_f64_fast(3.0);
    let a = two * T::from_usize_(self.n_freq) * self.mesh / self.period;
    let r = a - a.floor();
    let eta = if a.abs() > T::epsilon() {
      r * (T::one() - r) / (two * a * a)
    } else {
      T::zero()
    };
    let m = T::from_usize_(big_m);
    m * m * self.mesh / (three * self.period) * (T::one() + two * eta)
  }

  /// Bias-corrected Fourier coefficients of the spot volatility of volatility.
  ///
  /// This is equation (11) of Toscano et al. after rescaling `[0, T]`
  /// to `[0, 2π]`. Coefficients are returned for `k = -L, ..., L`.
  pub(super) fn bias_corrected_volvol_coefficients(
    &self,
    big_m: usize,
    big_l: usize,
  ) -> Array1<Complex<T>> {
    let mm = big_m + big_l;
    let c_v = self.vol_coeffs(mm);
    let k_const = self.compute_bias_correction_constant(big_m);
    let scale = T::from_f64_fast(std::f64::consts::TAU.powi(2)) / self.period;
    let m_plus_1 = T::from_usize_(big_m + 1);
    let mut corrected = Array1::<Complex<T>>::zeros(2 * big_l + 1);

    for (j, k) in (-(big_l as i64)..=(big_l as i64)).enumerate() {
      let mut derivative_sum = Complex::<T>::new(T::zero(), T::zero());
      let mut quarticity_sum = Complex::<T>::new(T::zero(), T::zero());
      for h in -(big_m as i64)..=(big_m as i64) {
        let product = c_v[(mm as i64 + h) as usize] * c_v[(mm as i64 + k - h) as usize];
        let fejer = T::one() - T::from_usize_(h.unsigned_abs() as usize) / m_plus_1;
        let derivative = T::from_f64_fast(h as f64 * (h - k) as f64);
        derivative_sum = derivative_sum + product * fejer * derivative;
        quarticity_sum = quarticity_sum + product;
      }
      corrected[j] = (derivative_sum / m_plus_1 - quarticity_sum * k_const) * scale;
    }

    corrected
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
