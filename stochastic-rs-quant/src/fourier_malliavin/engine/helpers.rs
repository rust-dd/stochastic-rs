//! Internal helpers: window resolution, bias-correction constant, and Fejér inversion.

use ndarray::Array1;
use ndarray::ArrayView1;
use num_complex::Complex;

use super::FmVol;
use crate::fourier_malliavin::coefficients::convolution_coefficients;
use crate::traits::RealExt;

const BIAS_CORRECTED_C_M: f64 = 0.05;

/// Default `M = ⌊c_M/√mesh⌋` with Toscano et al.'s (2022, §4.3) MSE-optimal `c_M = 0.05`, in the
/// square root of the caller's time unit as their `M ρ(n)^{1/2} → c_M` requires.
pub(super) fn default_bias_corrected_m<T: RealExt>(mesh: T) -> usize {
  let scaled = T::from_f64_fast(BIAS_CORRECTED_C_M) / mesh.sqrt();
  scaled.floor().to_usize().unwrap_or(usize::MAX).max(1)
}

impl<T: RealExt> FmVol<T> {
  /// Default spot window `m = ⌊√N⌋`, balancing the Cesàro bias `~1/m` against the variance `~m/N`
  /// (Malliavin & Mancino 2009, eq. 4.5; Mancino & Recchioni 2015, §3), truncated so `m ≤ √N`.
  pub(super) fn resolve_m(&self, m: Option<usize>) -> usize {
    m.unwrap_or((self.n_freq as f64).sqrt() as usize)
  }

  /// Default vol-of-vol and leverage window `m ≈ N^{0.4}`, below `√N` to bound the variance of the
  /// cubic and quartic cumulants (Mancino & Recchioni 2015, §4).
  pub(super) fn resolve_m_volvol(&self, m: Option<usize>) -> usize {
    m.unwrap_or((self.n_freq as f64).powf(0.4) as usize)
  }

  /// Default bias-corrected window under Toscano et al.'s Theorem 3.1, `M ρ(n)^{1/2} → c_M > 0`, on
  /// the grid's own largest gap, not the uniform proxy `period / n`.
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

  /// Bias-correction constant `K` of Toscano et al. (2022), eq. (51), with `c_M = Mρ(n)^{1/2}`,
  /// `c_N = Nρ(n)`, `η(a) = r(a)(1 − r(a))/(2a²)`, `r(a) = a − ⌊a⌋`, rescaled to a period `T`:
  ///
  /// $$
  /// K := \tfrac{1}{3}\cdot \tfrac{c_M^2}{2\pi}\,\bigl(1 + 2\eta(c_N/\pi)\bigr),\qquad
  /// K = \frac{M^2\rho(n)}{3T}\left(1 + 2\eta\left(\frac{2N\rho(n)}{T}\right)\right)
  /// $$
  ///
  /// Reference: Toscano et al. (2022), arXiv:2112.14529v3, p. 42
  pub(super) fn compute_bias_correction_constant(&self, big_m: usize) -> T {
    let two = T::from_f64_fast(2.0);
    let three = T::from_f64_fast(3.0);
    let a = two * T::from_usize_(self.n_freq) * self.mesh / self.period;
    let r = a - a.floor();
    let eta = if a.abs() > T::machine_epsilon() {
      r * (T::one() - r) / (two * a * a)
    } else {
      T::zero()
    };
    let m = T::from_usize_(big_m);
    m * m * self.mesh / (three * self.period) * (T::one() + two * eta)
  }

  /// Bias-corrected vol-of-vol Fourier coefficients for `k = −L..=L`, Toscano et al.'s eq. (11)
  /// rescaled from `[0, T]` to `[0, 2π]`.
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
    convolution_coefficients(self.dx.view(), self.dx.view(), self.period, self.n_freq, m)
  }
}

/// Fejér kernel inversion (internal helper).
pub(super) fn fejer_inversion<T: RealExt>(
  coeffs: ArrayView1<Complex<T>>,
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
