//! # Complex fOU
//!
//! $$
//! dZ_t=-(\lambda-i\omega)Z_t\,dt+\sqrt{a}\,d\zeta_t,\qquad
//! \zeta_t=\frac{B_t^{(1)}+iB_t^{(2)}}{\sqrt{2}}
//! $$
//!
//! Equivalent real-imaginary form used in this implementation:
//! $$
//! \begin{aligned}
//! dX_1(t)&=-(\lambda X_1(t)+\omega X_2(t))\,dt+\sqrt{a/2}\,dB_t^{(1)},\\
//! dX_2(t)&=(\omega X_1(t)-\lambda X_2(t))\,dt+\sqrt{a/2}\,dB_t^{(2)}.
//! \end{aligned}
//! $$
//!
//! Source:
//! - https://arxiv.org/abs/2406.18004
use ndarray::Array1;
use num_complex::Complex;

use crate::stochastic::noise::fgn::FGN;
use crate::traits::FloatExt;
use crate::traits::ProcessExt;

/// Complex fractional Ornstein-Uhlenbeck process.
///
/// Source:
/// - https://arxiv.org/abs/2406.18004
pub struct CFOU<T: FloatExt> {
  /// Hurst exponent of the driving fractional Brownian motion.
  pub hurst: T,
  /// Real part of the complex mean-reversion coefficient (`lambda > 0`).
  pub lambda: T,
  /// Imaginary-frequency part of the complex mean-reversion coefficient.
  pub omega: T,
  /// Noise intensity parameter in `sqrt(a) d\zeta_t` (`a > 0`).
  pub a: T,
  /// Number of discrete simulation points.
  pub n: usize,
  /// Initial value of the real part `X_1(0)`.
  pub x1_0: Option<T>,
  /// Initial value of the imaginary part `X_2(0)`.
  pub x2_0: Option<T>,
  /// Total simulation horizon.
  pub t: Option<T>,
  fgn: FGN<T>,
}

impl<T: FloatExt> CFOU<T> {
  #[must_use]
  pub fn new(
    hurst: T,
    lambda: T,
    omega: T,
    a: T,
    n: usize,
    x1_0: Option<T>,
    x2_0: Option<T>,
    t: Option<T>,
  ) -> Self {
    assert!(n >= 2, "n must be at least 2");
    assert!(lambda > T::zero(), "lambda must be positive");
    assert!(a > T::zero(), "a must be positive");

    Self {
      hurst,
      lambda,
      omega,
      a,
      n,
      x1_0,
      x2_0,
      t,
      fgn: FGN::new(hurst, n - 1, t),
    }
  }
}

impl<T: FloatExt> ProcessExt<T> for CFOU<T> {
  type Output = Array1<Complex<T>>;

  /// Samples the complex path directly as `Z_t = X_1(t) + i X_2(t)`.
  ///
  /// Euler step:
  /// `Z_{k+1} = Z_k + (-(lambda - i omega) Z_k) dt + sqrt(a) Δζ_k`,
  /// with `Δζ_k = (ΔB_k^{(1)} + iΔB_k^{(2)}) / sqrt(2)`.
  ///
  /// Source:
  /// - https://arxiv.org/abs/2406.18004
  fn sample(&self) -> Self::Output {
    let dt = self.fgn.dt();
    let noise_1 = self.fgn.sample();
    let noise_2 = self.fgn.sample();
    let gamma = Complex::new(self.lambda, -self.omega);
    let dt_c = Complex::new(dt, T::zero());
    let noise_scale = (self.a * T::from_f64_fast(0.5)).sqrt();

    let mut z = Array1::from_elem(self.n, Complex::new(T::zero(), T::zero()));
    z[0] = Complex::new(
      self.x1_0.unwrap_or(T::zero()),
      self.x2_0.unwrap_or(T::zero()),
    );

    for i in 1..self.n {
      let z_prev = z[i - 1];
      let drift = -gamma * z_prev;
      let d_zeta = Complex::new(noise_1[i - 1], noise_2[i - 1]);
      z[i] = z_prev + drift * dt_c + d_zeta * noise_scale;
    }

    z
  }
}

impl<T: FloatExt> CFOU<T> {
  /// Samples the process and returns explicit real/imaginary components.
  #[must_use]
  pub fn sample_components(&self) -> [Array1<T>; 2] {
    let z = <Self as ProcessExt<T>>::sample(self);
    let mut x1 = Array1::<T>::zeros(self.n);
    let mut x2 = Array1::<T>::zeros(self.n);
    for i in 0..self.n {
      x1[i] = z[i].re;
      x2[i] = z[i].im;
    }
    [x1, x2]
  }
}

py_process_2d!(PyCFOU, CFOU,
  sig: (hurst, lambda, omega, a, n, x1_0=None, x2_0=None, t=None, dtype=None),
  params: (hurst: f64, lambda: f64, omega: f64, a: f64, n: usize, x1_0: Option<f64>, x2_0: Option<f64>, t: Option<f64>)
);

#[cfg(test)]
mod tests {
  use super::CFOU;
  use crate::traits::ProcessExt;

  #[test]
  fn cfou_sample_is_complex_and_finite() {
    let p = CFOU::<f64>::new(0.7, 1.2, 3.0, 0.4, 256, Some(0.0), Some(0.0), Some(1.0));
    let z = p.sample();

    assert_eq!(z.len(), 256);
    assert!(z.iter().all(|v| v.re.is_finite() && v.im.is_finite()));
  }

  #[test]
  fn cfou_components_are_finite() {
    let p = CFOU::<f64>::new(0.65, 0.9, 2.5, 0.6, 128, Some(0.1), Some(-0.1), Some(1.0));
    let [x1, x2] = p.sample_components();
    assert_eq!(x1.len(), 128);
    assert_eq!(x2.len(), 128);
    assert!(x1.iter().all(|v| v.is_finite()));
    assert!(x2.iter().all(|v| v.is_finite()));
  }
}
