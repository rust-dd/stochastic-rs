//! # Gamma
//!
//! $$
//! f(x)=\frac{\beta^\alpha}{\Gamma(\alpha)}x^{\alpha-1}e^{-\beta x},\ x>0
//! $$
//!
use std::cell::UnsafeCell;
use stochastic_rs_core::simd_rng::Unseeded;

use rand::Rng;
use rand_distr::Distribution;

use super::SimdFloatExt;
use super::normal::SimdNormal;
use crate::simd_rng::SimdRng;

pub struct SimdGamma<T: SimdFloatExt> {
  alpha: T,
  scale: T,
  buffer: UnsafeCell<[T; 16]>,
  index: UnsafeCell<usize>,
  normal: SimdNormal<T>,
  simd_rng: UnsafeCell<SimdRng>,
}

impl<T: SimdFloatExt> SimdGamma<T> {


  /// Creates a gamma distribution with RNGs from a [`SeedExt`](crate::simd_rng::SeedExt) source.
  /// Each sub-component (normal, main rng) gets an independent stream.
  pub fn new<S: crate::simd_rng::SeedExt>(alpha: T, scale: T, seed: &S) -> Self {
    assert!(alpha > T::zero() && scale > T::zero());
    Self {
      alpha,
      scale,
      buffer: UnsafeCell::new([T::zero(); 16]),
      index: UnsafeCell::new(16),
      normal: SimdNormal::new(T::zero(), T::one(), seed),
      simd_rng: UnsafeCell::new(seed.rng()),
    }
  }

  /// Returns a single sample using the internal SIMD RNG.
  #[inline]
  pub fn sample_fast(&self) -> T {
    let index = unsafe { &mut *self.index.get() };
    if *index >= 16 {
      self.refill_buffer();
    }
    let buf = unsafe { &mut *self.buffer.get() };
    let z = buf[*index];
    *index += 1;
    z
  }

  pub fn fill_slice<R: Rng + ?Sized>(&self, _rng: &mut R, out: &mut [T]) {
    self.fill_slice_fast(out);
  }

  pub fn fill_slice_fast(&self, out: &mut [T]) {
    let rng = unsafe { &mut *self.simd_rng.get() };
    let third = T::from(1.0 / 3.0).unwrap();
    let c1 = T::from(0.0331).unwrap();
    let half = T::from(0.5).unwrap();
    let nine = T::from(9.0).unwrap();

    if self.alpha < T::one() {
      let alpha_plus_one = self.alpha + T::one();
      let d = alpha_plus_one - third;
      let c = T::one() / (nine * d).sqrt();
      let inv_alpha = T::one() / self.alpha;
      for x in out.iter_mut() {
        let g = loop {
          let z: T = self.normal.sample(rng);
          let v = (T::one() + c * z).powi(3);
          if v <= T::zero() {
            continue;
          }
          let u: T = T::sample_uniform_simd(rng);
          let z2 = z * z;
          if u < T::one() - c1 * z2 * z2 {
            break d * v;
          }
          if u.ln() < half * z2 + d * (T::one() - v + v.ln()) {
            break d * v;
          }
        };
        let u: T = T::sample_uniform_simd(rng);
        *x = self.scale * g * u.powf(inv_alpha);
      }
    } else {
      let d = self.alpha - third;
      let c = T::one() / (nine * d).sqrt();
      for x in out.iter_mut() {
        let val = loop {
          let z: T = self.normal.sample(rng);
          let v = (T::one() + c * z).powi(3);
          if v <= T::zero() {
            continue;
          }
          let u: T = T::sample_uniform_simd(rng);
          let z2 = z * z;
          if u < T::one() - c1 * z2 * z2 {
            break d * v;
          }
          if u.ln() < half * z2 + d * (T::one() - v + v.ln()) {
            break d * v;
          }
        };
        *x = self.scale * val;
      }
    }
  }

  fn refill_buffer(&self) {
    let buf = unsafe { &mut *self.buffer.get() };
    self.fill_slice_fast(buf);
    unsafe {
      *self.index.get() = 0;
    }
  }
}

impl<T: SimdFloatExt> Clone for SimdGamma<T> {
  fn clone(&self) -> Self {
    Self::new(self.alpha, self.scale, &Unseeded)
  }
}

impl<T: SimdFloatExt> crate::traits::DistributionExt for SimdGamma<T> {
  fn pdf(&self, x: f64) -> f64 {
    if x <= 0.0 {
      return 0.0;
    }
    let alpha = self.alpha.to_f64().unwrap();
    let scale = self.scale.to_f64().unwrap();
    // f(x) = x^(α−1) e^(−x/θ) / (θ^α Γ(α))
    let log_pdf =
      (alpha - 1.0) * x.ln() - x / scale - alpha * scale.ln() - crate::special::ln_gamma(alpha);
    log_pdf.exp()
  }

  fn cdf(&self, x: f64) -> f64 {
    if x <= 0.0 {
      return 0.0;
    }
    let alpha = self.alpha.to_f64().unwrap();
    let scale = self.scale.to_f64().unwrap();
    crate::special::gamma_p(alpha, x / scale)
  }

  fn inv_cdf(&self, p: f64) -> f64 {
    // Newton-bisection hybrid on the CDF.
    if p <= 0.0 {
      return 0.0;
    }
    if p >= 1.0 {
      return f64::INFINITY;
    }
    let alpha = self.alpha.to_f64().unwrap();
    let scale = self.scale.to_f64().unwrap();
    // Start from the Wilson-Hilferty Gaussian approximation.
    let z = crate::special::ndtri(p);
    let mut x = alpha * (1.0 - 1.0 / (9.0 * alpha) + z / (3.0 * alpha.sqrt())).powi(3);
    if x <= 0.0 {
      x = 0.5 * alpha;
    }
    x *= scale;
    // 30 Newton iterations using f(x) = P(α, x/θ) − p, f'(x) = pdf(x).
    for _ in 0..30 {
      let f = crate::special::gamma_p(alpha, x / scale) - p;
      let pdf =
        ((alpha - 1.0) * x.ln() - x / scale - alpha * scale.ln() - crate::special::ln_gamma(alpha))
          .exp();
      if pdf <= 0.0 {
        break;
      }
      let dx = f / pdf;
      let new_x = (x - dx).max(x * 1e-12);
      if (new_x - x).abs() < 1e-14 * x.max(1.0) {
        return new_x;
      }
      x = new_x;
    }
    x
  }

  fn mean(&self) -> f64 {
    self.alpha.to_f64().unwrap() * self.scale.to_f64().unwrap()
  }

  fn mode(&self) -> f64 {
    let alpha = self.alpha.to_f64().unwrap();
    if alpha < 1.0 {
      0.0
    } else {
      (alpha - 1.0) * self.scale.to_f64().unwrap()
    }
  }

  fn variance(&self) -> f64 {
    let alpha = self.alpha.to_f64().unwrap();
    let scale = self.scale.to_f64().unwrap();
    alpha * scale * scale
  }

  fn skewness(&self) -> f64 {
    let alpha = self.alpha.to_f64().unwrap();
    2.0 / alpha.sqrt()
  }

  fn kurtosis(&self) -> f64 {
    // Excess kurtosis.
    let alpha = self.alpha.to_f64().unwrap();
    6.0 / alpha
  }

  fn moment_generating_function(&self, t: f64) -> f64 {
    let alpha = self.alpha.to_f64().unwrap();
    let scale = self.scale.to_f64().unwrap();
    if t < 1.0 / scale {
      (1.0 - scale * t).powf(-alpha)
    } else {
      f64::INFINITY
    }
  }

  fn characteristic_function(&self, t: f64) -> num_complex::Complex64 {
    // φ(t) = (1 − i θ t)^{−α}
    let alpha = self.alpha.to_f64().unwrap();
    let scale = self.scale.to_f64().unwrap();
    let denom = num_complex::Complex64::new(1.0, -scale * t);
    denom.powf(-alpha)
  }

  fn entropy(&self) -> f64 {
    let alpha = self.alpha.to_f64().unwrap();
    let scale = self.scale.to_f64().unwrap();
    alpha
      + scale.ln()
      + crate::special::ln_gamma(alpha)
      + (1.0 - alpha) * crate::special::digamma(alpha)
  }

  fn median(&self) -> f64 {
    self.inv_cdf(0.5)
  }
}

impl<T: SimdFloatExt> Distribution<T> for SimdGamma<T> {
  fn sample<R: Rng + ?Sized>(&self, _rng: &mut R) -> T {
    let idx = unsafe { &mut *self.index.get() };
    if *idx >= 16 {
      self.refill_buffer();
    }
    let val = unsafe { (*self.buffer.get())[*idx] };
    *idx += 1;
    val
  }
}

py_distribution!(PyGamma, SimdGamma,
  sig: (alpha, scale, seed=None, dtype=None),
  params: (alpha: f64, scale: f64)
);
