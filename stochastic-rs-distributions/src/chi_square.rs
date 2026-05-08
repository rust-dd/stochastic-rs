//! # Chi Square
//!
//! $$
//! X\sim\chi^2_\nu,\quad f(x)=\frac{1}{2^{\nu/2}\Gamma(\nu/2)}x^{\nu/2-1}e^{-x/2}
//! $$
//!
use rand::Rng;
use rand_distr::Distribution;

use super::SimdFloatExt;
use super::gamma::SimdGamma;

pub struct SimdChiSquared<T: SimdFloatExt> {
  df: T,
  gamma: SimdGamma<T>,
}

impl<T: SimdFloatExt> SimdChiSquared<T> {
  #[inline]
  pub fn new(k: T) -> Self {
    Self::from_seed_source(k, &crate::simd_rng::Unseeded)
  }

  /// Creates a chi-squared distribution with a deterministic seed.
  #[inline]
  pub fn with_seed(k: T, seed: u64) -> Self {
    Self::from_seed_source(k, &crate::simd_rng::Deterministic::new(seed))
  }

  /// Creates a chi-squared distribution with RNGs from a [`SeedExt`](crate::simd_rng::SeedExt) source.
  pub fn from_seed_source(k: T, seed: &impl crate::simd_rng::SeedExt) -> Self {
    Self {
      df: k,
      gamma: SimdGamma::from_seed_source(k * T::from(0.5).unwrap(), T::from(2.0).unwrap(), seed),
    }
  }

  /// Returns a single sample using the internal SIMD RNG.
  #[inline]
  pub fn sample_fast(&self) -> T {
    self.gamma.sample_fast()
  }

  pub fn fill_slice<R: Rng + ?Sized>(&self, _rng: &mut R, out: &mut [T]) {
    self.gamma.fill_slice_fast(out);
  }

  pub fn fill_slice_fast(&self, out: &mut [T]) {
    self.gamma.fill_slice_fast(out);
  }
}

impl<T: SimdFloatExt> Clone for SimdChiSquared<T> {
  fn clone(&self) -> Self {
    Self::new(self.df)
  }
}

impl<T: SimdFloatExt> Distribution<T> for SimdChiSquared<T> {
  fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> T {
    self.gamma.sample(rng)
  }
}

impl<T: SimdFloatExt> crate::traits::DistributionExt for SimdChiSquared<T> {
  fn pdf(&self, x: f64) -> f64 {
    if x <= 0.0 {
      return 0.0;
    }
    let k = self.df.to_f64().unwrap();
    let half_k = 0.5 * k;
    // f(x) = x^(k/2 − 1) e^(−x/2) / (2^(k/2) Γ(k/2))
    let log_pdf = (half_k - 1.0) * x.ln()
      - 0.5 * x
      - half_k * std::f64::consts::LN_2
      - crate::special::ln_gamma(half_k);
    log_pdf.exp()
  }

  fn cdf(&self, x: f64) -> f64 {
    if x <= 0.0 {
      return 0.0;
    }
    let k = self.df.to_f64().unwrap();
    crate::special::gamma_p(0.5 * k, 0.5 * x)
  }

  fn inv_cdf(&self, p: f64) -> f64 {
    if p <= 0.0 {
      return 0.0;
    }
    if p >= 1.0 {
      return f64::INFINITY;
    }
    let k = self.df.to_f64().unwrap();
    // χ²_k = 2 · Gamma(α=k/2, scale=1) → use gamma quantile via Newton's
    // method, mirrored against a Wilson-Hilferty Gaussian start.
    let z = crate::special::ndtri(p);
    let mut x = k * (1.0 - 2.0 / (9.0 * k) + z * (2.0 / (9.0 * k)).sqrt()).powi(3);
    if x <= 0.0 {
      x = 0.5 * k;
    }
    let half_k = 0.5 * k;
    for _ in 0..30 {
      let f = crate::special::gamma_p(half_k, 0.5 * x) - p;
      let log_pdf = (half_k - 1.0) * x.ln()
        - 0.5 * x
        - half_k * std::f64::consts::LN_2
        - crate::special::ln_gamma(half_k);
      let pdf = log_pdf.exp();
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
    self.df.to_f64().unwrap()
  }

  fn median(&self) -> f64 {
    // Wilson-Hilferty approximation k * (1 - 2/(9k))³.
    let k = self.df.to_f64().unwrap();
    k * (1.0 - 2.0 / (9.0 * k)).powi(3)
  }

  fn mode(&self) -> f64 {
    let k = self.df.to_f64().unwrap();
    (k - 2.0).max(0.0)
  }

  fn variance(&self) -> f64 {
    2.0 * self.df.to_f64().unwrap()
  }

  fn skewness(&self) -> f64 {
    (8.0 / self.df.to_f64().unwrap()).sqrt()
  }

  fn kurtosis(&self) -> f64 {
    12.0 / self.df.to_f64().unwrap()
  }

  fn entropy(&self) -> f64 {
    let k = self.df.to_f64().unwrap();
    let half_k = 0.5 * k;
    half_k
      + std::f64::consts::LN_2
      + crate::special::ln_gamma(half_k)
      + (1.0 - half_k) * crate::special::digamma(half_k)
  }

  fn characteristic_function(&self, t: f64) -> num_complex::Complex64 {
    // φ(t) = (1 - 2it)^(-k/2)
    let k = self.df.to_f64().unwrap();
    let one_minus_2it = num_complex::Complex64::new(1.0, -2.0 * t);
    one_minus_2it.powf(-0.5 * k)
  }

  fn moment_generating_function(&self, t: f64) -> f64 {
    let k = self.df.to_f64().unwrap();
    if t < 0.5 {
      (1.0 - 2.0 * t).powf(-0.5 * k)
    } else {
      f64::INFINITY
    }
  }
}

py_distribution!(PyChiSquared, SimdChiSquared,
  sig: (k, seed=None, dtype=None),
  params: (k: f64)
);
