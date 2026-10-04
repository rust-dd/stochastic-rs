//! # Chi Square
//!
//! $$
//! X\sim\chi^2_\nu,\quad f(x)=\frac{1}{2^{\nu/2}\Gamma(\nu/2)}x^{\nu/2-1}e^{-x/2}
//! $$
//!
//! Sampling: `Gamma(k/2, 2)` by Marsaglia, G., Tsang, W.W. (2000), "A simple method for generating gamma variables", *ACM TOMS* 26(3), 363-372, DOI 10.1145/358407.358414.

use rand::Rng;
use rand::distr::Distribution;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::SimdRngExt;

use super::SimdFloatExt;
use super::gamma::GammaState;
use super::gamma::SimdGamma;
use crate::traits::distribution::Sealed;
use crate::traits::distribution::SimdDistribution;
use crate::traits::distribution::SimdKernel;

/// Chi-squared law with `k` degrees of freedom: parameters only; a [`Seeded`](crate::Seeded) stream draws it in bulk.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SimdChiSquared<T> {
  df: T,
  gamma: SimdGamma<T>,
}

impl<T: SimdFloatExt> SimdChiSquared<T> {
  /// Creates a chi-squared distribution, reparametrized internally as
  /// `Gamma(k/2, scale=2)` (a χ²_k variate is exactly `2·Gamma(k/2, 1)`).
  ///
  /// - `k` — degrees of freedom (the module header's own ν).
  pub fn new(k: T) -> Self {
    assert!(
      k.is_finite(),
      "k must satisfy `k.is_finite()`, got k = {k:?}"
    );
    assert!(
      k > T::zero(),
      "k must satisfy `k > T::zero()`, got k = {k:?}"
    );
    let shape = k * T::from(0.5).unwrap();
    assert!(
      shape > T::zero(),
      "k must satisfy `k / 2 > 0`, got k = {k:?}"
    );
    Self {
      df: k,
      gamma: SimdGamma::new(shape, T::from(2.0).unwrap()),
    }
  }

  /// The degrees of freedom `k`.
  pub fn k(&self) -> T {
    self.df
  }

  pub(crate) fn draw_with<G: Rng + ?Sized>(&self, rng: &mut G) -> T {
    self.gamma.draw_with(rng)
  }
}

impl<T: SimdFloatExt> Sealed for SimdChiSquared<T> {}

impl<T: SimdFloatExt> SimdDistribution for SimdChiSquared<T> {
  type State<R: SimdRngExt> = GammaState<T, R>;

  fn init<R: SimdRngExt, S: SeedExt>(&self, seed: &S) -> (GammaState<T, R>, u64) {
    self.gamma.init::<R, S>(seed)
  }
}

impl<T: SimdFloatExt> SimdKernel for SimdChiSquared<T> {
  type Item = T;

  #[inline]
  fn fill<R: SimdRngExt>(&self, state: &mut GammaState<T, R>, out: &mut [T]) {
    self.gamma.fill(state, out);
  }

  #[inline]
  fn next<R: SimdRngExt>(&self, state: &mut GammaState<T, R>) -> T {
    self.gamma.next(state)
  }
}

impl<T: SimdFloatExt> Distribution<T> for SimdChiSquared<T> {
  /// One scalar `Gamma(k/2, 2)` draw on the caller's rng.
  fn sample<G: Rng + ?Sized>(&self, rng: &mut G) -> T {
    self.draw_with(rng)
  }
}

impl<T: SimdFloatExt> crate::traits::DistributionExt for SimdChiSquared<T> {
  fn pdf(&self, x: f64) -> Option<f64> {
    if x <= 0.0 {
      return Some(0.0);
    }
    let k = self.df.to_f64().unwrap();
    let half_k = 0.5 * k;
    // f(x) = x^(k/2 − 1) e^(−x/2) / (2^(k/2) Γ(k/2))
    let log_pdf = (half_k - 1.0) * x.ln()
      - 0.5 * x
      - half_k * std::f64::consts::LN_2
      - crate::special::ln_gamma(half_k);
    Some(log_pdf.exp())
  }

  fn cdf(&self, x: f64) -> Option<f64> {
    if x <= 0.0 {
      return Some(0.0);
    }
    let k = self.df.to_f64().unwrap();
    Some(crate::special::gamma_p(0.5 * k, 0.5 * x))
  }

  fn quantile(&self, p: f64) -> Option<f64> {
    if p <= 0.0 {
      return Some(0.0);
    }
    if p >= 1.0 {
      return Some(f64::INFINITY);
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
        return Some(new_x);
      }
      x = new_x;
    }
    Some(x)
  }

  fn mean(&self) -> Option<f64> {
    Some(self.df.to_f64().unwrap())
  }

  fn median(&self) -> Option<f64> {
    // Wilson-Hilferty approximation k * (1 - 2/(9k))³.
    let k = self.df.to_f64().unwrap();
    Some(k * (1.0 - 2.0 / (9.0 * k)).powi(3))
  }

  fn mode(&self) -> Option<f64> {
    let k = self.df.to_f64().unwrap();
    Some((k - 2.0).max(0.0))
  }

  fn variance(&self) -> Option<f64> {
    Some(2.0 * self.df.to_f64().unwrap())
  }

  fn skewness(&self) -> Option<f64> {
    Some((8.0 / self.df.to_f64().unwrap()).sqrt())
  }

  fn kurtosis(&self) -> Option<f64> {
    Some(12.0 / self.df.to_f64().unwrap())
  }

  fn entropy(&self) -> Option<f64> {
    let k = self.df.to_f64().unwrap();
    let half_k = 0.5 * k;
    Some(
      half_k
        + std::f64::consts::LN_2
        + crate::special::ln_gamma(half_k)
        + (1.0 - half_k) * crate::special::digamma(half_k),
    )
  }

  fn characteristic_function(&self, t: f64) -> Option<num_complex::Complex64> {
    // φ(t) = (1 - 2it)^(-k/2)
    let k = self.df.to_f64().unwrap();
    let one_minus_2it = num_complex::Complex64::new(1.0, -2.0 * t);
    Some(one_minus_2it.powf(-0.5 * k))
  }

  fn moment_generating_function(&self, t: f64) -> Option<f64> {
    let k = self.df.to_f64().unwrap();
    if t < 0.5 {
      Some((1.0 - 2.0 * t).powf(-0.5 * k))
    } else {
      Some(f64::INFINITY)
    }
  }
}

py_distribution!(PyChiSquared, SimdChiSquared,
  sig: (k, seed=None, dtype=None),
  params: (k: f64)
);

#[cfg(test)]
mod tests {
  use super::*;
  use crate::tests::scalar_ks_best_p;
  use crate::traits::DistributionExt as _;

  /// The honest `Distribution` draws from the caller's rng and agrees with the cdf.
  #[test]
  fn scalar_sample_matches_cdf() {
    let d = SimdChiSquared::<f64>::new(6.0);
    let best = scalar_ks_best_p(&d, |x| d.cdf(x).unwrap());
    assert!(best > 0.01, "best p = {best}");
  }

  #[test]
  #[should_panic(expected = "k must satisfy `k > T::zero()`, got k = 0.0")]
  fn new_rejects_zero_degrees_of_freedom() {
    SimdChiSquared::<f64>::new(0.0);
  }
}
