//! # Lognormal
//!
//! $$
//! f(x)=\frac{1}{x\sigma\sqrt{2\pi}}\exp\!\left(-\frac{(\ln x-\mu)^2}{2\sigma^2}\right),\ x>0
//! $$
//!

use rand::Rng;
use rand::distr::Distribution;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::SimdRngExt;

use super::SimdFloatExt;
use super::normal::SimdNormal;
use crate::seeded::Buffered;
use crate::seeded::StreamState;
use crate::traits::distribution::Sealed;
use crate::traits::distribution::SimdDistribution;
use crate::traits::distribution::SimdKernel;

/// Log-normal law: `ln X ~ N(mu, sigma)`; parameters only, a [`Seeded`](crate::Seeded) stream draws it in bulk.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SimdLogNormal<T> {
  mu: T,
  sigma: T,
}

/// A stream that transforms its own standard normal sub-stream, plus its single-draw buffer.
#[doc(hidden)]
#[derive(Clone, Debug)]
pub struct NormalDerivedState<T: SimdFloatExt, R: SimdRngExt> {
  pub(crate) normal: StreamState<T, R, 64>,
  pub(crate) buf: Buffered<T, 16>,
}

impl<T: SimdFloatExt> SimdLogNormal<T> {
  /// Creates a log-normal distribution.
  ///
  /// - `mu` — mean of **ln(X)**, not of X itself (matches the module
  ///   header's μ; `mean() = exp(μ + σ²/2)` is the LogNormal-mean
  ///   formula derived from it).
  /// - `sigma` — standard deviation of **ln(X)**, not of X itself
  ///   (matches the module header's σ), must be > 0.
  pub fn new(mu: T, sigma: T) -> Self {
    assert!(
      sigma > T::zero(),
      "sigma must satisfy `sigma > T::zero()`, got sigma = {sigma:?}"
    );
    Self { mu, sigma }
  }

  /// The mean `μ` of `ln X`.
  pub fn mu(&self) -> T {
    self.mu
  }

  /// The standard deviation `σ` of `ln X`.
  pub fn sigma(&self) -> T {
    self.sigma
  }

  fn fill_parts<R: SimdRngExt>(&self, normal: &mut StreamState<T, R, 64>, out: &mut [T]) {
    let mm = T::splat(self.mu);
    let ss = T::splat(self.sigma);
    let mut tmp = [T::zero(); 16];
    let (chunks, rem) = out.as_chunks_mut::<16>();
    for chunk in chunks {
      SimdNormal::<T>::fill_standard(&mut normal.rng, &mut tmp[..16]);
      for half in 0..2 {
        let base = half * 8;
        let mut a = [T::zero(); 8];
        a.copy_from_slice(&tmp[base..base + 8]);
        let z = T::simd_from_array(a);
        let x = T::simd_to_array(T::simd_exp(mm + ss * z));
        chunk[base..base + 8].copy_from_slice(&x);
      }
    }
    if !rem.is_empty() {
      SimdNormal::<T>::fill_standard(&mut normal.rng, &mut tmp[..rem.len()]);
      let mut done = 0;
      while done + 8 <= rem.len() {
        let mut a = [T::zero(); 8];
        a.copy_from_slice(&tmp[done..done + 8]);
        let z = T::simd_from_array(a);
        let x = T::simd_to_array(T::simd_exp(mm + ss * z));
        rem[done..done + 8].copy_from_slice(&x);
        done += 8;
      }
      if done < rem.len() {
        let left = rem.len() - done;
        let mut a = [T::zero(); 8];
        a[..left].copy_from_slice(&tmp[done..done + left]);
        let z = T::simd_from_array(a);
        let x = T::simd_to_array(T::simd_exp(mm + ss * z));
        rem[done..done + left].copy_from_slice(&x[..left]);
      }
    }
  }

  pub(crate) fn draw_with<G: Rng + ?Sized>(&self, rng: &mut G) -> T {
    (self.mu + self.sigma * SimdNormal::<T>::standard().draw_with(rng)).exp()
  }
}

/// LogNormal(μ=0, σ=1) — the standard log-normal, matching [`SimdNormal`]'s
/// own N(0,1) default.
impl<T: SimdFloatExt> Default for SimdLogNormal<T> {
  fn default() -> Self {
    Self::new(T::zero(), T::one())
  }
}

impl<T: SimdFloatExt> Sealed for SimdLogNormal<T> {}

impl<T: SimdFloatExt> SimdDistribution for SimdLogNormal<T> {
  type State<R: SimdRngExt> = NormalDerivedState<T, R>;

  fn init<R: SimdRngExt, S: SeedExt>(&self, seed: &S) -> (NormalDerivedState<T, R>, u64) {
    let (normal, basis) = SimdNormal::<T>::standard().init::<R, S>(seed);
    (
      NormalDerivedState {
        normal,
        buf: Buffered::new(),
      },
      basis,
    )
  }
}

impl<T: SimdFloatExt> SimdKernel for SimdLogNormal<T> {
  type Item = T;

  #[inline]
  fn fill<R: SimdRngExt>(&self, state: &mut NormalDerivedState<T, R>, out: &mut [T]) {
    self.fill_parts(&mut state.normal, out);
  }

  #[inline]
  fn next<R: SimdRngExt>(&self, state: &mut NormalDerivedState<T, R>) -> T {
    let NormalDerivedState { normal, buf } = state;
    buf.pop(|b| self.fill_parts(normal, b))
  }
}

impl<T: SimdFloatExt> crate::traits::DistributionExt for SimdLogNormal<T> {
  fn pdf(&self, x: f64) -> f64 {
    if x <= 0.0 {
      return 0.0;
    }
    let mu = self.mu.to_f64().unwrap();
    let sigma = self.sigma.to_f64().unwrap();
    let z = (x.ln() - mu) / sigma;
    crate::special::norm_pdf(z) / (sigma * x)
  }

  fn cdf(&self, x: f64) -> f64 {
    if x <= 0.0 {
      return 0.0;
    }
    let mu = self.mu.to_f64().unwrap();
    let sigma = self.sigma.to_f64().unwrap();
    crate::special::norm_cdf((x.ln() - mu) / sigma)
  }

  fn inv_cdf(&self, p: f64) -> f64 {
    let mu = self.mu.to_f64().unwrap();
    let sigma = self.sigma.to_f64().unwrap();
    (mu + sigma * crate::special::ndtri(p)).exp()
  }

  fn mean(&self) -> f64 {
    let mu = self.mu.to_f64().unwrap();
    let sigma = self.sigma.to_f64().unwrap();
    (mu + 0.5 * sigma * sigma).exp()
  }

  fn median(&self) -> f64 {
    self.mu.to_f64().unwrap().exp()
  }

  fn mode(&self) -> f64 {
    let mu = self.mu.to_f64().unwrap();
    let sigma = self.sigma.to_f64().unwrap();
    (mu - sigma * sigma).exp()
  }

  fn variance(&self) -> f64 {
    let mu = self.mu.to_f64().unwrap();
    let sigma = self.sigma.to_f64().unwrap();
    let s2 = sigma * sigma;
    (s2.exp() - 1.0) * (2.0 * mu + s2).exp()
  }

  fn skewness(&self) -> f64 {
    let sigma = self.sigma.to_f64().unwrap();
    let s2 = sigma * sigma;
    (s2.exp() + 2.0) * (s2.exp() - 1.0).sqrt()
  }

  fn kurtosis(&self) -> f64 {
    // Excess kurtosis.
    let sigma = self.sigma.to_f64().unwrap();
    let s2 = sigma * sigma;
    (4.0 * s2).exp() + 2.0 * (3.0 * s2).exp() + 3.0 * (2.0 * s2).exp() - 6.0
  }

  fn entropy(&self) -> f64 {
    let mu = self.mu.to_f64().unwrap();
    let sigma = self.sigma.to_f64().unwrap();
    0.5 + 0.5 * (2.0 * std::f64::consts::PI * sigma * sigma).ln() + mu
  }
}

impl<T: SimdFloatExt> Distribution<T> for SimdLogNormal<T> {
  /// `exp(mu + sigma·Z)` with one scalar standard normal `Z` from the caller's rng.
  fn sample<G: Rng + ?Sized>(&self, rng: &mut G) -> T {
    self.draw_with(rng)
  }
}

py_distribution!(PyLogNormal, SimdLogNormal,
  sig: (mu, sigma, seed=None, dtype=None),
  params: (mu: f64, sigma: f64)
);

#[cfg(test)]
mod tests {
  use super::*;
  use crate::tests::scalar_ks_best_p;
  use crate::traits::DistributionExt as _;

  /// The honest `Distribution` draws from the caller's rng and agrees with the cdf.
  #[test]
  fn scalar_sample_matches_cdf() {
    let d = SimdLogNormal::<f64>::new(0.2, 0.6);
    let best = scalar_ks_best_p(&d, |x| d.cdf(x));
    assert!(best > 0.01, "best p = {best}");
  }
}
