//! # Weibull
//!
//! $$
//! f(x)=\frac{k}{\lambda}\left(\frac{x}{\lambda}\right)^{k-1}e^{-(x/\lambda)^k},\ x\ge0
//! $$
//!
//! Sampling: `λ·E^{1/k}` with `E ~ Exp(1)`, inversion, Devroye, L. (1986), *Non-Uniform Random Variate Generation*, Springer, §II.2, DOI 10.1007/978-1-4613-8643-8.

use rand::Rng;
use rand::distr::Distribution;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::SimdRngExt;

use super::SimdFloatExt;
use super::exp::SimdExp;
use crate::seeded::Buffered;
use crate::seeded::StreamState;
use crate::traits::distribution::Sealed;
use crate::traits::distribution::SimdDistribution;
use crate::traits::distribution::SimdKernel;

/// Weibull law with scale `lambda` and shape `k`: parameters only; a [`Seeded`](crate::Seeded) stream draws it in bulk.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SimdWeibull<T> {
  lambda: T,
  k: T,
  inv_k: T,
}

/// A stream that transforms its own `Exp(1)` sub-stream, plus its single-draw buffer.
#[doc(hidden)]
#[derive(Clone, Debug)]
pub struct ExpDerivedState<T: SimdFloatExt, R: SimdRngExt> {
  exp: StreamState<T, R, 64>,
  buf: Buffered<T, 16>,
}

impl<T: SimdFloatExt> SimdWeibull<T> {
  /// Creates a Weibull distribution.
  ///
  /// - `lambda` — scale λ > 0 (matches the module header's λ).
  /// - `k` — shape k > 0 (matches the module header's k).
  pub fn new(lambda: T, k: T) -> Self {
    assert!(
      lambda > T::zero() && k > T::zero(),
      "lambda must satisfy `lambda > T::zero() && k > T::zero()`, got lambda = {lambda:?}, k = {k:?}"
    );
    Self {
      lambda,
      k,
      inv_k: T::one() / k,
    }
  }

  /// The scale `λ`.
  pub fn lambda(&self) -> T {
    self.lambda
  }

  /// The shape `k`.
  pub fn k(&self) -> T {
    self.k
  }

  /// `λ·E^{1/k}` over `Exp(1)` magnitudes drawn in 64-blocks, the power running 8-wide.
  fn fill_parts<R: SimdRngExt>(&self, exp: &mut StreamState<T, R, 64>, out: &mut [T]) {
    let exp1 = SimdExp::<T>::standard();
    let lambda = T::splat(self.lambda);
    let inv_k = self.inv_k;
    let mut tmp = [T::zero(); 64];
    let (chunks, rem) = out.as_chunks_mut::<64>();
    for chunk in chunks {
      exp1.fill(exp, &mut tmp);
      for (sub, e8) in chunk
        .as_chunks_mut::<8>()
        .0
        .iter_mut()
        .zip(tmp.as_chunks::<8>().0.iter())
      {
        let y = lambda * T::simd_powf(T::simd_from_array(*e8), inv_k);
        *sub = T::simd_to_array(y);
      }
    }
    if !rem.is_empty() {
      let n = rem.len();
      exp1.fill(exp, &mut tmp[..n]);
      let mut off = 0;
      let (sub, sub_rem) = rem.as_chunks_mut::<8>();
      for s in sub.iter_mut() {
        let mut a = [T::zero(); 8];
        a.copy_from_slice(&tmp[off..off + 8]);
        let y = lambda * T::simd_powf(T::simd_from_array(a), inv_k);
        *s = T::simd_to_array(y);
        off += 8;
      }
      for (i, x) in sub_rem.iter_mut().enumerate() {
        *x = self.lambda * tmp[off + i].powf(inv_k);
      }
    }
  }

  pub(crate) fn draw_with<G: Rng + ?Sized>(&self, rng: &mut G) -> T {
    self.lambda * SimdExp::<T>::standard().draw_with(rng).powf(self.inv_k)
  }
}

impl<T: SimdFloatExt> Sealed for SimdWeibull<T> {}

impl<T: SimdFloatExt> SimdDistribution for SimdWeibull<T> {
  type State<R: SimdRngExt> = ExpDerivedState<T, R>;

  fn init<R: SimdRngExt, S: SeedExt>(&self, seed: &S) -> (ExpDerivedState<T, R>, u64) {
    let (exp, basis) = SimdExp::<T>::standard().init::<R, S>(seed);
    (
      ExpDerivedState {
        exp,
        buf: Buffered::new(),
      },
      basis,
    )
  }
}

impl<T: SimdFloatExt> SimdKernel for SimdWeibull<T> {
  type Item = T;

  #[inline]
  fn fill<R: SimdRngExt>(&self, state: &mut ExpDerivedState<T, R>, out: &mut [T]) {
    self.fill_parts(&mut state.exp, out);
  }

  #[inline]
  fn next<R: SimdRngExt>(&self, state: &mut ExpDerivedState<T, R>) -> T {
    let ExpDerivedState { exp, buf } = state;
    buf.pop(|b| self.fill_parts(exp, b))
  }
}

impl<T: SimdFloatExt> Distribution<T> for SimdWeibull<T> {
  /// `λ·E^{1/k}` with one scalar Ziggurat `Exp(1)` draw on the caller's rng.
  fn sample<G: Rng + ?Sized>(&self, rng: &mut G) -> T {
    self.draw_with(rng)
  }
}

impl<T: SimdFloatExt> crate::traits::DistributionExt for SimdWeibull<T> {
  fn pdf(&self, x: f64) -> f64 {
    let lambda = self.lambda.to_f64().unwrap();
    let k = self.k.to_f64().unwrap();
    if x < 0.0 {
      0.0
    } else {
      let r = x / lambda;
      (k / lambda) * r.powf(k - 1.0) * (-r.powf(k)).exp()
    }
  }

  fn cdf(&self, x: f64) -> f64 {
    let lambda = self.lambda.to_f64().unwrap();
    let k = self.k.to_f64().unwrap();
    if x < 0.0 {
      0.0
    } else {
      1.0 - (-(x / lambda).powf(k)).exp()
    }
  }

  fn inv_cdf(&self, p: f64) -> f64 {
    let lambda = self.lambda.to_f64().unwrap();
    let inv_k = self.inv_k.to_f64().unwrap();
    lambda * (-(1.0 - p).ln()).powf(inv_k)
  }

  fn mean(&self) -> f64 {
    use crate::special::gamma;
    let lambda = self.lambda.to_f64().unwrap();
    let inv_k = self.inv_k.to_f64().unwrap();
    lambda * gamma(1.0 + inv_k)
  }

  fn median(&self) -> f64 {
    let lambda = self.lambda.to_f64().unwrap();
    let inv_k = self.inv_k.to_f64().unwrap();
    lambda * (std::f64::consts::LN_2).powf(inv_k)
  }

  fn mode(&self) -> f64 {
    let lambda = self.lambda.to_f64().unwrap();
    let k = self.k.to_f64().unwrap();
    if k > 1.0 {
      lambda * ((k - 1.0) / k).powf(1.0 / k)
    } else {
      0.0
    }
  }

  fn variance(&self) -> f64 {
    use crate::special::gamma;
    let lambda = self.lambda.to_f64().unwrap();
    let inv_k = self.inv_k.to_f64().unwrap();
    let g1 = gamma(1.0 + inv_k);
    let g2 = gamma(1.0 + 2.0 * inv_k);
    lambda * lambda * (g2 - g1 * g1)
  }

  fn skewness(&self) -> f64 {
    use crate::special::gamma;
    let inv_k = self.inv_k.to_f64().unwrap();
    let g1 = gamma(1.0 + inv_k);
    let g2 = gamma(1.0 + 2.0 * inv_k);
    let g3 = gamma(1.0 + 3.0 * inv_k);
    let mu = g1;
    let sigma2 = g2 - g1 * g1;
    let sigma = sigma2.sqrt();
    (g3 - 3.0 * mu * sigma2 - mu.powi(3)) / sigma.powi(3)
  }

  fn kurtosis(&self) -> f64 {
    use crate::special::gamma;
    let inv_k = self.inv_k.to_f64().unwrap();
    let g1 = gamma(1.0 + inv_k);
    let g2 = gamma(1.0 + 2.0 * inv_k);
    let g3 = gamma(1.0 + 3.0 * inv_k);
    let g4 = gamma(1.0 + 4.0 * inv_k);
    let sigma2 = g2 - g1 * g1;
    (-6.0 * g1.powi(4) + 12.0 * g1 * g1 * g2 - 3.0 * g2 * g2 - 4.0 * g1 * g3 + g4) / sigma2.powi(2)
  }

  fn entropy(&self) -> f64 {
    let lambda = self.lambda.to_f64().unwrap();
    let inv_k = self.inv_k.to_f64().unwrap();
    let euler = 0.577_215_664_901_532_9_f64;
    euler * (1.0 - inv_k) + (lambda * inv_k).ln() + 1.0
  }
}

py_distribution!(PyWeibull, SimdWeibull,
  sig: (lambda_, k, seed=None, dtype=None),
  params: (lambda_: f64, k: f64)
);

#[cfg(test)]
mod tests {
  use super::*;
  use crate::tests::scalar_ks_best_p;
  use crate::traits::DistributionExt as _;

  /// The honest `Distribution` draws from the caller's rng and agrees with the cdf.
  #[test]
  fn scalar_sample_matches_cdf() {
    let d = SimdWeibull::<f64>::new(2.0, 1.5);
    let best = scalar_ks_best_p(&d, |x| d.cdf(x));
    assert!(best > 0.01, "best p = {best}");
  }
}
