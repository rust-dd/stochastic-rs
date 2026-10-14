//! # Inverse Gauss
//!
//! $$
//! f(x)=\sqrt{\frac{\lambda}{2\pi x^3}}\exp\!\left(-\frac{\lambda(x-\mu)^2}{2\mu^2 x}\right),\ x>0
//! $$
//!
//! Sampling: Michael, J.R., Schucany, W.R., Haas, R.W. (1976), "Generating Random Variates Using Transformations with Multiple Roots", *The American Statistician* 30(2), 88-90, DOI 10.1080/00031305.1976.10479147.

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

const SMALL_INVERSE_GAUSS_THRESHOLD: usize = 16;

/// Inverse Gaussian law with mean `mu` and shape `lambda`: parameters only; a [`Seeded`](crate::Seeded) stream draws it in bulk.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SimdInverseGauss<T> {
  mu: T,
  lambda: T,
}

/// A stream with its own standard normal sub-stream, its own uniform engine and its single-draw buffer.
#[doc(hidden)]
#[derive(Clone, Debug)]
pub struct NormalPlusOwnState<T: SimdFloatExt, R: SimdRngExt> {
  pub(crate) normal: StreamState<T, R, 64>,
  pub(crate) rng: R,
  pub(crate) buf: Buffered<T, 16>,
}

impl<T: SimdFloatExt, R: SimdRngExt> NormalPlusOwnState<T, R> {
  /// The normal sub-stream, then the engine; the engine's seed is the fork basis.
  pub(crate) fn init<S: SeedExt>(seed: &S) -> (Self, u64) {
    let (normal, _) = SimdNormal::<T>::standard().init::<R, S>(seed);
    let stream_seed = seed.next_seed();
    (
      Self {
        normal,
        rng: R::from_seed(stream_seed),
        buf: Buffered::new(),
      },
      stream_seed,
    )
  }
}

impl<T: SimdFloatExt> SimdInverseGauss<T> {
  /// Creates an inverse-Gaussian distribution.
  ///
  /// - `mu` — mean μ > 0 (matches the module header's μ).
  /// - `lambda` — shape λ > 0 (matches the module header's λ; despite
  ///   the name, this is a shape, not a rate — variance = μ³/λ).
  pub fn new(mu: T, lambda: T) -> Self {
    assert!(
      mu.is_finite(),
      "mu must satisfy `mu.is_finite()`, got mu = {mu:?}"
    );
    assert!(
      lambda.is_finite(),
      "lambda must satisfy `lambda.is_finite()`, got lambda = {lambda:?}"
    );
    assert!(
      mu > T::zero(),
      "mu must satisfy `mu > T::zero()`, got mu = {mu:?}"
    );
    assert!(
      lambda > T::zero(),
      "lambda must satisfy `lambda > T::zero()`, got lambda = {lambda:?}"
    );
    Self { mu, lambda }
  }

  /// The mean `μ`.
  pub fn mu(&self) -> T {
    self.mu
  }

  /// The shape `λ`.
  pub fn lambda(&self) -> T {
    self.lambda
  }

  /// One Michael–Schucany–Haas draw from the normal `z` and the uniform `u`.
  #[inline]
  fn msh(&self, z: T, u: T) -> T {
    let two = T::from(2.0).unwrap();
    let four = T::from(4.0).unwrap();
    let w = z * z;
    let t1 = self.mu + (self.mu * self.mu * w) / (two * self.lambda);
    let rad = (four * self.mu * self.lambda * w + self.mu * self.mu * w * w).sqrt();
    // The small root as `μ²/big` rather than `t1 − (μ/2λ)·rad`, which cancels once `w` is large and in `f32`
    // can come out zero or negative, where the law is strictly positive.
    let big = t1 + (self.mu / (two * self.lambda)) * rad;
    let small = self.mu * self.mu / big;
    let check = self.mu / (self.mu + small);
    if u < check { small } else { big }
  }

  fn fill_parts<R: SimdRngExt>(
    &self,
    normal: &mut StreamState<T, R, 64>,
    rng: &mut R,
    out: &mut [T],
  ) {
    if out.len() < SMALL_INVERSE_GAUSS_THRESHOLD {
      for x in out.iter_mut() {
        *x = self.msh(
          SimdNormal::<T>::standard().next(normal),
          T::sample_uniform_simd(rng),
        );
      }
      return;
    }
    let two = T::splat(T::from(2.0).unwrap());
    let four = T::splat(T::from(4.0).unwrap());
    let mu = T::splat(self.mu);
    let lam = T::splat(self.lambda);
    let mut zbuf = [T::zero(); 64];
    let mut ubuf = [T::zero(); 64];
    let (chunks, rem) = out.as_chunks_mut::<64>();
    for chunk in chunks {
      SimdNormal::<T>::fill_standard(&mut normal.rng, &mut zbuf);
      T::fill_uniform_simd(rng, &mut ubuf);
      for (sub, (z8, u8)) in chunk.as_chunks_mut::<8>().0.iter_mut().zip(
        zbuf
          .as_chunks::<8>()
          .0
          .iter()
          .zip(ubuf.as_chunks::<8>().0.iter()),
      ) {
        let z = T::simd_from_array(*z8);
        let w = z * z;
        let t1 = mu + (mu * mu * w) / (two * lam);
        let rad = T::simd_sqrt(four * mu * lam * w + mu * mu * w * w);
        // The conjugate form of the small root; see `msh`.
        let alt = t1 + (mu / (two * lam)) * rad;
        let x = (mu * mu) / alt;
        let check = mu / (mu + x);
        let xa = T::simd_to_array(x);
        let ca = T::simd_to_array(check);
        let aa = T::simd_to_array(alt);
        for j in 0..8 {
          sub[j] = if u8[j] < ca[j] { xa[j] } else { aa[j] };
        }
      }
    }
    if !rem.is_empty() {
      let n = rem.len();
      SimdNormal::<T>::fill_standard(&mut normal.rng, &mut zbuf[..n]);
      T::fill_uniform_simd(rng, &mut ubuf[..n]);
      for i in 0..n {
        rem[i] = self.msh(zbuf[i], ubuf[i]);
      }
    }
  }

  pub(crate) fn draw_with<G: Rng + ?Sized>(&self, rng: &mut G) -> T {
    let z = SimdNormal::<T>::standard().draw_with(rng);
    self.msh(z, T::sample_uniform(rng))
  }
}

impl<T: SimdFloatExt> Sealed for SimdInverseGauss<T> {}

impl<T: SimdFloatExt> SimdDistribution for SimdInverseGauss<T> {
  type State<R: SimdRngExt> = NormalPlusOwnState<T, R>;

  fn init<R: SimdRngExt, S: SeedExt>(&self, seed: &S) -> (NormalPlusOwnState<T, R>, u64) {
    NormalPlusOwnState::init(seed)
  }
}

impl<T: SimdFloatExt> SimdKernel for SimdInverseGauss<T> {
  type Item = T;

  #[inline]
  fn fill<R: SimdRngExt>(&self, state: &mut NormalPlusOwnState<T, R>, out: &mut [T]) {
    self.fill_parts(&mut state.normal, &mut state.rng, out);
  }

  #[inline]
  fn next<R: SimdRngExt>(&self, state: &mut NormalPlusOwnState<T, R>) -> T {
    let NormalPlusOwnState { normal, rng, buf } = state;
    buf.pop(|b| self.fill_parts(normal, rng, b))
  }
}

impl<T: SimdFloatExt> Distribution<T> for SimdInverseGauss<T> {
  /// One scalar Michael–Schucany–Haas draw: a normal, then a uniform, from the caller's rng.
  fn sample<G: Rng + ?Sized>(&self, rng: &mut G) -> T {
    self.draw_with(rng)
  }
}

impl<T: SimdFloatExt> crate::traits::DistributionExt for SimdInverseGauss<T> {
  fn pdf(&self, x: f64) -> Option<f64> {
    let mu = self.mu.to_f64().unwrap();
    let lambda = self.lambda.to_f64().unwrap();
    if x <= 0.0 {
      Some(0.0)
    } else {
      Some(
        (lambda / (2.0 * std::f64::consts::PI * x.powi(3))).sqrt()
          * (-lambda * (x - mu).powi(2) / (2.0 * mu * mu * x)).exp(),
      )
    }
  }

  fn cdf(&self, x: f64) -> Option<f64> {
    let mu = self.mu.to_f64().unwrap();
    let lambda = self.lambda.to_f64().unwrap();
    if x <= 0.0 {
      return Some(0.0);
    }
    // F(x) = Φ(√(λ/x)·(x/μ-1)) + e^(2λ/μ) Φ(-√(λ/x)·(x/μ+1))
    let sqrt_lambda_over_x = (lambda / x).sqrt();
    let a = sqrt_lambda_over_x * (x / mu - 1.0);
    let b = sqrt_lambda_over_x * (x / mu + 1.0);
    Some(crate::special::norm_cdf(a) + (2.0 * lambda / mu).exp() * crate::special::norm_cdf(-b))
  }

  fn mean(&self) -> Option<f64> {
    Some(self.mu.to_f64().unwrap())
  }

  fn mode(&self) -> Option<f64> {
    let mu = self.mu.to_f64().unwrap();
    let lambda = self.lambda.to_f64().unwrap();
    Some(mu * ((1.0 + 9.0 * mu * mu / (4.0 * lambda * lambda)).sqrt() - 3.0 * mu / (2.0 * lambda)))
  }

  fn variance(&self) -> Option<f64> {
    let mu = self.mu.to_f64().unwrap();
    let lambda = self.lambda.to_f64().unwrap();
    Some(mu.powi(3) / lambda)
  }

  fn skewness(&self) -> Option<f64> {
    let mu = self.mu.to_f64().unwrap();
    let lambda = self.lambda.to_f64().unwrap();
    Some(3.0 * (mu / lambda).sqrt())
  }

  fn kurtosis(&self) -> Option<f64> {
    let mu = self.mu.to_f64().unwrap();
    let lambda = self.lambda.to_f64().unwrap();
    Some(15.0 * mu / lambda)
  }

  fn characteristic_function(&self, t: f64) -> Option<num_complex::Complex64> {
    // φ(t) = exp(λ/μ · (1 - sqrt(1 - 2 i μ² t / λ)))
    let mu = self.mu.to_f64().unwrap();
    let lambda = self.lambda.to_f64().unwrap();
    let inner = num_complex::Complex64::new(1.0, -2.0 * mu * mu * t / lambda);
    Some(
      (num_complex::Complex64::new(1.0, 0.0) - inner.sqrt())
        .scale(lambda / mu)
        .exp(),
    )
  }

  fn moment_generating_function(&self, t: f64) -> Option<f64> {
    // M(t) = exp(λ/μ · (1 - sqrt(1 - 2 μ² t / λ)))
    let mu = self.mu.to_f64().unwrap();
    let lambda = self.lambda.to_f64().unwrap();
    let arg = 1.0 - 2.0 * mu * mu * t / lambda;
    if arg < 0.0 {
      Some(f64::INFINITY)
    } else {
      Some(((lambda / mu) * (1.0 - arg.sqrt())).exp())
    }
  }
}

py_distribution!(PyInverseGauss, SimdInverseGauss,
  sig: (mu, lambda_, seed=None, dtype=None),
  params: (mu: f64, lambda_: f64)
);

#[cfg(test)]
mod tests {
  use super::SimdInverseGauss;
  use crate::tests::scalar_ks_best_p;
  use crate::traits::DistributionExt;

  #[test]
  fn scalar_sample_matches_cdf() {
    let d = SimdInverseGauss::<f64>::new(1.5, 3.0);
    let best = scalar_ks_best_p(&d, |x| d.cdf(x).unwrap());
    assert!(best > 0.01, "best p = {best}");
  }
}
