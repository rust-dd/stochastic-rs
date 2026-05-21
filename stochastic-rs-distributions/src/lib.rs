//! # stochastic-rs-distributions
//!
//! Probability distributions with SIMD bulk sampling, plus the foundational
//! `FloatExt` / `SimdFloatExt` trait machinery and float impls.

#![allow(non_snake_case)]
#![allow(clippy::type_complexity)]
#![allow(clippy::too_many_arguments)]

use rand::Rng;
pub use stochastic_rs_core::simd_rng;

#[macro_use]
mod macros;

pub mod float_impls;
mod simd_float_impls;
pub mod special;
pub mod traits;

#[cfg(test)]
mod tests;

#[cfg(feature = "python")]
pub use crate::traits::CallableDist;
pub use crate::traits::DistributionExt;
pub use crate::traits::DistributionSampler;
pub use crate::traits::FloatExt;
pub use crate::traits::Fn1D;
pub use crate::traits::Fn2D;
pub use crate::traits::SimdFloatExt;

pub mod alpha_stable;
pub mod beta;
pub mod binomial;
pub mod cauchy;
pub mod chi_square;
pub mod complex;
pub mod exp;
pub mod gamma;
pub mod geometric;
pub mod hypergeometric;
pub mod inverse_gauss;
pub mod lognormal;
pub mod non_central_chi_squared;
pub mod normal;
pub mod normal_inverse_gauss;

/// Type alias for `SimdNormal` backed by the experimental dual-stream RNG.
///
/// Enabled by the `dual-stream-rng` cargo feature. Production code continues
/// to use the default [`normal::SimdNormal`] alias parameter
/// (`R = SimdRng`); switching to this alias picks the same struct
/// monomorphised over `SimdRngDual`, which unrolls the Ziggurat hot loop
/// 2× for ≈ 5–11 % extra throughput on bulk Normal fills.
#[cfg(feature = "dual-stream-rng")]
pub type SimdNormalDual<T, const N: usize = 64> =
  normal::SimdNormal<T, N, stochastic_rs_core::simd_rng_dual::SimdRngDual>;

/// Type alias for [`exp::SimdExp`] backed by the experimental dual-stream
/// RNG. Same trade-offs as [`SimdNormalDual`].
#[cfg(feature = "dual-stream-rng")]
pub type SimdExpDual<T> = exp::SimdExp<T, stochastic_rs_core::simd_rng_dual::SimdRngDual>;

/// Type alias for [`exp::SimdExpZig`] (the bulk-fill primitive that powers
/// [`exp::SimdExp`]) backed by the dual-stream RNG.
#[cfg(feature = "dual-stream-rng")]
pub type SimdExpZigDual<T, const N: usize = 64> =
  exp::SimdExpZig<T, N, stochastic_rs_core::simd_rng_dual::SimdRngDual>;
pub mod pareto;
pub mod poisson;
pub mod studentt;
pub mod uniform;
pub mod weibull;

macro_rules! impl_distribution_sampler_float {
  ($($dist:ty),+ $(,)?) => {
    $(
      impl<T: SimdFloatExt> DistributionSampler<T> for $dist {
        #[inline]
        fn fill_slice<R: Rng + ?Sized>(&self, rng: &mut R, out: &mut [T]) {
          self.fill_slice(rng, out);
        }
      }
    )+
  };
}

macro_rules! impl_distribution_sampler_int {
  ($($dist:ty),+ $(,)?) => {
    $(
      impl<T: num_traits::PrimInt> DistributionSampler<T> for $dist {
        #[inline]
        fn fill_slice<R: Rng + ?Sized>(&self, rng: &mut R, out: &mut [T]) {
          self.fill_slice(rng, out);
        }
      }
    )+
  };
}

macro_rules! impl_distribution_sampler_float_const_n {
  ($($dist:ty),+ $(,)?) => {
    $(
      impl<T: SimdFloatExt, const N: usize> DistributionSampler<T> for $dist {
        #[inline]
        fn fill_slice<R: Rng + ?Sized>(&self, rng: &mut R, out: &mut [T]) {
          self.fill_slice(rng, out);
        }
      }
    )+
  };
}

impl_distribution_sampler_float!(
  alpha_stable::SimdAlphaStable<T>,
  beta::SimdBeta<T>,
  cauchy::SimdCauchy<T>,
  chi_square::SimdChiSquared<T>,
  exp::SimdExp<T>,
  gamma::SimdGamma<T>,
  inverse_gauss::SimdInverseGauss<T>,
  lognormal::SimdLogNormal<T>,
  normal_inverse_gauss::SimdNormalInverseGauss<T>,
  pareto::SimdPareto<T>,
  studentt::SimdStudentT<T>,
  uniform::SimdUniform<T>,
  weibull::SimdWeibull<T>,
);

impl_distribution_sampler_int!(
  binomial::SimdBinomial<T>,
  geometric::SimdGeometric<T>,
  hypergeometric::SimdHypergeometric<T>,
  poisson::SimdPoisson<T>,
);

impl_distribution_sampler_float_const_n!(normal::SimdNormal<T, N>, exp::SimdExpZig<T, N>,);

#[cfg(test)]
mod distribution_sampler_tests {
  use stochastic_rs_core::simd_rng::Unseeded;

  use super::DistributionSampler;
  use super::normal::SimdNormal;
  use super::poisson::SimdPoisson;

  #[test]
  fn sample_n_returns_requested_length() {
    let dist = SimdNormal::<f64>::new(0.0, 1.0, &Unseeded);
    let out = dist.sample_n(1024);
    assert_eq!(out.len(), 1024);
  }

  #[test]
  fn sample_matrix_float_has_expected_shape() {
    let dist = SimdNormal::<f32>::new(0.0, 1.0, &Unseeded);
    let out = dist.sample_matrix(32, 64);
    assert_eq!(out.shape(), &[32, 64]);
  }

  #[test]
  fn sample_matrix_int_has_expected_shape() {
    let dist = SimdPoisson::<i64>::new(1.5, &Unseeded);
    let out = dist.sample_matrix(16, 8);
    assert_eq!(out.shape(), &[16, 8]);
  }
}
