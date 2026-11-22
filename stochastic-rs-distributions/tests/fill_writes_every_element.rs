//! Every `SimdKernel::fill` writes each element it is handed: no sentinel survives at any length
//! class, so the callers' `assume_init` is sound.

use std::mem::MaybeUninit;

use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::SimdRng;
use stochastic_rs_core::simd_rng::SimdRngExt;
#[cfg(feature = "unstable-dual-stream-rng")]
use stochastic_rs_core::simd_rng_dual::SimdRngDual;
use stochastic_rs_distributions::SimdKernel;
use stochastic_rs_distributions::alpha_stable::SimdAlphaStable;
use stochastic_rs_distributions::beta::SimdBeta;
use stochastic_rs_distributions::binomial::SimdBinomial;
use stochastic_rs_distributions::cauchy::SimdCauchy;
use stochastic_rs_distributions::chi_square::SimdChiSquared;
use stochastic_rs_distributions::complex::ComplexDistribution;
use stochastic_rs_distributions::exp::SimdExp;
use stochastic_rs_distributions::gamma::SimdGamma;
use stochastic_rs_distributions::ged::SimdGed;
use stochastic_rs_distributions::generalized_hyperbolic::SimdGeneralizedHyperbolic;
use stochastic_rs_distributions::generalized_inverse_gauss::SimdGig;
use stochastic_rs_distributions::geometric::SimdGeometric;
use stochastic_rs_distributions::gev::SimdGev;
use stochastic_rs_distributions::gpd::SimdGpd;
use stochastic_rs_distributions::hypergeometric::SimdHypergeometric;
use stochastic_rs_distributions::inverse_gauss::SimdInverseGauss;
use stochastic_rs_distributions::johnson_su::SimdJohnsonSu;
use stochastic_rs_distributions::lognormal::SimdLogNormal;
use stochastic_rs_distributions::normal::SimdNormal;
use stochastic_rs_distributions::normal_inverse_gauss::SimdNormalInverseGauss;
use stochastic_rs_distributions::pareto::SimdPareto;
use stochastic_rs_distributions::poisson::SimdPoisson;
use stochastic_rs_distributions::skellam::SimdSkellam;
use stochastic_rs_distributions::skew_t::SimdSkewT;
use stochastic_rs_distributions::studentt::SimdStudentT;
use stochastic_rs_distributions::tempered_stable::SimdTemperedStable;
use stochastic_rs_distributions::truncated::SimdTruncatedBeta;
use stochastic_rs_distributions::truncated::SimdTruncatedExp;
use stochastic_rs_distributions::truncated::SimdTruncatedGamma;
use stochastic_rs_distributions::truncated::SimdTruncatedNormal;
use stochastic_rs_distributions::uniform::SimdUniform;
use stochastic_rs_distributions::variance_gamma::SimdVarianceGamma;
use stochastic_rs_distributions::weibull::SimdWeibull;

/// The empty slice, the `< 16` scalar path, the 8-lane body with remainders 0, 1 and 7, the
/// 64-draw refill and the 256-block kernels.
const LENGTHS: [usize; 13] = [0, 1, 7, 8, 9, 15, 16, 64, 255, 256, 257, 1_001, 4_096];

const SENTINEL: u8 = 0xA5;

fn check_on<R: SimdRngExt, D: SimdKernel>(name: &str, law: D) {
  let (mut state, _) = law.init::<R, _>(&Deterministic::new(3));
  let size = size_of::<D::Item>();
  for n in LENGTHS {
    let mut out = vec![MaybeUninit::<D::Item>::uninit(); n];
    // SAFETY: any byte pattern may be stored into `MaybeUninit` storage.
    unsafe {
      out
        .as_mut_ptr()
        .cast::<u8>()
        .write_bytes(SENTINEL, n * size)
    };
    law.fill(&mut state, &mut out);
    // SAFETY: every byte holds the sentinel or a written draw, and no `Item` has padding.
    let bytes = unsafe { std::slice::from_raw_parts(out.as_ptr().cast::<u8>(), n * size) };
    for (i, item) in bytes.chunks_exact(size).enumerate() {
      assert!(
        item.iter().any(|&b| b != SENTINEL),
        "{name}: element {i} of {n} was not written"
      );
    }
  }
}

fn check<D: SimdKernel>(name: &str, law: D) {
  check_on::<SimdRng, D>(name, law);
}

#[test]
fn continuous_kernels_write_every_element() {
  check("normal_f64", SimdNormal::<f64>::new(0.3, 1.7));
  check("normal_f32", SimdNormal::<f32>::new(0.3, 1.7));
  check("exp_f64", SimdExp::<f64>::new(1.8));
  check("exp_f32", SimdExp::<f32>::new(1.8));
  check("uniform_f64", SimdUniform::<f64>::new(-2.0, 3.0));
  check("uniform_f32", SimdUniform::<f32>::new(-2.0, 3.0));
  check("uniform_unit_f64", SimdUniform::<f64>::new(0.0, 1.0));
  check("uniform_unit_f32", SimdUniform::<f32>::new(0.0, 1.0));
  check("gamma_f64", SimdGamma::<f64>::new(2.5, 1.5));
  check("gamma_f32", SimdGamma::<f32>::new(2.5, 1.5));
  check("gamma_boost_f64", SimdGamma::<f64>::new(0.5, 2.0));
  check("gamma_boost_f32", SimdGamma::<f32>::new(0.5, 2.0));
  check("chi_squared_f64", SimdChiSquared::<f64>::new(6.0));
  check("chi_squared_f32", SimdChiSquared::<f32>::new(6.0));
  check("beta_f64", SimdBeta::<f64>::new(2.5, 4.0));
  check("beta_f32", SimdBeta::<f32>::new(2.5, 4.0));
  check("lognormal_f64", SimdLogNormal::<f64>::new(0.2, 0.6));
  check("lognormal_f32", SimdLogNormal::<f32>::new(0.2, 0.6));
  check("student_t_f64", SimdStudentT::<f64>::new(6.0));
  check("student_t_f32", SimdStudentT::<f32>::new(6.0));
  check("cauchy_f64", SimdCauchy::<f64>::new(1.0, 0.5));
  check("cauchy_f32", SimdCauchy::<f32>::new(1.0, 0.5));
  check("weibull_f64", SimdWeibull::<f64>::new(2.0, 1.5));
  check("weibull_f32", SimdWeibull::<f32>::new(2.0, 1.5));
  check("pareto_f64", SimdPareto::<f64>::new(1.0, 1.16));
  check("pareto_f32", SimdPareto::<f32>::new(1.0, 1.16));
  check(
    "alpha_stable_f64",
    SimdAlphaStable::<f64>::new(1.7, 0.3, 1.0, 0.0),
  );
  check(
    "alpha_stable_f32",
    SimdAlphaStable::<f32>::new(1.7, 0.3, 1.0, 0.0),
  );
  check(
    "alpha_stable_cauchy_f64",
    SimdAlphaStable::<f64>::new(1.0, 0.3, 1.0, 0.0),
  );
  check(
    "alpha_stable_cauchy_f32",
    SimdAlphaStable::<f32>::new(1.0, 0.3, 1.0, 0.0),
  );
  check(
    "alpha_stable_gauss_f64",
    SimdAlphaStable::<f64>::new(2.0, 0.0, 1.0, 0.0),
  );
  check(
    "alpha_stable_gauss_f32",
    SimdAlphaStable::<f32>::new(2.0, 0.0, 1.0, 0.0),
  );
  check(
    "tempered_stable_f64",
    SimdTemperedStable::<f64>::new(0.6, 2.0, 1.5),
  );
  check(
    "tempered_stable_f32",
    SimdTemperedStable::<f32>::new(0.6, 2.0, 1.5),
  );
  check(
    "tempered_stable_point_mass_f64",
    SimdTemperedStable::<f64>::new(0.5, 1e160, 1e300),
  );
  check(
    "tempered_stable_point_mass_f32",
    SimdTemperedStable::<f32>::new(0.9, 1e38, 1.0),
  );
  check(
    "nig_f64",
    SimdNormalInverseGauss::<f64>::new(2.0, 0.5, 1.0, 0.1),
  );
  check(
    "nig_f32",
    SimdNormalInverseGauss::<f32>::new(2.0, 0.5, 1.0, 0.1),
  );
  check(
    "variance_gamma_f64",
    SimdVarianceGamma::<f64>::new(0.2, 0.5, -0.1, 0.05),
  );
  check(
    "variance_gamma_f32",
    SimdVarianceGamma::<f32>::new(0.2, 0.5, -0.1, 0.05),
  );
  check(
    "generalized_hyperbolic_f64",
    SimdGeneralizedHyperbolic::<f64>::new(1.0, 2.0, 0.5, 1.5, -0.2),
  );
  check(
    "generalized_hyperbolic_f32",
    SimdGeneralizedHyperbolic::<f32>::new(1.0, 2.0, 0.5, 1.5, -0.2),
  );
  check("gig_a_f64", SimdGig::<f64>::new(0.3, 2.0, 0.5));
  check("gig_a_f32", SimdGig::<f32>::new(0.3, 2.0, 0.5));
  check("gig_b_f64", SimdGig::<f64>::new(0.2, 0.01, 1.0));
  check("gig_b_f32", SimdGig::<f32>::new(0.2, 0.01, 1.0));
  check("gig_c_f64", SimdGig::<f64>::new(3.0, 1.0, 1.0));
  check("gig_c_f32", SimdGig::<f32>::new(3.0, 1.0, 1.0));
  check("gig_negative_f64", SimdGig::<f64>::new(-0.5, 2.0, 0.5));
  check("gig_negative_f32", SimdGig::<f32>::new(-0.5, 2.0, 0.5));
  check("inverse_gauss_f64", SimdInverseGauss::<f64>::new(1.5, 3.0));
  check("inverse_gauss_f32", SimdInverseGauss::<f32>::new(1.5, 3.0));
  check("skew_t_f64", SimdSkewT::<f64>::new(5.0, -0.3));
  check("skew_t_f32", SimdSkewT::<f32>::new(5.0, -0.3));
  check(
    "johnson_su_f64",
    SimdJohnsonSu::<f64>::new(-0.5, 1.5, 0.2, 2.0),
  );
  check(
    "johnson_su_f32",
    SimdJohnsonSu::<f32>::new(-0.5, 1.5, 0.2, 2.0),
  );
  check("ged_f64", SimdGed::<f64>::new(0.0, 1.0, 1.5));
  check("ged_f32", SimdGed::<f32>::new(0.0, 1.0, 1.5));
  check("gev_frechet_f64", SimdGev::<f64>::new(0.0, 1.0, 0.3));
  check("gev_frechet_f32", SimdGev::<f32>::new(0.0, 1.0, 0.3));
  check("gev_gumbel_f64", SimdGev::<f64>::new(0.0, 1.0, 0.0));
  check("gev_gumbel_f32", SimdGev::<f32>::new(0.0, 1.0, 0.0));
  check("gev_weibull_f64", SimdGev::<f64>::new(0.0, 1.0, -0.3));
  check("gev_weibull_f32", SimdGev::<f32>::new(0.0, 1.0, -0.3));
  check("gpd_frechet_f64", SimdGpd::<f64>::new(0.0, 1.0, 0.3));
  check("gpd_frechet_f32", SimdGpd::<f32>::new(0.0, 1.0, 0.3));
  check("gpd_exp_f64", SimdGpd::<f64>::new(0.0, 1.0, 0.0));
  check("gpd_exp_f32", SimdGpd::<f32>::new(0.0, 1.0, 0.0));
  check("gpd_bounded_f64", SimdGpd::<f64>::new(0.0, 1.0, -0.3));
  check("gpd_bounded_f32", SimdGpd::<f32>::new(0.0, 1.0, -0.3));
}

#[test]
fn discrete_kernels_write_every_element() {
  check("poisson_u32", SimdPoisson::<u32>::new(12.0));
  check("poisson_u64", SimdPoisson::<u64>::new(12.0));
  check("poisson_i64", SimdPoisson::<i64>::new(12.0));
  check("poisson_large_u64", SimdPoisson::<u64>::new(800.0));
  check("poisson_ptrs_u64", SimdPoisson::<u64>::new(1500.0));
  check("binomial_btrs_u32", SimdBinomial::<u32>::new(60, 0.4));
  check("binomial_waiting_u32", SimdBinomial::<u32>::new(15, 0.3));
  check("binomial_btrs_flip_u32", SimdBinomial::<u32>::new(60, 0.7));
  check(
    "binomial_waiting_flip_u32",
    SimdBinomial::<u32>::new(15, 0.8),
  );
  check("binomial_no_trials_u32", SimdBinomial::<u32>::new(0, 0.4));
  check("binomial_never_u32", SimdBinomial::<u32>::new(15, 0.0));
  check("binomial_always_u32", SimdBinomial::<u32>::new(15, 1.0));
  check("geometric_u64", SimdGeometric::<u64>::new(0.15));
  check(
    "hypergeometric_u32",
    SimdHypergeometric::<u32>::new(60, 25, 20),
  );
  check("skellam", SimdSkellam::new(9.0, 5.0));
}

#[test]
fn truncated_kernels_write_every_element() {
  check(
    "truncated_normal_f64",
    SimdTruncatedNormal::<f64>::new(0.0, 1.0, -1.0, 2.0),
  );
  check(
    "truncated_normal_f32",
    SimdTruncatedNormal::<f32>::new(0.0, 1.0, -1.0, 2.0),
  );
  check(
    "truncated_normal_exp_tail_f64",
    SimdTruncatedNormal::<f64>::new(0.0, 1.0, 3.0, 6.0),
  );
  check(
    "truncated_normal_uniform_tail_f64",
    SimdTruncatedNormal::<f64>::new(0.0, 1.0, 3.0, 3.2),
  );
  check(
    "truncated_normal_exp_tail_short_f64",
    SimdTruncatedNormal::<f64>::new(0.0, 1.0, 3.0, 3.4),
  );
  check(
    "truncated_normal_mirrored_f64",
    SimdTruncatedNormal::<f64>::new(0.0, 1.0, -6.0, -3.0),
  );
  check(
    "truncated_normal_inverse_cdf_f64",
    SimdTruncatedNormal::<f64>::new(0.0, 1.0, -0.05, 0.05),
  );
  check(
    "truncated_exp_f64",
    SimdTruncatedExp::<f64>::new(2.0, 0.0, 1.5),
  );
  check(
    "truncated_exp_f32",
    SimdTruncatedExp::<f32>::new(2.0, 0.0, 1.5),
  );
  check(
    "truncated_beta_f64",
    SimdTruncatedBeta::<f64>::new(2.0, 2.0, 0.2, 0.8),
  );
  check(
    "truncated_beta_f32",
    SimdTruncatedBeta::<f32>::new(2.0, 2.0, 0.2, 0.8),
  );
  check(
    "truncated_beta_inversion_f64",
    SimdTruncatedBeta::<f64>::new(2.0, 3.0, 0.99, 0.999),
  );
  check(
    "truncated_gamma_f64",
    SimdTruncatedGamma::<f64>::new(2.0, 1.0, 1.0, 5.0),
  );
  check(
    "truncated_gamma_f32",
    SimdTruncatedGamma::<f32>::new(2.0, 1.0, 1.0, 5.0),
  );
  check(
    "truncated_gamma_inversion_f64",
    SimdTruncatedGamma::<f64>::new(2.5, 1.2, 45.0, 60.0),
  );
}

#[test]
fn complex_kernel_writes_every_element() {
  check(
    "complex_normal_f64",
    ComplexDistribution::new(
      SimdNormal::<f64>::new(0.0, 1.0),
      SimdNormal::<f64>::new(0.5, 2.0),
    ),
  );
}

/// The two-engine stream takes the ziggurats' 16-lane step and its own uniform stores.
#[cfg(feature = "unstable-dual-stream-rng")]
#[test]
fn dual_stream_kernels_write_every_element() {
  check_on::<SimdRngDual, _>("normal_dual_f64", SimdNormal::<f64>::new(0.3, 1.7));
  check_on::<SimdRngDual, _>("normal_dual_f32", SimdNormal::<f32>::new(0.3, 1.7));
  check_on::<SimdRngDual, _>("exp_dual_f64", SimdExp::<f64>::new(1.8));
  check_on::<SimdRngDual, _>("exp_dual_f32", SimdExp::<f32>::new(1.8));
  check_on::<SimdRngDual, _>("uniform_unit_dual_f64", SimdUniform::<f64>::new(0.0, 1.0));
  check_on::<SimdRngDual, _>("uniform_unit_dual_f32", SimdUniform::<f32>::new(0.0, 1.0));
  check_on::<SimdRngDual, _>("uniform_dual_f64", SimdUniform::<f64>::new(-2.0, 3.0));
}
