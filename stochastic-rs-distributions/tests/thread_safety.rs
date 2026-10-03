//! The stateless laws and their seeded streams cross thread boundaries; the asserts are compile-time.

use stochastic_rs_core::simd_rng::SimdRng;
use stochastic_rs_core::simd_rng::SimdRngExt;
use stochastic_rs_distributions::Seeded;
use stochastic_rs_distributions::SimdDistribution;
use stochastic_rs_distributions::alpha_stable::SimdAlphaStable;
use stochastic_rs_distributions::beta::SimdBeta;
use stochastic_rs_distributions::cauchy::SimdCauchy;
use stochastic_rs_distributions::chi_square::SimdChiSquared;
use stochastic_rs_distributions::complex::ComplexDistribution;
use stochastic_rs_distributions::exp::SimdExp;
use stochastic_rs_distributions::gamma::SimdGamma;
use stochastic_rs_distributions::ged::SimdGed;
use stochastic_rs_distributions::generalized_hyperbolic::SimdGeneralizedHyperbolic;
use stochastic_rs_distributions::generalized_inverse_gauss::SimdGig;
use stochastic_rs_distributions::gev::SimdGev;
use stochastic_rs_distributions::gpd::SimdGpd;
use stochastic_rs_distributions::inverse_gauss::SimdInverseGauss;
use stochastic_rs_distributions::johnson_su::SimdJohnsonSu;
use stochastic_rs_distributions::lognormal::SimdLogNormal;
use stochastic_rs_distributions::normal::SimdNormal;
use stochastic_rs_distributions::normal_inverse_gauss::SimdNormalInverseGauss;
use stochastic_rs_distributions::pareto::SimdPareto;
use stochastic_rs_distributions::skew_t::SimdSkewT;
use stochastic_rs_distributions::studentt::SimdStudentT;
use stochastic_rs_distributions::tempered_stable::SimdTemperedStable;
use stochastic_rs_distributions::uniform::SimdUniform;
use stochastic_rs_distributions::variance_gamma::SimdVarianceGamma;
use stochastic_rs_distributions::weibull::SimdWeibull;

fn assert_send_sync<T: Send + Sync>() {}

/// Checked against the trait bounds alone: every law's stream is `Send + Sync` on every engine.
fn assert_seeded_send_sync<D: SimdDistribution, R: SimdRngExt>() {
  assert_send_sync::<Seeded<D, R>>();
}

const _: fn() = assert_seeded_send_sync::<SimdNormal<f64>, SimdRng>;

const _: fn() = assert_send_sync::<SimdNormal<f64>>;
const _: fn() = assert_send_sync::<SimdNormal<f32>>;
const _: fn() = assert_send_sync::<Seeded<SimdNormal<f64>>>;
const _: fn() = assert_send_sync::<Seeded<SimdNormal<f32>, SimdRng>>;
const _: fn() = assert_send_sync::<ComplexDistribution<SimdNormal<f64>>>;
const _: fn() = assert_send_sync::<Seeded<ComplexDistribution<SimdNormal<f64>>>>;
const _: fn() = assert_send_sync::<SimdExp<f64>>;
const _: fn() = assert_send_sync::<Seeded<SimdExp<f64>>>;
const _: fn() = assert_send_sync::<SimdUniform<f64>>;
const _: fn() = assert_send_sync::<Seeded<SimdUniform<f64>>>;
const _: fn() = assert_send_sync::<SimdGamma<f64>>;
const _: fn() = assert_send_sync::<Seeded<SimdGamma<f64>>>;
const _: fn() = assert_send_sync::<SimdChiSquared<f64>>;
const _: fn() = assert_send_sync::<Seeded<SimdChiSquared<f64>>>;
const _: fn() = assert_send_sync::<SimdBeta<f64>>;
const _: fn() = assert_send_sync::<Seeded<SimdBeta<f64>>>;
const _: fn() = assert_send_sync::<SimdLogNormal<f64>>;
const _: fn() = assert_send_sync::<Seeded<SimdLogNormal<f64>>>;
const _: fn() = assert_send_sync::<SimdStudentT<f64>>;
const _: fn() = assert_send_sync::<Seeded<SimdStudentT<f64>>>;
const _: fn() = assert_send_sync::<SimdCauchy<f64>>;
const _: fn() = assert_send_sync::<Seeded<SimdCauchy<f64>>>;
const _: fn() = assert_send_sync::<SimdWeibull<f64>>;
const _: fn() = assert_send_sync::<Seeded<SimdWeibull<f64>>>;
const _: fn() = assert_send_sync::<SimdPareto<f64>>;
const _: fn() = assert_send_sync::<Seeded<SimdPareto<f64>>>;
const _: fn() = assert_send_sync::<SimdAlphaStable<f64>>;
const _: fn() = assert_send_sync::<Seeded<SimdAlphaStable<f64>>>;
const _: fn() = assert_send_sync::<SimdTemperedStable<f64>>;
const _: fn() = assert_send_sync::<Seeded<SimdTemperedStable<f64>>>;
const _: fn() = assert_send_sync::<SimdNormalInverseGauss<f64>>;
const _: fn() = assert_send_sync::<Seeded<SimdNormalInverseGauss<f64>>>;
const _: fn() = assert_send_sync::<SimdVarianceGamma<f64>>;
const _: fn() = assert_send_sync::<Seeded<SimdVarianceGamma<f64>>>;
const _: fn() = assert_send_sync::<SimdGeneralizedHyperbolic<f64>>;
const _: fn() = assert_send_sync::<Seeded<SimdGeneralizedHyperbolic<f64>>>;
const _: fn() = assert_send_sync::<SimdGig<f64>>;
const _: fn() = assert_send_sync::<Seeded<SimdGig<f64>>>;
const _: fn() = assert_send_sync::<SimdInverseGauss<f64>>;
const _: fn() = assert_send_sync::<Seeded<SimdInverseGauss<f64>>>;
const _: fn() = assert_send_sync::<SimdSkewT<f64>>;
const _: fn() = assert_send_sync::<Seeded<SimdSkewT<f64>>>;
const _: fn() = assert_send_sync::<SimdJohnsonSu<f64>>;
const _: fn() = assert_send_sync::<Seeded<SimdJohnsonSu<f64>>>;
const _: fn() = assert_send_sync::<SimdGed<f64>>;
const _: fn() = assert_send_sync::<Seeded<SimdGed<f64>>>;
const _: fn() = assert_send_sync::<SimdGev<f64>>;
const _: fn() = assert_send_sync::<Seeded<SimdGev<f64>>>;
const _: fn() = assert_send_sync::<SimdGpd<f64>>;
const _: fn() = assert_send_sync::<Seeded<SimdGpd<f64>>>;
