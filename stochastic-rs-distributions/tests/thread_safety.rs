//! The stateless laws and their seeded streams cross thread boundaries; the asserts are compile-time.

use stochastic_rs_core::simd_rng::SimdRng;
use stochastic_rs_core::simd_rng::SimdRngExt;
use stochastic_rs_distributions::Seeded;
use stochastic_rs_distributions::SimdDistribution;
use stochastic_rs_distributions::beta::SimdBeta;
use stochastic_rs_distributions::cauchy::SimdCauchy;
use stochastic_rs_distributions::chi_square::SimdChiSquared;
use stochastic_rs_distributions::complex::ComplexDistribution;
use stochastic_rs_distributions::exp::SimdExp;
use stochastic_rs_distributions::gamma::SimdGamma;
use stochastic_rs_distributions::lognormal::SimdLogNormal;
use stochastic_rs_distributions::normal::SimdNormal;
use stochastic_rs_distributions::pareto::SimdPareto;
use stochastic_rs_distributions::studentt::SimdStudentT;
use stochastic_rs_distributions::uniform::SimdUniform;
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
