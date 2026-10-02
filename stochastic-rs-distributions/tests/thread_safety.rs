//! The stateless laws and their seeded streams cross thread boundaries; the asserts are compile-time.

use stochastic_rs_core::simd_rng::SimdRng;
use stochastic_rs_distributions::Seeded;
use stochastic_rs_distributions::complex::ComplexDistribution;
use stochastic_rs_distributions::normal::SimdNormal;

fn assert_send_sync<T: Send + Sync>() {}

const _: fn() = assert_send_sync::<SimdNormal<f64>>;
const _: fn() = assert_send_sync::<SimdNormal<f32>>;
const _: fn() = assert_send_sync::<Seeded<SimdNormal<f64>>>;
const _: fn() = assert_send_sync::<Seeded<SimdNormal<f32>, SimdRng>>;
const _: fn() = assert_send_sync::<ComplexDistribution<SimdNormal<f64>>>;
const _: fn() = assert_send_sync::<Seeded<ComplexDistribution<SimdNormal<f64>>>>;
