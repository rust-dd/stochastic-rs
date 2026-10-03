//! `compile_fail` checks: the sealed traits reject a downstream implementation.

/// ```compile_fail,E0277
/// use stochastic_rs_core::simd_rng::{SeedExt, SimdRng, SimdRngExt};
///
/// #[derive(Clone)]
/// struct Mine;
///
/// impl SeedExt for Mine {
///   fn rng(&self) -> SimdRng { SimdRng::new() }
///   fn derive(&self) -> Self { Mine }
///   fn rng_ext<R: SimdRngExt>(&self) -> R { R::new() }
///   fn next_seed(&self) -> u64 { 0 }
/// }
/// ```
pub struct SeedExtIsSealed;

/// ```compile_fail,E0277
/// use core::convert::Infallible;
/// use rand::TryRng;
/// use stochastic_rs_core::simd_rng::SimdRngExt;
///
/// #[derive(Clone, Debug)]
/// struct Mine;
///
/// impl TryRng for Mine {
///   type Error = Infallible;
///   fn try_next_u32(&mut self) -> Result<u32, Infallible> { Ok(0) }
///   fn try_next_u64(&mut self) -> Result<u64, Infallible> { Ok(0) }
///   fn try_fill_bytes(&mut self, _dst: &mut [u8]) -> Result<(), Infallible> { Ok(()) }
/// }
///
/// impl SimdRngExt for Mine {
///   fn new() -> Self { Mine }
///   fn from_seed(_seed: u64) -> Self { Mine }
///   fn next_i32x8(&mut self) -> wide::i32x8 { wide::i32x8::splat(0) }
///   fn next_i32(&mut self) -> i32 { 0 }
///   fn next_f64(&mut self) -> f64 { 0.0 }
///   fn next_f32(&mut self) -> f32 { 0.0 }
///   fn fill_uniform_f64(&mut self, _out: &mut [f64]) {}
///   fn fill_uniform_f32(&mut self, _out: &mut [f32]) {}
/// }
/// ```
pub struct SimdRngExtIsSealed;
