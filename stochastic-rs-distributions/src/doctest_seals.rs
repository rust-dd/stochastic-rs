//! `compile_fail` checks: the sealed traits reject a downstream implementation.

/// ```compile_fail,E0277
/// use stochastic_rs_distributions::DistributionSampler;
/// struct Mine;
/// impl DistributionSampler<f64> for Mine {
///   fn fill_slice(&mut self, _out: &mut [f64]) {}
///   fn fork(&mut self, _stream_idx: u64) -> Self { Mine }
/// }
/// ```
pub struct DistributionSamplerIsSealed;

/// ```compile_fail,E0277
/// use stochastic_rs_distributions::simd_rng::SeedExt;
/// use stochastic_rs_distributions::simd_rng::SimdRngExt;
/// use stochastic_rs_distributions::traits::SimdDistribution;
/// #[derive(Clone)]
/// struct Mine;
/// impl SimdDistribution for Mine {
///   type State<R: SimdRngExt> = ();
///   fn init<R: SimdRngExt, S: SeedExt>(&self, _: &S) -> ((), u64) { ((), 0) }
/// }
/// ```
pub struct SimdDistributionIsSealed;
