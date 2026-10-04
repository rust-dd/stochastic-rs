//! compile_fail checks: the sealed traits reject a downstream impl, a stream without a kernel has no bulk fill, a non-exhaustive callable needs a wildcard arm.

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

/// ```compile_fail,E0599
/// use stochastic_rs_distributions::DistributionSampler;
/// use stochastic_rs_distributions::SimdDistribution;
/// use stochastic_rs_distributions::dirichlet::SimdDirichlet;
/// use stochastic_rs_distributions::simd_rng::Unseeded;
/// let mut s = SimdDirichlet::<f64>::new(vec![1.0, 2.0]).seeded(&Unseeded);
/// s.fill_slice(&mut [0.0; 2]);
/// ```
pub struct SeededDirichletHasNoFillSlice;

/// ```compile_fail,E0004
/// use stochastic_rs_distributions::traits::Fn2D;
/// fn kind(f: &Fn2D<f64>) -> &'static str {
///   match f {
///     Fn2D::Native(_) => "closure",
///     Fn2D::Expr(_) => "program",
///     Fn2D::Grid(_) => "grid",
///   }
/// }
/// ```
pub struct Fn2DNeedsAWildcardArm;
