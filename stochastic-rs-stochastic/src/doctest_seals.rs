//! `compile_fail` checks: the sealed traits reject a downstream implementation.

/// ```compile_fail,E0277
/// use stochastic_rs_stochastic::device::{Backend, DeviceError, DeviceInfo};
/// #[derive(Clone, Copy)]
/// struct Mine;
/// impl Backend for Mine {
///   fn probe(&self) -> Result<DeviceInfo, DeviceError> { unimplemented!() }
/// }
/// ```
pub struct BackendIsSealed;

/// ```compile_fail,E0277
/// use ndarray::Array1;
/// use stochastic_rs_core::simd_rng::SeedExt;
/// use stochastic_rs_stochastic::traits::FloatExt;
/// use stochastic_rs_stochastic::volatility::heston::{Heston, HestonScheme};
/// struct Mine;
/// impl HestonScheme for Mine {
///   fn simulate<T: FloatExt, S: SeedExt, B>(_model: &Heston<T, S, Self, B>, _seed: &S) -> [Array1<T>; 2] {
///     unimplemented!()
///   }
/// }
/// ```
pub struct HestonSchemeIsSealed;

/// ```compile_fail,E0277
/// use ndarray::Array1;
/// use stochastic_rs_stochastic::traits::PathSampler;
/// struct Mine;
/// impl PathSampler<f64> for Mine {
///   type Output = Array1<f64>;
///   fn sample_into(&mut self, _out: &mut Array1<f64>) {}
///   fn sample(&mut self) -> Array1<f64> { Array1::zeros(1) }
/// }
/// ```
pub struct PathSamplerIsSealed;

/// ```compile_fail,E0277
/// use ndarray::Array1;
/// use stochastic_rs_stochastic::diffusion::ou::Ou;
/// use stochastic_rs_stochastic::traits::ProcessExt;
/// struct Mine(Ou<f64>);
/// impl ProcessExt<f64> for Mine {
///   type Output = Array1<f64>;
///   type Sampler<'a> = <Ou<f64> as ProcessExt<f64>>::Sampler<'a>;
///   fn sampler(&self) -> Self::Sampler<'_> { self.0.sampler() }
/// }
/// ```
pub struct ProcessExtIsSealed;
