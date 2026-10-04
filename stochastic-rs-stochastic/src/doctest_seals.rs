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
