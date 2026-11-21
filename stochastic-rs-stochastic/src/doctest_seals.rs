//! `compile_fail` checks: the sealed traits reject a downstream implementation, and a removed path
//! stays removed.

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
///   fn simulate<T: FloatExt, S: SeedExt, B>(_model: &Heston<T, Self, S, B>, _seed: &S) -> [Array1<T>; 2] {
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

/// The normal-log-jump Bates model is `jump::bates::Bates` with a `LogNormalJump` law.
///
/// ```compile_fail,E0432
/// use stochastic_rs_stochastic::volatility::bates_svj::BatesSvj;
/// ```
pub struct BatesSvjIsBatesWithALogNormalJump;

/// The LIBOR market model is `interest::lmm::Lmm`; there is no uncoupled `Bgm`.
///
/// ```compile_fail,E0432
/// use stochastic_rs_stochastic::interest::bgm::Bgm;
/// ```
pub struct BgmIsLmm;

/// GARCH estimation is `stochastic_rs_stats::garch`; the process crate estimates nothing.
///
/// ```compile_fail,E0432
/// use stochastic_rs_stochastic::autoregressive::drr_garch::DrrGarch;
/// ```
pub struct DrrGarchIsStatsGarch;

/// The Longstaff–Schwartz pricer is `stochastic_rs_quant::pricing::BermudanLsmPricer`.
///
/// ```compile_fail,E0432
/// use stochastic_rs_stochastic::mc::lsm::Lsm;
/// ```
pub struct LsmIsBermudanLsmPricer;

/// The crate root holds no default grid, spot or strike.
///
/// ```compile_fail,E0432
/// use stochastic_rs_stochastic::N;
/// ```
pub struct NoDefaultConstants;

/// A seed is part of the process, given at construction; no sampler takes one of its own.
///
/// ```compile_fail,E0599
/// use stochastic_rs_core::simd_rng::Unseeded;
/// use stochastic_rs_stochastic::noise::cgns::Cgns;
/// let _ = Cgns::<f64>::new(0.5, 16, None, Unseeded).sample_with_seed(1);
/// ```
pub struct NoSeedBypassingSampler;

/// `sample_par` is the batch path of the rough processes.
///
/// ```compile_fail,E0599
/// use stochastic_rs_core::simd_rng::Unseeded;
/// use stochastic_rs_stochastic::rough::rl_fbm::RlFbm;
/// let _ = RlFbm::<f64>::new(0.3, 64, None, None, Unseeded).sample_batch(8);
/// ```
pub struct NoRoughBatchSampler;

/// A jump-diffusion's rate is `with_lambda`'s and its law `with_jump_law`'s; no driver to swap.
///
/// ```compile_fail,E0599
/// use stochastic_rs_distributions::normal::SimdNormal;
/// use stochastic_rs_stochastic::jump::merton::Merton;
/// let m = Merton::<f64, SimdNormal<f64>>::default();
/// let _ = m.with_cpoisson(todo!());
/// ```
pub struct NoCompoundPoissonToSwap;

/// A single-precision device refuses an `f64` process at compile time.
///
/// ```compile_fail,E0277
/// use stochastic_rs_core::simd_rng::Unseeded;
/// use stochastic_rs_stochastic::device::Metal;
/// use stochastic_rs_stochastic::diffusion::gbm::Gbm;
/// use stochastic_rs_stochastic::traits::ProcessExt;
///
/// let gbm = Gbm::<f64, _>::new(0.05, 0.2, 16, 1.0, None, Unseeded);
/// let _ = gbm.on::<Metal>().sample();
/// ```
#[cfg(metal_backend)]
pub struct MetalRefusesDoublePrecision;

/// The Python class macros are internal to the crate.
///
/// ```compile_fail,E0433
/// stochastic_rs_stochastic::py_process_1d!(PyNothing, Nothing, sig: (), params: ());
/// ```
pub struct PythonMacrosAreInternal;
