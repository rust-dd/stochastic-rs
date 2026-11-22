use ndarray::Array1;
use ndarray::Array2;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::exp::SimdExp;
use stochastic_rs_distributions::normal::SimdNormal;
use stochastic_rs_stochastic::autoregressive::ar::ArSampler;
use stochastic_rs_stochastic::correlation::heston_stoch_corr::HestonStochCorrSampler;
use stochastic_rs_stochastic::device::Cpu;
use stochastic_rs_stochastic::diffusion::regime_switching::RegimeSwitchingDiffusion;
use stochastic_rs_stochastic::euler::EulerBackend;
use stochastic_rs_stochastic::euler::LiftSpec;
use stochastic_rs_stochastic::interest::adg::Adg;
use stochastic_rs_stochastic::interest::cir_2f::Cir2F;
use stochastic_rs_stochastic::interest::hjm::Hjm;
use stochastic_rs_stochastic::interest::lmm::Lmm;
use stochastic_rs_stochastic::interest::wu_zhang::WuZhang;
use stochastic_rs_stochastic::isonormal::IsoNormal;
use stochastic_rs_stochastic::ito::DiffusionProcessFn;
use stochastic_rs_stochastic::ito::Function2D;
use stochastic_rs_stochastic::ito::ItoCalculator;
use stochastic_rs_stochastic::ito::ItoResult;
use stochastic_rs_stochastic::jump::bates::Bates;
use stochastic_rs_stochastic::jump::jump_fou::JumpFou;
use stochastic_rs_stochastic::jump::jump_fou_custom::JumpFouCustom;
use stochastic_rs_stochastic::jump::levy_diffusion::LevyDiffusion;
use stochastic_rs_stochastic::noise::cgns::Cgns;
use stochastic_rs_stochastic::process::ccustom::CompoundCustom;
use stochastic_rs_stochastic::process::customjt::CustomJt;
use stochastic_rs_stochastic::process::multivariate_hawkes::MultivariateHawkes;
use stochastic_rs_stochastic::process::subordinator::ctrw::Ctrw;
use stochastic_rs_stochastic::process::subordinator::ctrw::CtrwJumpLaw;
use stochastic_rs_stochastic::process::subordinator::ctrw::CtrwWaitingLaw;
use stochastic_rs_stochastic::rough::rl_fbm::RlFbm;
use stochastic_rs_stochastic::sde::NoiseModel;
use stochastic_rs_stochastic::sde::Sde;
use stochastic_rs_stochastic::sheet::fbs::FbsSampler;
use stochastic_rs_stochastic::volatility::bns::Bns;
use stochastic_rs_stochastic::volatility::double_heston::DoubleHeston;
use stochastic_rs_stochastic::volatility::fbates_svj::FbatesSvj;
use stochastic_rs_stochastic::volatility::fheston::Fheston;
use stochastic_rs_stochastic::volatility::heston::AndersenQe;
use stochastic_rs_stochastic::volatility::heston::Euler;
use stochastic_rs_stochastic::volatility::heston_hull_white_2f::HestonHullWhite2F;
use stochastic_rs_stochastic::volatility::heston2d::Heston2D;
use stochastic_rs_stochastic::volatility::hkde::Hkde;
use stochastic_rs_stochastic::volatility::multifactor_heston::MultifactorHeston;
use stochastic_rs_stochastic::volatility::multifactor_sabr::MultifactorSabr;
use stochastic_rs_stochastic::volterra::ExponentialKernel;
use stochastic_rs_stochastic::volterra::GaussianPolynomialVolatility;
use stochastic_rs_stochastic::volterra::Volterra;
use stochastic_rs_stochastic::volterra::VolterraSde;
use stochastic_rs_stochastic::volterra::VolterraSquareRoot;

fn is_debug_and_clone<T: core::fmt::Debug + Clone>() {}

fn is_debug<T: core::fmt::Debug>() {}

/// Generic over the backend, so a `Debug` or `Clone` written for `Cpu` alone fails to compile.
fn every_process_is_debug_and_clone_on<B: EulerBackend<f64> + core::fmt::Debug>() {
  is_debug_and_clone::<Adg<f64, Unseeded, B>>();
  is_debug_and_clone::<Bates<f64, SimdNormal<f64>, Unseeded, B>>();
  is_debug_and_clone::<Bns<f64, Unseeded, B>>();
  is_debug_and_clone::<Cir2F<f64, Unseeded, B>>();
  is_debug_and_clone::<CompoundCustom<f64, SimdNormal<f64>, SimdExp<f64>, Unseeded, B>>();
  is_debug_and_clone::<Ctrw<f64, Unseeded, B>>();
  is_debug_and_clone::<CustomJt<f64, SimdExp<f64>, Unseeded, B>>();
  is_debug_and_clone::<DoubleHeston<f64, Unseeded, B>>();
  is_debug_and_clone::<FbatesSvj<f64, Unseeded, B>>();
  is_debug_and_clone::<Fheston<f64, Unseeded, B>>();
  is_debug_and_clone::<GaussianPolynomialVolatility<f64, ExponentialKernel<f64>, Unseeded, B>>();
  is_debug_and_clone::<Heston2D<f64, Unseeded, B>>();
  is_debug_and_clone::<HestonHullWhite2F<f64, Euler, Unseeded, B>>();
  is_debug_and_clone::<HestonHullWhite2F<f64, AndersenQe, Unseeded, B>>();
  is_debug_and_clone::<Hjm<f64, Unseeded, B>>();
  is_debug_and_clone::<Hkde<f64, Unseeded, B>>();
  is_debug_and_clone::<JumpFou<f64, SimdNormal<f64>, Unseeded, B>>();
  is_debug_and_clone::<JumpFouCustom<f64, SimdExp<f64>, SimdNormal<f64>, Unseeded, B>>();
  is_debug_and_clone::<LevyDiffusion<f64, SimdNormal<f64>, Unseeded, B>>();
  is_debug_and_clone::<Lmm<f64, Unseeded, B>>();
  is_debug_and_clone::<MultifactorHeston<f64, 2, Unseeded, B>>();
  is_debug_and_clone::<MultifactorSabr<f64, Unseeded, B>>();
  is_debug_and_clone::<MultivariateHawkes<f64, Unseeded, B>>();
  is_debug_and_clone::<RegimeSwitchingDiffusion<f64, Unseeded, B>>();
  is_debug_and_clone::<Volterra<f64, Unseeded, B>>();
  is_debug_and_clone::<VolterraSde<f64, ExponentialKernel<f64>, Unseeded, B>>();
  is_debug_and_clone::<VolterraSquareRoot<f64, ExponentialKernel<f64>, Unseeded, B>>();
  is_debug_and_clone::<WuZhang<f64, Unseeded, B>>();
}

#[test]
fn every_process_is_debug_and_clone() {
  every_process_is_debug_and_clone_on::<Cpu>();
}

#[test]
fn the_other_public_types_are_debug() {
  type DriftFn = fn(&Array1<f64>, f64) -> Array1<f64>;
  type DiffusionFn = fn(&Array1<f64>, f64) -> Array2<f64>;
  is_debug::<AndersenQe>();
  is_debug::<ArSampler<f64>>();
  is_debug::<Cgns<f64>>();
  is_debug::<CtrwJumpLaw<f64>>();
  is_debug::<CtrwWaitingLaw<f64>>();
  is_debug::<DiffusionProcessFn>();
  is_debug::<Euler>();
  is_debug::<FbsSampler<f64, Unseeded>>();
  is_debug::<Function2D>();
  is_debug::<HestonStochCorrSampler<f64, Unseeded>>();
  is_debug::<IsoNormal<fn(usize, usize) -> f64>>();
  is_debug::<ItoCalculator>();
  is_debug::<ItoResult>();
  is_debug::<LiftSpec<'static, f64>>();
  is_debug::<NoiseModel>();
  is_debug::<RlFbm<f64>>();
  is_debug::<Sde<f64, DriftFn, DiffusionFn>>();
}
