use ndarray::Array1;
use ndarray::Array2;
use ndarray::array;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::exp::SimdExp;
use stochastic_rs_distributions::normal::SimdNormal;
use stochastic_rs_distributions::uniform::SimdUniform;
use stochastic_rs_stochastic::autoregressive::agarch::Agarch;
use stochastic_rs_stochastic::autoregressive::ar::Ar;
use stochastic_rs_stochastic::autoregressive::ar::ArSampler;
use stochastic_rs_stochastic::autoregressive::arima::Arima;
use stochastic_rs_stochastic::autoregressive::garch::Garch;
use stochastic_rs_stochastic::autoregressive::gjr_garch::GjrGarch;
use stochastic_rs_stochastic::autoregressive::ma::Ma;
use stochastic_rs_stochastic::autoregressive::sarima::Sarima;
use stochastic_rs_stochastic::correlation::VanEmmerich;
use stochastic_rs_stochastic::correlation::heston_stoch_corr::HestonStochCorr;
use stochastic_rs_stochastic::correlation::heston_stoch_corr::HestonStochCorrSampler;
use stochastic_rs_stochastic::correlation::stochastic_correlated_bm;
use stochastic_rs_stochastic::device::Cpu;
use stochastic_rs_stochastic::diffusion::bessel::Bessel;
use stochastic_rs_stochastic::diffusion::bessel::SquaredBessel;
use stochastic_rs_stochastic::diffusion::cev::Cev;
use stochastic_rs_stochastic::diffusion::cfou::Cfou;
use stochastic_rs_stochastic::diffusion::cir::Cir;
use stochastic_rs_stochastic::diffusion::displaced_diffusion::DisplacedDiffusion;
use stochastic_rs_stochastic::diffusion::fcir::Fcir;
use stochastic_rs_stochastic::diffusion::fgbm::Fgbm;
use stochastic_rs_stochastic::diffusion::fjacobi::Fjacobi;
use stochastic_rs_stochastic::diffusion::fou::Fou;
use stochastic_rs_stochastic::diffusion::gbm::Gbm;
use stochastic_rs_stochastic::diffusion::gbm_log::GbmLog;
use stochastic_rs_stochastic::diffusion::multi_gbm::MultiGbm;
use stochastic_rs_stochastic::diffusion::ou::Ou;
use stochastic_rs_stochastic::diffusion::regime_switching::RegimeSwitchingDiffusion;
use stochastic_rs_stochastic::diffusion::wishart::Wishart;
use stochastic_rs_stochastic::euler::EulerBackend;
use stochastic_rs_stochastic::euler::EulerCoefficients;
use stochastic_rs_stochastic::euler::EulerSystem;
use stochastic_rs_stochastic::euler::LiftSpec;
use stochastic_rs_stochastic::interest::adg::Adg;
use stochastic_rs_stochastic::interest::black_karasinski::BlackKarasinski;
use stochastic_rs_stochastic::interest::cir_2f::Cir2F;
use stochastic_rs_stochastic::interest::cir_pp::CirPlusPlus;
use stochastic_rs_stochastic::interest::duffie_kan::DuffieKan;
use stochastic_rs_stochastic::interest::duffie_kan_jump_exp::DuffieKanJumpExp;
use stochastic_rs_stochastic::interest::hjm::Hjm;
use stochastic_rs_stochastic::interest::hull_white::HullWhite;
use stochastic_rs_stochastic::interest::lmm::Lmm;
use stochastic_rs_stochastic::interest::wu_zhang::WuZhang;
use stochastic_rs_stochastic::isonormal::IsoNormal;
use stochastic_rs_stochastic::ito::DiffusionProcessFn;
use stochastic_rs_stochastic::ito::Function2D;
use stochastic_rs_stochastic::ito::ItoCalculator;
use stochastic_rs_stochastic::ito::ItoResult;
use stochastic_rs_stochastic::jump::bates::Bates;
use stochastic_rs_stochastic::jump::bates::LogNormalJump;
use stochastic_rs_stochastic::jump::bilateral_gamma::BilateralGamma;
use stochastic_rs_stochastic::jump::bilateral_gamma::BilateralGammaMotion;
use stochastic_rs_stochastic::jump::cts::Cts;
use stochastic_rs_stochastic::jump::jump_fou::JumpFou;
use stochastic_rs_stochastic::jump::jump_fou_custom::JumpFouCustom;
use stochastic_rs_stochastic::jump::kou::Kou;
use stochastic_rs_stochastic::jump::levy_diffusion::LevyDiffusion;
use stochastic_rs_stochastic::jump::merton::Merton;
use stochastic_rs_stochastic::jump::mjd_log::MjdLog;
use stochastic_rs_stochastic::jump::nig::Nig;
use stochastic_rs_stochastic::jump::rdts::Rdts;
use stochastic_rs_stochastic::jump::vg::Vg;
use stochastic_rs_stochastic::noise::cfgns::Cfgns;
use stochastic_rs_stochastic::noise::cgns::Cgns;
use stochastic_rs_stochastic::noise::fgn::Fgn;
use stochastic_rs_stochastic::noise::gn::Gn;
use stochastic_rs_stochastic::noise::wn::Wn;
use stochastic_rs_stochastic::params::Drift;
use stochastic_rs_stochastic::params::Horizon;
use stochastic_rs_stochastic::params::VarianceBoundary;
use stochastic_rs_stochastic::process::bm::Bm;
use stochastic_rs_stochastic::process::brownian_bridge::BrownianBridge;
use stochastic_rs_stochastic::process::ccustom::CompoundCustom;
use stochastic_rs_stochastic::process::cfbms::Cfbms;
use stochastic_rs_stochastic::process::customjt::CustomJt;
use stochastic_rs_stochastic::process::hawkes::Hawkes;
use stochastic_rs_stochastic::process::lfsm::Lfsm;
use stochastic_rs_stochastic::process::multivariate_hawkes::MultivariateHawkes;
use stochastic_rs_stochastic::process::poisson::Poisson;
use stochastic_rs_stochastic::process::subordinator::ctrw::Ctrw;
use stochastic_rs_stochastic::process::subordinator::ctrw::CtrwJumpLaw;
use stochastic_rs_stochastic::process::subordinator::ctrw::CtrwWaitingLaw;
use stochastic_rs_stochastic::process::subordinator::gamma_subordinator::GammaSubordinator;
use stochastic_rs_stochastic::process::subordinator::ig_subordinator::IgSubordinator;
use stochastic_rs_stochastic::rough::rl_fbm::RlFbm;
use stochastic_rs_stochastic::sde::NoiseModel;
use stochastic_rs_stochastic::sde::Sde;
use stochastic_rs_stochastic::sde::SdeMethod;
use stochastic_rs_stochastic::sheet::fbs::Fbs;
use stochastic_rs_stochastic::sheet::fbs::FbsSampler;
use stochastic_rs_stochastic::traits::ProcessExt;
use stochastic_rs_stochastic::volatility::HestonPow;
use stochastic_rs_stochastic::volatility::bergomi::Bergomi;
use stochastic_rs_stochastic::volatility::bns::Bns;
use stochastic_rs_stochastic::volatility::double_heston::DoubleHeston;
use stochastic_rs_stochastic::volatility::fbates_svj::FbatesSvj;
use stochastic_rs_stochastic::volatility::fheston::Fheston;
use stochastic_rs_stochastic::volatility::heston::AndersenQe;
use stochastic_rs_stochastic::volatility::heston::Euler;
use stochastic_rs_stochastic::volatility::heston::Heston;
use stochastic_rs_stochastic::volatility::heston_log::HestonLog;
use stochastic_rs_stochastic::volatility::heston_slv::HestonSlv;
use stochastic_rs_stochastic::volatility::heston2d::Heston2D;
use stochastic_rs_stochastic::volatility::hkde::Hkde;
use stochastic_rs_stochastic::volatility::multifactor_heston::MultifactorHeston;
use stochastic_rs_stochastic::volatility::multifactor_sabr::MultifactorSabr;
use stochastic_rs_stochastic::volatility::rough_bergomi::RoughBergomi;
use stochastic_rs_stochastic::volatility::sabr::Sabr;
use stochastic_rs_stochastic::volterra::ExponentialKernel;
use stochastic_rs_stochastic::volterra::GaussianPolynomialVolatility;
use stochastic_rs_stochastic::volterra::Volterra;
use stochastic_rs_stochastic::volterra::VolterraKernelSpec;
use stochastic_rs_stochastic::volterra::VolterraSde;
use stochastic_rs_stochastic::volterra::VolterraSquareRoot;

#[test]
#[should_panic(expected = "rho must satisfy `")]
fn a_correlated_fgn_pair_names_the_predicate_it_broke() {
  let _ = Cfgns::<f64>::new(0.7, 1.5, 16, None, Unseeded);
}

#[test]
#[should_panic(
  expected = "rho_path[1] must satisfy `(-1.0..=1.0).contains(&rho_path[i])`, got rho_path[1] = 1.2"
)]
fn correlated_brownian_motions_refuse_a_correlation_outside_the_unit_interval() {
  let _ = stochastic_correlated_bm(&array![0.2, 1.2, 0.1], None, &Unseeded);
}

#[test]
#[should_panic(
  expected = "rho_path[0] must satisfy `(-1.0..=1.0).contains(&rho_path[i])`, got rho_path[0] = NaN"
)]
fn correlated_brownian_motions_refuse_a_nan_correlation() {
  let _ = stochastic_correlated_bm(&array![f64::NAN, 0.1], None, &Unseeded);
}

fn solve_over(t0: f64, t1: f64, steps: usize) -> ndarray::Array3<f64> {
  Sde::new(
    |_: &Array1<f64>, _: f64| array![0.0],
    |_: &Array1<f64>, _: f64| Array2::<f64>::zeros((1, 1)),
    NoiseModel::Gaussian,
    None,
  )
  .solve(&array![0.0], t0, t1, steps, 1, SdeMethod::Euler, &Unseeded)
}

macro_rules! refuses {
  ($($name:ident: $message:literal => $built:expr;)*) => {
    $(
      #[test]
      #[should_panic(expected = $message)]
      fn $name() {
        let _ = $built;
      }
    )*
  };
}

refuses! {
  an_sde_refuses_a_reversed_interval: "t1 must satisfy `t1 > t0 && (t1 - t0).is_finite()`, got t0 = 1.0, t1 = 0.0"
    => solve_over(1.0, 0.0, 8);
  an_sde_refuses_a_nan_bound: "t1 must satisfy `t1 > t0 && (t1 - t0).is_finite()`, got t0 = 0.0, t1 = NaN"
    => solve_over(0.0, f64::NAN, 8);
  an_sde_refuses_no_steps: "steps must satisfy `steps >= 1`, got steps = 0"
    => solve_over(0.0, 1.0, 0);
  a_nan_parameter_is_refused_at_construction: "sigma must satisfy `sigma.is_finite()`, got sigma = NaN"
    => Ou::<f64>::new(1.0, 0.0, f64::NAN, 64, None, None, Unseeded);
  a_grid_of_one_point_is_refused: "n must satisfy `n >= 2`, got n = 1"
    => Ou::<f64>::new(1.0, 0.0, 0.2, 1, None, None, Unseeded);
  a_noise_of_no_increments_is_refused: "n must satisfy `n >= 1`, got n = 0"
    => Gn::<f64>::new(0, None, Unseeded);
  a_zero_horizon_is_refused: "t must satisfy `t.is_none_or(|t| t > T::zero() && t.is_finite())`, got t = Some(0.0)"
    => Gbm::<f64>::new(0.05, 0.2, 64, 1.0, Some(0.0), Unseeded);
  a_negative_horizon_is_refused: "t must satisfy `t.is_none_or(|t| t > T::zero() && t.is_finite())`, got t = Some(-1.0)"
    => Ou::<f64>::new(1.0, 0.0, 0.2, 64, None, Some(-1.0), Unseeded);
  gbm_refuses_a_negative_volatility: "sigma must satisfy `sigma >= T::zero()`, got sigma = -0.2"
    => Gbm::<f64>::new(0.05, -0.2, 64, 1.0, None, Unseeded);
  gbm_refuses_a_non_positive_start: "x0 must satisfy `x0 > T::zero()`, got x0 = 0.0"
    => Gbm::<f64>::new(0.05, 0.2, 64, 0.0, None, Unseeded);
  the_tempered_stable_constant_has_a_pole_at_alpha_one: "alpha must satisfy `alpha != T::one()`, got alpha = 1.0"
    => Cts::<f64>::new(2.0, 3.0, 1.0, 64, 8, None, None, Unseeded);
  the_rapidly_decreasing_drift_has_a_pole_at_alpha_one: "alpha must satisfy `alpha != T::one()`, got alpha = 1.0"
    => Rdts::<f64>::new(2.0, 3.0, 1.0, 64, 8, None, None, Unseeded);
  a_volterra_kernel_validates_its_parameter: "h must satisfy `h > T::zero() && h < T::one()`, got h = 1.5"
    => Volterra::<f64>::new(VolterraKernelSpec::FractionalBM { h: 1.5 }, 64, None, Unseeded);
  a_power_law_kernel_is_square_integrable: "gamma must satisfy `gamma.is_finite() && gamma > T::from_f64_fast(-0.5)`, got gamma = -0.5"
    => Volterra::<f64>::new(VolterraKernelSpec::PowerLaw { gamma: -0.5 }, 64, None, Unseeded);
  a_power_law_kernel_has_a_finite_exponent: "gamma must satisfy `gamma.is_finite() && gamma > T::from_f64_fast(-0.5)`, got gamma = inf"
    => Volterra::<f64>::new(VolterraKernelSpec::PowerLaw { gamma: f64::INFINITY }, 64, None, Unseeded);
  an_exponential_kernel_decays: "beta must satisfy `beta.is_finite() && beta > T::zero()`, got beta = 0.0"
    => Volterra::<f64>::new(VolterraKernelSpec::Exponential { beta: 0.0 }, 64, None, Unseeded);
  an_exponential_kernel_has_a_finite_rate: "beta must satisfy `beta.is_finite() && beta > T::zero()`, got beta = inf"
    => Volterra::<f64>::new(VolterraKernelSpec::Exponential { beta: f64::INFINITY }, 64, None, Unseeded);

  a_discrete_series_refuses_no_observations: "n must satisfy `n >= 1`, got n = 0"
    => Ar::<f64>::new(array![0.5], 1.0, 0, None, Unseeded);
  an_autoregression_refuses_a_nan_coefficient: "phi[0] must satisfy `phi[i].is_finite()`, got phi[0] = NaN"
    => Ar::<f64>::new(array![f64::NAN], 1.0, 64, None, Unseeded);
  a_correlation_process_refuses_a_nan_level: "mu must satisfy `mu.is_finite()`, got mu = NaN"
    => VanEmmerich::<f64>::new(1.0, f64::NAN, 0.2, 0.0, 64, None, Unseeded);
  correlated_assets_refuse_a_nan_drift: "mu[1] must satisfy `mu[i].is_finite()`, got mu[1] = NaN"
    => MultiGbm::<f64>::new(array![0.0, f64::NAN], array![0.2, 0.2], array![[1.0, 0.0], [0.0, 1.0]], 64, array![1.0, 1.0], None, Unseeded);
  correlated_assets_refuse_one_point: "n must satisfy `n >= 2`, got n = 1"
    => MultiGbm::<f64>::new(array![0.0, 0.0], array![0.2, 0.2], array![[1.0, 0.0], [0.0, 1.0]], 1, array![1.0, 1.0], None, Unseeded);
  a_matrix_diffusion_refuses_one_point: "n must satisfy `n >= 2`, got n = 1"
    => Wishart::<f64>::new(2.5, Array2::zeros((2, 2)), Array2::eye(2), Array2::eye(2), 1, None, Unseeded);
  a_rate_model_refuses_a_nan_entry: "alpha[0] must satisfy `alpha[i].is_finite()`, got alpha[0] = NaN"
    => WuZhang::<f64>::new(array![f64::NAN], array![0.1], array![0.1], array![0.1], array![0.1], array![0.1], 1, None, 64, Unseeded);
  an_excitation_matrix_names_its_nan_entry: "alpha[(1, 0)] must satisfy `alpha[(i, j)].is_finite()`, got alpha[(1, 0)] = NaN"
    => MultivariateHawkes::<f64>::new(array![1.0, 1.0], array![[0.1, 0.1], [f64::NAN, 0.1]], array![[1.0, 1.0], [1.0, 1.0]], Horizon::Until(1.0), Unseeded);
  a_fractional_jump_process_refuses_a_nan_volatility: "sigma must satisfy `sigma.is_finite()`, got sigma = NaN"
    => JumpFou::<f64, SimdNormal<f64>>::new(0.7, 1.0, 0.0, f64::NAN, 2.0, SimdNormal::new(0.0, 0.1), 64, None, None, Unseeded);
  a_custom_jump_process_refuses_a_nan_level: "mu must satisfy `mu.is_finite()`, got mu = NaN"
    => JumpFouCustom::<f64, SimdNormal<f64>, SimdNormal<f64>>::new(0.7, 1.0, f64::NAN, 0.2, 64, None, None, SimdNormal::new(0.5, 0.01), SimdNormal::new(0.0, 0.1), Unseeded);
  a_fractional_noise_refuses_a_nan_horizon: "t must satisfy `t.is_none_or(|t| t > T::zero() && t.is_finite())`, got t = Some(NaN)"
    => Fgn::<f64>::new(0.7, 64, Some(f64::NAN), Unseeded);
  a_subordinator_refuses_a_nan_rate: "rate must satisfy `rate.is_finite()`, got rate = NaN"
    => GammaSubordinator::<f64>::new(1.0, f64::NAN, 64, None, None, Unseeded);
  a_rough_process_refuses_a_nan_hurst: "hurst must satisfy `hurst.is_finite()`, got hurst = NaN"
    => RlFbm::<f64>::new(f64::NAN, 64, None, None, Unseeded);
  a_fractional_sheet_refuses_a_nan_hurst: "hurst must satisfy `hurst.is_finite()`, got hurst = NaN"
    => Fbs::<f64>::new(f64::NAN, 16, 16, 1.0, Unseeded);
  a_volatility_model_refuses_a_nan_correlation: "rho must satisfy `rho.is_finite()`, got rho = NaN"
    => Heston::<f64>::new(100.0, 0.04, 1.5, 0.04, 0.3, f64::NAN, 0.05, 64, None, HestonPow::Sqrt, VarianceBoundary::Floor, Unseeded);
  a_volterra_kernel_refuses_a_nan_weight: "c must satisfy `c.is_finite()`, got c = NaN"
    => ExponentialKernel::<f64>::new(1.0, f64::NAN);
  a_volterra_volatility_refuses_a_nan_level: "theta must satisfy `theta.is_finite()`, got theta = NaN"
    => VolterraSquareRoot::<f64, _>::new(ExponentialKernel::new(1.0, 1.0), 1.0, f64::NAN, 0.3, 64, 0.04, None, Unseeded);

  an_inverse_gaussian_process_refuses_one_point: "n must satisfy `n >= 2`, got n = 1"
    => IgSubordinator::<f64>::new(1.0, 1.0, 1, Some(3.5), Some(1.0), Unseeded);
  a_normal_inverse_gaussian_process_refuses_one_point: "n must satisfy `n >= 2`, got n = 1"
    => Nig::<f64>::new(0.1, 0.2, 0.3, 1, Some(4.0), Some(1.0), Unseeded);
  a_variance_gamma_process_refuses_one_point: "n must satisfy `n >= 2`, got n = 1"
    => Vg::<f64>::new(0.1, 0.2, 0.3, 1, Some(2.5), Some(1.0), Unseeded);
  a_bilateral_gamma_process_refuses_one_point: "n must satisfy `n >= 2`, got n = 1"
    => BilateralGamma::<f64>::new(1.0, 2.0, 1.5, 2.5, 1, Some(3.0), Some(1.0), Unseeded);
  a_bilateral_gamma_motion_refuses_one_point: "n must satisfy `n >= 2`, got n = 1"
    => BilateralGammaMotion::<f64>::new(0.2, 1.0, 2.0, 1.5, 2.5, 1, Some(3.0), Some(1.0), Unseeded);
  a_multifactor_sabr_refuses_one_point: "n must satisfy `n >= 2`, got n = 1"
    => MultifactorSabr::<f64>::new(Some(1.0), Some(0.2), vec![], vec![0.5], vec![-0.3], vec![0.4], 1, Some(1.0), Unseeded);
  a_linear_fractional_stable_motion_refuses_one_point: "n must satisfy `n >= 2`, got n = 1"
    => Lfsm::<f64>::new(1.5, 0.0, 0.75, 1.0, 1, Some(0.0), Some(1.0), Unseeded);
  a_volterra_process_refuses_one_point: "n must satisfy `n >= 2`, got n = 1"
    => Volterra::<f64>::new(VolterraKernelSpec::Exponential { beta: 1.0 }, 1, Some(1.0), Unseeded);

  a_counting_process_refuses_a_zero_horizon: "t_max must satisfy `t_max > T::zero() && t_max.is_finite()`, got t_max = 0.0"
    => Poisson::<f64>::new(2.0, Horizon::Until(0.0), Unseeded);
  a_self_exciting_process_refuses_a_negative_horizon: "t_max must satisfy `t_max > T::zero() && t_max.is_finite()`, got t_max = -1.0"
    => Hawkes::<f64>::new(0.5, 0.3, 1.0, Horizon::Until(-1.0), Unseeded);
  a_renewal_process_refuses_an_infinite_horizon: "t_max must satisfy `t_max > T::zero() && t_max.is_finite()`, got t_max = inf"
    => CustomJt::<f64, _>::new(Horizon::Until(f64::INFINITY), SimdExp::new(1.0), Unseeded);
  a_renewal_process_refuses_a_nan_horizon: "t_max must satisfy `t_max > T::zero() && t_max.is_finite()`, got t_max = NaN"
    => CustomJt::<f64, _>::new(Horizon::Until(f64::NAN), SimdExp::new(1.0), Unseeded);
  a_multivariate_hawkes_refuses_a_zero_horizon: "t_max must satisfy `t_max > T::zero() && t_max.is_finite()`, got t_max = 0.0"
    => MultivariateHawkes::<f64>::new(array![1.0], array![[0.1]], array![[1.0]], Horizon::Until(0.0), Unseeded);
  hawkes_refuses_a_zero_baseline_at_construction: "mu must satisfy `mu > T::zero()`, got mu = 0.0"
    => Hawkes::<f64>::new(0.0, 0.3, 1.0, Horizon::Events(8), Unseeded);
  hawkes_refuses_a_negative_excitation_at_construction: "alpha must satisfy `alpha >= T::zero()`, got alpha = -0.1"
    => Hawkes::<f64>::new(0.5, -0.1, 1.0, Horizon::Events(8), Unseeded);
  hawkes_refuses_a_zero_decay_at_construction: "beta must satisfy `beta > T::zero()`, got beta = 0.0"
    => Hawkes::<f64>::new(0.5, 0.3, 0.0, Horizon::Events(8), Unseeded);
  hawkes_refuses_a_supercritical_branching_ratio_at_construction: "alpha must satisfy `alpha < beta`, got alpha = 2.0, beta = 1.0"
    => Hawkes::<f64>::new(0.5, 2.0, 1.0, Horizon::Events(8), Unseeded);
  the_hawkes_horizon_setter_checks_its_time: "t_max must satisfy `t_max > T::zero() && t_max.is_finite()`, got t_max = 0.0"
    => Hawkes::<f64>::new(0.5, 0.3, 1.0, Horizon::Events(8), Unseeded).with_horizon(Horizon::Until(0.0));
  the_renewal_horizon_setter_checks_its_time: "t_max must satisfy `t_max > T::zero() && t_max.is_finite()`, got t_max = -2.0"
    => CustomJt::<f64, _>::new(Horizon::Events(8), SimdExp::new(1.0), Unseeded).with_horizon(Horizon::Until(-2.0));
  the_compound_renewal_horizon_setter_checks_its_time: "t_max must satisfy `t_max > T::zero() && t_max.is_finite()`, got t_max = inf"
    => CompoundCustom::<f64, _, _>::new(SimdNormal::new(0.0, 1.0), CustomJt::new(Horizon::Events(8), SimdExp::new(1.0), Unseeded), Unseeded).with_horizon(Horizon::Until(f64::INFINITY));
  the_multivariate_hawkes_horizon_setter_checks_its_time: "t_max must satisfy `t_max > T::zero() && t_max.is_finite()`, got t_max = NaN"
    => MultivariateHawkes::<f64>::new(array![1.0], array![[0.1]], array![[1.0]], Horizon::Events(8), Unseeded).with_horizon(Horizon::Until(f64::NAN));

  a_log_price_model_refuses_a_zero_spot: "s0 must satisfy `s0 > T::zero()`, got s0 = 0.0"
    => GbmLog::<f64>::new(Drift::Mu(0.05), 0.2, 64, 0.0, None, Unseeded);
  a_log_heston_refuses_a_negative_spot: "s0 must satisfy `s0 > T::zero()`, got s0 = -1.0"
    => HestonLog::<f64>::new(Drift::Mu(0.05), 1.5, 0.04, 0.3, -0.7, 64, -1.0, 0.04, None, VarianceBoundary::Floor, Unseeded);
  a_log_jump_diffusion_refuses_a_zero_spot: "s0 must satisfy `s0 > T::zero()`, got s0 = 0.0"
    => MjdLog::<f64>::new(Drift::Mu(0.05), 0.2, 1.0, 0.0, 0.1, 64, 0.0, None, Unseeded);
  a_double_exponential_heston_refuses_a_zero_spot: "s0 must satisfy `s0 > T::zero()`, got s0 = 0.0"
    => Hkde::<f64>::new(0.05, 1.5, 0.04, 0.3, -0.7, 0.04, 0.5, 0.4, 5.0, 5.0, 64, 0.0, None, VarianceBoundary::Floor, Unseeded);
  the_log_heston_spot_setter_checks_its_value: "s0 must satisfy `s0 > T::zero()`, got s0 = 0.0"
    => HestonLog::<f64>::new(Drift::Mu(0.05), 1.5, 0.04, 0.3, -0.7, 64, 100.0, 0.04, None, VarianceBoundary::Floor, Unseeded).with_s0(0.0);
  the_double_exponential_heston_spot_setter_checks_its_value: "s0 must satisfy `s0 > T::zero()`, got s0 = 0.0"
    => Hkde::<f64>::new(0.05, 1.5, 0.04, 0.3, -0.7, 0.04, 0.5, 0.4, 5.0, 5.0, 64, 100.0, None, VarianceBoundary::Floor, Unseeded).with_s0(0.0);
  a_bates_model_refuses_a_negative_initial_variance: "v0 must satisfy `v0 >= T::zero()`, got v0 = -0.01"
    => Bates::<f64>::new(Drift::Mu(0.05), 0.5, 0.05, 0.04, 1.5, 0.3, -0.6, SimdNormal::new(0.0, 0.1), 64, 100.0, -0.01, None, VarianceBoundary::Floor, Unseeded);
  a_double_exponential_heston_refuses_a_negative_initial_variance: "v0 must satisfy `v0 >= T::zero()`, got v0 = -0.01"
    => Hkde::<f64>::new(0.05, 1.5, 0.04, 0.3, -0.7, -0.01, 0.5, 0.4, 5.0, 5.0, 64, 100.0, None, VarianceBoundary::Floor, Unseeded);
  the_bates_variance_setter_checks_its_value: "v0 must satisfy `v0 >= T::zero()`, got v0 = -0.01"
    => Bates::<f64>::new(Drift::Mu(0.05), 0.5, 0.05, 0.04, 1.5, 0.3, -0.6, SimdNormal::new(0.0, 0.1), 64, 100.0, 0.04, None, VarianceBoundary::Floor, Unseeded).with_v0(-0.01);
  the_double_exponential_heston_variance_setter_checks_its_value: "v0 must satisfy `v0 >= T::zero()`, got v0 = -0.01"
    => Hkde::<f64>::new(0.05, 1.5, 0.04, 0.3, -0.7, 0.04, 0.5, 0.4, 5.0, 5.0, 64, 100.0, None, VarianceBoundary::Floor, Unseeded).with_v0(-0.01);

  a_bates_model_refuses_a_negative_intensity: "lambda must satisfy `lambda >= T::zero()`, got lambda = -0.5"
    => Bates::<f64>::new(Drift::Mu(0.05), -0.5, 0.05, 0.04, 1.5, 0.3, -0.6, SimdNormal::new(0.0, 0.1), 64, 100.0, 0.04, None, VarianceBoundary::Floor, Unseeded);
  the_bates_intensity_setter_checks_its_value: "lambda must satisfy `lambda >= T::zero()`, got lambda = -0.5"
    => Bates::<f64>::new(Drift::Mu(0.05), 0.5, 0.05, 0.04, 1.5, 0.3, -0.6, SimdNormal::new(0.0, 0.1), 64, 100.0, 0.04, None, VarianceBoundary::Floor, Unseeded).with_lambda(-0.5);
  a_merton_model_refuses_a_negative_intensity: "lambda must satisfy `lambda >= T::zero()`, got lambda = -0.5"
    => Merton::<f64, SimdNormal<f64>>::new(0.05, 0.2, -0.5, 0.0, SimdNormal::new(0.0, 0.1), 64, None, None, Unseeded);
  the_merton_intensity_setter_checks_its_value: "lambda must satisfy `lambda >= T::zero()`, got lambda = -0.5"
    => Merton::<f64, SimdNormal<f64>>::default().with_lambda(-0.5);
  a_kou_model_refuses_a_negative_intensity: "lambda must satisfy `lambda >= T::zero()`, got lambda = -0.5"
    => Kou::<f64, SimdNormal<f64>>::new(0.05, 0.2, -0.5, 0.0, SimdNormal::new(0.0, 0.1), 64, None, None, Unseeded);
  a_levy_diffusion_refuses_a_negative_intensity: "lambda must satisfy `lambda >= T::zero()`, got lambda = -0.5"
    => LevyDiffusion::<f64, SimdNormal<f64>>::new(0.05, 0.2, -0.5, SimdNormal::new(0.0, 0.1), 64, None, None, Unseeded);
  a_fractional_jump_process_refuses_a_negative_intensity: "lambda must satisfy `lambda >= T::zero()`, got lambda = -0.5"
    => JumpFou::<f64, SimdNormal<f64>>::new(0.7, 1.0, 0.0, 0.2, -0.5, SimdNormal::new(0.0, 0.1), 64, None, None, Unseeded);
  the_fractional_jump_intensity_setter_checks_its_value: "lambda must satisfy `lambda >= T::zero()`, got lambda = -0.5"
    => JumpFou::<f64, SimdNormal<f64>>::new(0.7, 1.0, 0.0, 0.2, 2.0, SimdNormal::new(0.0, 0.1), 64, None, None, Unseeded).with_lambda(-0.5);
  a_counting_process_refuses_a_negative_rate: "lambda must satisfy `lambda >= T::zero()`, got lambda = -1.0"
    => Poisson::<f64>::new(-1.0, Horizon::Until(1.0), Unseeded);
  a_counting_process_at_a_zero_rate_refuses_an_arrival: "horizon must satisfy `lambda > T::zero() || !matches!(horizon, Horizon::Events(1..))`, got horizon = Events(3), lambda = 0.0"
    => Poisson::<f64>::new(0.0, Horizon::Events(3), Unseeded);
  the_counting_rate_setter_checks_the_horizon: "horizon must satisfy `lambda > T::zero() || !matches!(horizon, Horizon::Events(1..))`, got horizon = Events(251), lambda = 0.0"
    => Poisson::<f64>::default().with_lambda(0.0);
  the_counting_horizon_setter_checks_the_rate: "horizon must satisfy `lambda > T::zero() || !matches!(horizon, Horizon::Events(1..))`, got horizon = Events(2), lambda = 0.0"
    => Poisson::<f64>::new(0.0, Horizon::Until(1.0), Unseeded).with_horizon(Horizon::Events(2));
  a_lognormal_jump_refuses_a_negative_deviation: "omega must satisfy `omega >= T::zero()`, got omega = -0.1"
    => LogNormalJump::<f64>::new(-0.05, -0.1);
  a_garch_without_a_finite_unconditional_variance_is_refused: "alpha must satisfy `1 - alpha.sum() - beta.sum() > 0`, got alpha = [0.5], beta = [0.5]"
    => Garch::<f64>::new(0.1, array![0.5], array![0.5], 64, Unseeded);
  an_asymmetric_garch_without_a_finite_unconditional_variance_is_refused: "alpha must satisfy `1 - alpha.sum() - 0.5 * delta.sum() - beta.sum() > 0`, got alpha = [0.3], delta = [0.4], beta = [0.5]"
    => Agarch::<f64>::new(0.1, array![0.3], array![0.4], array![0.5], 64, Unseeded);
  a_threshold_garch_without_a_finite_unconditional_variance_is_refused: "alpha must satisfy `1 - alpha.sum() - 0.5 * gamma.sum() - beta.sum() > 0`, got alpha = [0.3], gamma = [0.4], beta = [0.5]"
    => GjrGarch::<f64>::new(0.1, array![0.3], array![0.4], array![0.5], 64, Unseeded);
  a_random_walk_refuses_a_zero_waiting_rate: "rate must satisfy `rate > T::zero() && rate.is_finite()`, got rate = 0.0"
    => Ctrw::<f64>::new(CtrwWaitingLaw::Exponential { rate: 0.0 }, CtrwJumpLaw::Normal { mean: 0.0, std: 1.0 }, 64, None, None, Unseeded);
  a_random_walk_refuses_a_stable_jump_index_above_two: "alpha must satisfy `alpha > T::zero() && alpha <= T::from_usize_(2)`, got alpha = 2.5"
    => Ctrw::<f64>::new(CtrwWaitingLaw::Exponential { rate: 1.0 }, CtrwJumpLaw::SymmetricStable { alpha: 2.5, scale: 1.0 }, 64, None, None, Unseeded);

  a_heston_grid_setter_refuses_no_points: "n must satisfy `n >= 2`, got n = 0"
    => Heston::<f64>::default().with_points(0);
  a_sabr_grid_setter_refuses_no_points: "n must satisfy `n >= 2`, got n = 0"
    => Sabr::<f64>::default().with_points(0);
  a_bates_grid_setter_refuses_no_points: "n must satisfy `n >= 2`, got n = 0"
    => Bates::<f64, _>::new(Drift::Mu(0.05), 0.5, 0.05, 0.04, 1.5, 0.3, -0.7, SimdNormal::new(0.0, 1.0), 64, 100.0, 0.04, Some(1.0), VarianceBoundary::Floor, Unseeded).with_points(0);
  a_bergomi_grid_setter_refuses_no_points: "n must satisfy `n >= 2`, got n = 0"
    => Bergomi::<f64>::default().with_points(0);
  a_rough_bergomi_grid_setter_refuses_no_points: "n must satisfy `n >= 2`, got n = 0"
    => RoughBergomi::<f64>::default().with_points(0);
  a_double_heston_grid_setter_refuses_no_points: "n must satisfy `n >= 2`, got n = 0"
    => DoubleHeston::<f64>::new(100.0, 0.02, 0.02, 3.0, 0.02, 0.4, -0.6, 0.5, 0.02, 0.2, -0.3, 0.05, 64, Some(1.0), VarianceBoundary::Reflect, Unseeded).with_points(0);
  a_duffie_kan_grid_setter_refuses_no_points: "n must satisfy `n >= 2`, got n = 0"
    => DuffieKan::<f64>::new(0.5, 0.04, 0.5, -0.3, 0.01, 0.0, 0.0, 0.01, 0.0, 0.5, 0.0, 0.005, 64, Some(0.05), Some(0.05), Some(1.0), Unseeded).with_points(0);
  a_jump_duffie_kan_grid_setter_refuses_no_points: "n must satisfy `n >= 2`, got n = 0"
    => DuffieKanJumpExp::<f64>::new(0.5, 0.04, 0.5, -0.3, 0.01, 0.0, 0.0, 0.01, 0.0, 0.5, 0.0, 0.005, 0.5, 0.01, 64, Some(0.05), Some(0.05), Some(1.0), Unseeded).with_points(0);
  a_grid_setter_refuses_one_point: "n must satisfy `n >= 2`, got n = 1"
    => Ou::<f64>::default().with_points(1);
  a_noise_setter_refuses_no_increments: "n must satisfy `n >= 1`, got n = 0"
    => Gn::<f64>::default().with_increments(0);
  a_horizon_setter_refuses_a_zero_horizon: "t must satisfy `t.is_none_or(|t| t > T::zero() && t.is_finite())`, got t = Some(0.0)"
    => Bm::<f64>::default().with_horizon(Some(0.0));
  a_float_setter_refuses_nan: "sigma must satisfy `sigma.is_finite()`, got sigma = NaN"
    => Cir::<f64>::default().with_sigma(f64::NAN);
  an_optional_setter_refuses_nan: "x0 must satisfy `x0.is_none_or(|v| v.is_finite())`, got x0 = Some(NaN)"
    => Ou::<f64>::default().with_x0(Some(f64::NAN));
  the_gbm_start_setter_checks_positivity: "x0 must satisfy `x0 > T::zero()`, got x0 = 0.0"
    => Gbm::<f64>::default().with_x0(0.0);
  the_log_heston_spot_setter_checks_finiteness: "s0 must satisfy `s0.is_finite()`, got s0 = inf"
    => HestonLog::<f64>::new(Drift::Mu(0.05), 1.5, 0.04, 0.3, -0.7, 64, 100.0, 0.04, None, VarianceBoundary::Floor, Unseeded).with_s0(f64::INFINITY);
  the_bates_variance_setter_checks_finiteness: "v0 must satisfy `v0.is_finite()`, got v0 = inf"
    => Bates::<f64>::new(Drift::Mu(0.05), 0.5, 0.05, 0.04, 1.5, 0.3, -0.6, SimdNormal::new(0.0, 0.1), 64, 100.0, 0.04, None, VarianceBoundary::Floor, Unseeded).with_v0(f64::INFINITY);
  the_double_exponential_heston_spot_setter_checks_finiteness: "s0 must satisfy `s0.is_finite()`, got s0 = inf"
    => Hkde::<f64>::new(0.05, 1.5, 0.04, 0.3, -0.7, 0.04, 0.5, 0.4, 5.0, 5.0, 64, 100.0, None, VarianceBoundary::Floor, Unseeded).with_s0(f64::INFINITY);
  a_bessel_setter_refuses_a_nan_dimension: "delta must satisfy `delta.is_finite()`, got delta = NaN"
    => Bessel::<f64>::default().with_delta(f64::NAN);
  a_squared_bessel_setter_refuses_a_nan_start: "x0 must satisfy `x0.is_none_or(|v| v.is_finite())`, got x0 = Some(NaN)"
    => SquaredBessel::<f64>::default().with_x0(Some(f64::NAN));
  a_black_karasinski_setter_refuses_a_nan_rate: "r0 must satisfy `r0.is_none_or(|v| v.is_finite())`, got r0 = Some(NaN)"
    => BlackKarasinski::<f64>::default().with_r0(Some(f64::NAN));
  a_bridge_setter_refuses_a_nan_end: "xt must satisfy `xt.is_none_or(|v| v.is_finite())`, got xt = Some(NaN)"
    => BrownianBridge::<f64>::default().with_xt(Some(f64::NAN));
  a_cev_setter_refuses_a_nan_elasticity: "gamma must satisfy `gamma.is_finite()`, got gamma = NaN"
    => Cev::<f64>::default().with_gamma(f64::NAN);
  a_correlated_fbm_setter_refuses_a_nan_correlation: "rho must satisfy `rho.is_finite()`, got rho = NaN"
    => Cfbms::<f64>::new(0.7, 0.3, 16, None, Unseeded).with_rho(f64::NAN);
  a_correlated_fgn_setter_refuses_a_nan_correlation: "rho must satisfy `rho.is_finite()`, got rho = NaN"
    => Cfgns::<f64>::new(0.7, 0.3, 16, None, Unseeded).with_rho(f64::NAN);
  a_correlated_fou_setter_refuses_a_nan_frequency: "omega must satisfy `omega.is_finite()`, got omega = NaN"
    => Cfou::<f64>::new(0.7, 1.0, 0.5, 0.2, 16, None, None, None, Unseeded).with_omega(f64::NAN);
  a_cir_pp_setter_refuses_a_nan_speed: "kappa must satisfy `kappa.is_finite()`, got kappa = NaN"
    => CirPlusPlus::<f64>::default().with_kappa(f64::NAN);
  a_displaced_diffusion_setter_refuses_a_nan_shift: "beta must satisfy `beta.is_finite()`, got beta = NaN"
    => DisplacedDiffusion::<f64>::default().with_beta(f64::NAN);
  a_fractional_bates_setter_refuses_a_nan_vol_of_vol: "xi must satisfy `xi.is_finite()`, got xi = NaN"
    => FbatesSvj::<f64>::new(0.1, 0.05, 100.0, 0.04, 0.04, 2.0, 0.3, -0.7, 0.5, -0.01, 0.1, 64, Some(1.0), Unseeded).with_xi(f64::NAN);
  a_fractional_jacobi_setter_refuses_a_nan_start: "x0 must satisfy `x0.is_none_or(|v| v.is_finite())`, got x0 = Some(NaN)"
    => Fjacobi::<f64>::new(0.7, 1.0, 2.0, 0.2, 16, None, None, Unseeded).with_x0(Some(f64::NAN));
  a_fractional_cir_setter_refuses_a_nan_speed: "theta must satisfy `theta.is_finite()`, got theta = NaN"
    => Fcir::<f64>::new(0.7, 1.0, 0.04, 0.1, 16, None, None, VarianceBoundary::Floor, Unseeded).with_theta(f64::NAN);
  a_fractional_gbm_setter_refuses_a_nan_drift: "mu must satisfy `mu.is_finite()`, got mu = NaN"
    => Fgbm::<f64>::new(0.7, 0.05, 0.2, 16, None, None, Unseeded).with_mu(f64::NAN);
  a_fractional_ou_setter_refuses_a_nan_volatility: "sigma must satisfy `sigma.is_finite()`, got sigma = NaN"
    => Fou::<f64>::new(0.7, 1.0, 0.0, 0.2, 16, None, None, Unseeded).with_sigma(f64::NAN);
  a_polynomial_volatility_setter_refuses_a_nan_coefficient: "coefficients[0] must satisfy `coefficients[i].is_finite()`, got coefficients[0] = NaN"
    => GaussianPolynomialVolatility::new(ExponentialKernel::<f64>::new(1.5, 1.0), array![0.1], 64, None, Unseeded).with_coefficients(array![f64::NAN]);
  a_local_stochastic_volatility_setter_refuses_an_infinite_mixing: "eta must satisfy `eta.is_finite()`, got eta = inf"
    => HestonSlv::<f64>::default().with_eta(f64::INFINITY);
  a_stochastic_correlation_setter_refuses_a_nan_start: "rho0 must satisfy `rho0.is_finite()`, got rho0 = NaN"
    => HestonStochCorr::<f64>::new(0.0, 100.0, 0.02, 2.1, 0.03, 0.2, -0.4, 3.4, -0.6, 0.1, 0.4, 64, Some(1.0), Unseeded).with_rho0(f64::NAN);
  a_hull_white_setter_refuses_a_nan_speed: "alpha must satisfy `alpha.is_finite()`, got alpha = NaN"
    => HullWhite::<f64>::default().with_alpha(f64::NAN);
  a_custom_jump_fou_setter_refuses_a_nan_level: "mu must satisfy `mu.is_finite()`, got mu = NaN"
    => JumpFouCustom::<f64, SimdNormal<f64>, SimdNormal<f64>>::new(0.7, 1.0, 0.0, 0.2, 16, None, None, SimdNormal::new(0.5, 0.01), SimdNormal::new(0.0, 0.1), Unseeded).with_mu(f64::NAN);
  a_jump_fou_setter_refuses_a_nan_speed: "theta must satisfy `theta.is_finite()`, got theta = NaN"
    => JumpFou::<f64, SimdNormal<f64>>::new(0.7, 1.0, 0.0, 0.2, 2.0, SimdNormal::new(0.0, 0.1), 16, None, None, Unseeded).with_theta(f64::NAN);
  a_libor_market_setter_refuses_a_nan_forward: "l0[1] must satisfy `l0[i].is_finite()`, got l0[1] = NaN"
    => Lmm::<f64>::new(array![0.0, 0.5, 1.0, 1.5, 2.0], array![0.03, 0.035, 0.04, 0.045], array![0.2, 0.2, 0.2, 0.2], 64, Some(2.0), Unseeded).with_l0(array![0.03, f64::NAN, 0.04, 0.045]);
  a_merton_setter_refuses_a_nan_drift: "alpha must satisfy `alpha.is_finite()`, got alpha = NaN"
    => Merton::<f64, SimdNormal<f64>>::default().with_alpha(f64::NAN);
  a_counting_setter_refuses_a_nan_intensity: "lambda must satisfy `lambda.is_finite()`, got lambda = NaN"
    => Poisson::<f64>::default().with_lambda(f64::NAN);
  a_variance_gamma_setter_refuses_a_nan_variance_rate: "nu must satisfy `nu.is_finite()`, got nu = NaN"
    => Vg::<f64>::default().with_nu(f64::NAN);
  a_volterra_setter_refuses_a_nan_start: "x0 must satisfy `x0.is_none_or(|v| v.is_finite())`, got x0 = Some(NaN)"
    => VolterraSde::new(ExponentialKernel::<f64>::new(1.0, 1.0), zero as fn(f64, f64) -> f64, zero as fn(f64, f64) -> f64, 64, None, None, Unseeded).with_x0(Some(f64::NAN));
  a_volterra_variance_setter_refuses_an_infinite_speed: "kappa must satisfy `kappa.is_finite()`, got kappa = inf"
    => VolterraSquareRoot::<f64, _>::new(ExponentialKernel::new(1.0, 1.0), 1.0, 0.04, 0.3, 64, 0.04, None, Unseeded).with_kappa(f64::INFINITY);
  the_rough_heston_spot_setter_checks_positivity: "s0 must satisfy `s0 > T::zero()`, got s0 = 0.0"
    => Fheston::<f64>::new(0.1, 0.2, 0.04, 1.0, 0.3, None, None, None, 64, Unseeded).with_s0(0.0);
  the_rough_heston_drift_setter_checks_finiteness: "mu must satisfy `mu.is_finite()`, got mu = NaN"
    => Fheston::<f64>::new(0.1, 0.2, 0.04, 1.0, 0.3, None, None, None, 64, Unseeded).with_mu(f64::NAN);
  the_rough_heston_correlation_setter_checks_the_unit_interval: "rho must satisfy `(-T::one()..=T::one()).contains(&rho)`, got rho = 1.5"
    => Fheston::<f64>::new(0.1, 0.2, 0.04, 1.0, 0.3, None, None, None, 64, Unseeded).with_rho(1.5);
}

fn zero(_t: f64, _x: f64) -> f64 {
  0.0
}

fn scales_with_sigma(path: impl Fn(f64) -> Array1<f64>) {
  assert!(
    path(0.0).iter().all(|x| *x == 0.0),
    "no innovation, no motion from a zero start"
  );
  let (a, b) = (path(0.5), path(1.0));
  assert!(
    a.iter().any(|x| *x != 0.0),
    "a positive sigma moves the series"
  );
  assert_eq!(b, a.mapv(|x| 2.0 * x), "the innovations scale with sigma");
}

#[test]
fn a_zero_volatility_autoregression_is_the_deterministic_recursion() {
  scales_with_sigma(|sigma| {
    Ar::<f64, Deterministic>::new(array![0.6], sigma, 64, None, Deterministic::new(3)).sample()
  });
}

#[test]
fn a_zero_volatility_moving_average_is_flat() {
  scales_with_sigma(|sigma| {
    Ma::<f64, Deterministic>::new(array![0.4, -0.2], sigma, 64, Deterministic::new(3)).sample()
  });
}

#[test]
fn a_zero_volatility_arima_is_flat() {
  scales_with_sigma(|sigma| {
    Arima::<f64, Deterministic>::new(
      array![0.5],
      array![0.3],
      1,
      sigma,
      64,
      Deterministic::new(3),
    )
    .sample()
  });
}

#[test]
fn a_zero_volatility_sarima_is_flat() {
  scales_with_sigma(|sigma| {
    Sarima::<f64, Deterministic>::new(
      array![0.5],
      array![0.3],
      array![0.2],
      array![0.1],
      1,
      1,
      4,
      sigma,
      64,
      Deterministic::new(3),
    )
    .sample()
  });
}

#[test]
fn a_zero_deviation_white_noise_is_its_mean() {
  let flat =
    Wn::<f64, Deterministic>::new(64, Some(1.5), Some(0.0), Deterministic::new(3)).sample();
  assert!(
    flat.iter().all(|x| *x == 1.5),
    "no deviation leaves every draw at the mean"
  );
  scales_with_sigma(|sd| {
    Wn::<f64, Deterministic>::new(64, None, Some(sd), Deterministic::new(3)).sample()
  });
}

#[test]
#[should_panic(
  expected = "sigma must satisfy `sigma.is_finite() && sigma >= T::zero()`, got sigma = NaN"
)]
fn a_nan_sigma_written_to_a_moving_average_is_refused() {
  let mut p = Ma::<f64, Deterministic>::new(array![0.4, -0.2], 1.0, 64, Deterministic::new(3));
  p.sigma = f64::NAN;
  let _ = p.sample();
}

#[test]
#[should_panic(
  expected = "sigma must satisfy `sigma.is_finite() && sigma >= T::zero()`, got sigma = -1.0"
)]
fn a_negative_sigma_written_to_an_arima_is_refused() {
  let mut p =
    Arima::<f64, Deterministic>::new(array![0.5], array![0.3], 1, 1.0, 64, Deterministic::new(3));
  p.sigma = -1.0;
  let _ = p.sample();
}

#[test]
#[should_panic(
  expected = "sigma must satisfy `sigma.is_finite() && sigma >= T::zero()`, got sigma = NaN"
)]
fn a_nan_sigma_written_to_a_sarima_is_refused() {
  let mut p = Sarima::<f64, Deterministic>::new(
    array![0.5],
    array![0.3],
    array![0.2],
    array![0.1],
    1,
    1,
    4,
    1.0,
    64,
    Deterministic::new(3),
  );
  p.sigma = f64::NAN;
  let _ = p.sample();
}

#[test]
#[should_panic(
  expected = "sigma must satisfy `sigma.is_finite() && sigma >= T::zero()`, got sigma = -1.0"
)]
fn a_negative_deviation_written_to_white_noise_is_refused() {
  let mut p = Wn::<f64, Deterministic>::new(64, None, Some(1.0), Deterministic::new(3));
  p.sigma = Some(-1.0);
  let _ = p.sample();
}

/// A size law the kernels do not draw, so only the zero intensity makes these processes device-ready.
fn jump_sizes() -> SimdUniform<f64> {
  SimdUniform::new(-0.1, 0.1)
}

fn is_the_diffusion_alone(finite: bool, ready: bool, intensity: Option<f64>) {
  assert!(finite, "a zero intensity samples a finite path");
  assert!(ready, "with no jumps the size law never reaches a kernel");
  assert_eq!(intensity, None, "a zero intensity launches no jump block");
}

#[test]
fn a_zero_intensity_merton_is_its_diffusion() {
  let p = Merton::new(
    0.05,
    0.2,
    0.0,
    0.0,
    jump_sizes(),
    64,
    Some(1.0),
    Some(1.0),
    Deterministic::new(9),
  );
  is_the_diffusion_alone(
    p.sample().iter().all(|x| x.is_finite()),
    p.device_ready(),
    EulerCoefficients::jump_intensity(&p),
  );
}

#[test]
fn a_zero_intensity_kou_is_its_diffusion() {
  let p = Kou::new(
    0.05,
    0.2,
    0.0,
    0.0,
    jump_sizes(),
    64,
    Some(1.0),
    Some(1.0),
    Deterministic::new(9),
  );
  is_the_diffusion_alone(
    p.sample().iter().all(|x| x.is_finite()),
    p.device_ready(),
    EulerCoefficients::jump_intensity(&p),
  );
}

#[test]
fn a_zero_intensity_levy_diffusion_is_its_diffusion() {
  let p = LevyDiffusion::new(
    0.05,
    0.2,
    0.0,
    jump_sizes(),
    64,
    Some(1.0),
    Some(1.0),
    Deterministic::new(9),
  );
  is_the_diffusion_alone(
    p.sample().iter().all(|x| x.is_finite()),
    p.device_ready(),
    EulerCoefficients::jump_intensity(&p),
  );
}

#[test]
fn a_zero_intensity_bates_is_its_diffusion() {
  let p = Bates::new(
    Drift::Mu(0.05),
    0.0,
    0.0,
    0.04,
    1.5,
    0.3,
    -0.6,
    jump_sizes(),
    64,
    100.0,
    0.04,
    Some(1.0),
    VarianceBoundary::Floor,
    Deterministic::new(9),
  );
  let [s, v] = p.sample();
  is_the_diffusion_alone(
    s.iter().chain(v.iter()).all(|x| x.is_finite()),
    p.device_ready(),
    EulerSystem::jump_intensity(&p),
  );
}

#[test]
fn a_zero_intensity_jump_fou_is_its_diffusion() {
  let p = JumpFou::new(
    0.65,
    1.5,
    0.0,
    0.2,
    0.0,
    jump_sizes(),
    64,
    Some(0.0),
    Some(1.0),
    Deterministic::new(9),
  );
  is_the_diffusion_alone(
    p.sample().iter().all(|x| x.is_finite()),
    p.device_ready(),
    EulerCoefficients::jump_intensity(&p),
  );
}

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
