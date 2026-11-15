use ndarray::Array1;
use ndarray::Array2;
use ndarray::array;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::normal::SimdNormal;
use stochastic_rs_distributions::uniform::SimdUniform;
use stochastic_rs_stochastic::autoregressive::ar::ARp;
use stochastic_rs_stochastic::autoregressive::arima::Arima;
use stochastic_rs_stochastic::autoregressive::ma::MAq;
use stochastic_rs_stochastic::autoregressive::sarima::Sarima;
use stochastic_rs_stochastic::correlation::VanEmmerich;
use stochastic_rs_stochastic::correlation::stochastic_correlated_bm;
use stochastic_rs_stochastic::diffusion::gbm::Gbm;
use stochastic_rs_stochastic::diffusion::multi_gbm::MultiGbm;
use stochastic_rs_stochastic::diffusion::ou::Ou;
use stochastic_rs_stochastic::diffusion::wishart::Wishart;
use stochastic_rs_stochastic::euler::EulerCoefficients;
use stochastic_rs_stochastic::euler::EulerSystem;
use stochastic_rs_stochastic::interest::wu_zhang::WuZhangD;
use stochastic_rs_stochastic::jump::bates::Bates1996;
use stochastic_rs_stochastic::jump::bilateral_gamma::BilateralGamma;
use stochastic_rs_stochastic::jump::bilateral_gamma::BilateralGammaMotion;
use stochastic_rs_stochastic::jump::cts::Cts;
use stochastic_rs_stochastic::jump::ig::Ig;
use stochastic_rs_stochastic::jump::jump_fou::JumpFou;
use stochastic_rs_stochastic::jump::jump_fou_custom::JumpFOUCustom;
use stochastic_rs_stochastic::jump::kou::Kou;
use stochastic_rs_stochastic::jump::levy_diffusion::LevyDiffusion;
use stochastic_rs_stochastic::jump::merton::Merton;
use stochastic_rs_stochastic::jump::nig::Nig;
use stochastic_rs_stochastic::jump::rdts::Rdts;
use stochastic_rs_stochastic::jump::vg::Vg;
use stochastic_rs_stochastic::noise::cfgns::Cfgns;
use stochastic_rs_stochastic::noise::fgn::Fgn;
use stochastic_rs_stochastic::noise::gn::Gn;
use stochastic_rs_stochastic::noise::wn::Wn;
use stochastic_rs_stochastic::process::lfsm::Lfsm;
use stochastic_rs_stochastic::process::multivariate_hawkes::MultivariateHawkes;
use stochastic_rs_stochastic::process::subordinator::gamma_subordinator::GammaSubordinator;
use stochastic_rs_stochastic::process::volterra::Volterra;
use stochastic_rs_stochastic::process::volterra::VolterraKernelSpec;
use stochastic_rs_stochastic::rough::rl_fbm::RlFBm;
use stochastic_rs_stochastic::sheet::fbs::Fbs;
use stochastic_rs_stochastic::traits::ProcessExt;
use stochastic_rs_stochastic::volatility::HestonPow;
use stochastic_rs_stochastic::volatility::heston::Heston;
use stochastic_rs_stochastic::volatility::multifactor_sabr::MultifactorSabr;
use stochastic_rs_stochastic::volterra::ExponentialKernel;
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
  let _ = stochastic_correlated_bm(&array![0.2, 1.2, 0.1], None);
}

#[test]
#[should_panic(
  expected = "rho_path[0] must satisfy `(-1.0..=1.0).contains(&rho_path[i])`, got rho_path[0] = NaN"
)]
fn correlated_brownian_motions_refuse_a_nan_correlation() {
  let _ = stochastic_correlated_bm(&array![f64::NAN, 0.1], None);
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
  a_nan_parameter_is_refused_at_construction: "sigma must satisfy `sigma.is_finite()`, got sigma = NaN"
    => Ou::<f64>::new(1.0, 0.0, f64::NAN, 64, None, None, Unseeded);
  a_grid_of_one_point_is_refused: "n must satisfy `n >= 2`, got n = 1"
    => Ou::<f64>::new(1.0, 0.0, 0.2, 1, None, None, Unseeded);
  a_noise_of_no_increments_is_refused: "n must satisfy `n >= 1`, got n = 0"
    => Gn::<f64>::new(0, None, Unseeded);
  a_zero_horizon_is_refused: "t must satisfy `t.is_none_or(|t| t > T::zero() && t.is_finite())`, got t = Some(0.0)"
    => Gbm::<f64>::new(0.05, 0.2, 64, None, Some(0.0), Unseeded);
  a_negative_horizon_is_refused: "t must satisfy `t.is_none_or(|t| t > T::zero() && t.is_finite())`, got t = Some(-1.0)"
    => Ou::<f64>::new(1.0, 0.0, 0.2, 64, None, Some(-1.0), Unseeded);
  gbm_refuses_a_negative_volatility: "sigma must satisfy `sigma >= T::zero()`, got sigma = -0.2"
    => Gbm::<f64>::new(0.05, -0.2, 64, None, None, Unseeded);
  gbm_refuses_a_non_positive_start: "x0 must satisfy `x0.is_none_or(|x0| x0 > T::zero())`, got x0 = Some(0.0)"
    => Gbm::<f64>::new(0.05, 0.2, 64, Some(0.0), None, Unseeded);
  the_tempered_stable_constant_has_a_pole_at_alpha_one: "alpha must satisfy `alpha != T::one()`, got alpha = 1.0"
    => Cts::<f64>::new(2.0, 3.0, 1.0, 64, 8, None, None, Unseeded);
  the_rapidly_decreasing_drift_has_a_pole_at_alpha_one: "alpha must satisfy `alpha != T::one()`, got alpha = 1.0"
    => Rdts::<f64>::new(2.0, 3.0, 1.0, 64, 8, None, None, Unseeded);
  a_volterra_kernel_validates_its_parameter: "h must satisfy `h > 0.0 && h < 1.0`, got h = 1.5"
    => Volterra::<f64>::new(VolterraKernelSpec::FractionalBM { h: 1.5 }, 64, None, Unseeded);
  a_power_law_kernel_is_square_integrable: "gamma must satisfy `gamma.is_finite() && gamma > -0.5`, got gamma = -0.5"
    => Volterra::<f64>::new(VolterraKernelSpec::PowerLaw { gamma: -0.5 }, 64, None, Unseeded);
  a_power_law_kernel_has_a_finite_exponent: "gamma must satisfy `gamma.is_finite() && gamma > -0.5`, got gamma = inf"
    => Volterra::<f64>::new(VolterraKernelSpec::PowerLaw { gamma: f64::INFINITY }, 64, None, Unseeded);
  an_exponential_kernel_decays: "beta must satisfy `beta.is_finite() && beta > 0.0`, got beta = 0.0"
    => Volterra::<f64>::new(VolterraKernelSpec::Exponential { beta: 0.0 }, 64, None, Unseeded);
  an_exponential_kernel_has_a_finite_rate: "beta must satisfy `beta.is_finite() && beta > 0.0`, got beta = inf"
    => Volterra::<f64>::new(VolterraKernelSpec::Exponential { beta: f64::INFINITY }, 64, None, Unseeded);

  a_discrete_series_refuses_no_observations: "n must satisfy `n >= 1`, got n = 0"
    => ARp::<f64>::new(array![0.5], 1.0, 0, None, Unseeded);
  an_autoregression_refuses_a_nan_coefficient: "phi[0] must satisfy `phi[i].is_finite()`, got phi[0] = NaN"
    => ARp::<f64>::new(array![f64::NAN], 1.0, 64, None, Unseeded);
  a_correlation_process_refuses_a_nan_level: "mu must satisfy `mu.is_finite()`, got mu = NaN"
    => VanEmmerich::<f64>::new(1.0, f64::NAN, 0.2, 0.0, 64, None, Unseeded);
  correlated_assets_refuse_a_nan_drift: "mu[1] must satisfy `mu[i].is_finite()`, got mu[1] = NaN"
    => MultiGbm::<f64>::new(array![0.0, f64::NAN], array![0.2, 0.2], array![[1.0, 0.0], [0.0, 1.0]], 64, array![1.0, 1.0], None, Unseeded);
  correlated_assets_refuse_one_point: "n must satisfy `n >= 2`, got n = 1"
    => MultiGbm::<f64>::new(array![0.0, 0.0], array![0.2, 0.2], array![[1.0, 0.0], [0.0, 1.0]], 1, array![1.0, 1.0], None, Unseeded);
  a_matrix_diffusion_refuses_one_point: "n must satisfy `n >= 2`, got n = 1"
    => Wishart::<f64>::new(2.5, Array2::zeros((2, 2)), Array2::eye(2), Array2::eye(2), 1, None, Unseeded);
  a_rate_model_refuses_a_nan_entry: "alpha[0] must satisfy `alpha[i].is_finite()`, got alpha[0] = NaN"
    => WuZhangD::<f64>::new(array![f64::NAN], array![0.1], array![0.1], array![0.1], array![0.1], array![0.1], 1, None, 64, Unseeded);
  an_excitation_matrix_names_its_nan_entry: "alpha[(1, 0)] must satisfy `alpha[(i, j)].is_finite()`, got alpha[(1, 0)] = NaN"
    => MultivariateHawkes::<f64>::new(array![1.0, 1.0], array![[0.1, 0.1], [f64::NAN, 0.1]], array![[1.0, 1.0], [1.0, 1.0]], 1.0, Unseeded);
  a_fractional_jump_process_refuses_a_nan_volatility: "sigma must satisfy `sigma.is_finite()`, got sigma = NaN"
    => JumpFou::<f64, SimdNormal<f64>>::new(0.7, 1.0, 0.0, f64::NAN, 2.0, SimdNormal::new(0.0, 0.1), 64, None, None, Unseeded);
  a_custom_jump_process_refuses_a_nan_level: "mu must satisfy `mu.is_finite()`, got mu = NaN"
    => JumpFOUCustom::<f64, SimdNormal<f64>>::new(0.7, 1.0, f64::NAN, 0.2, 64, None, None, SimdNormal::new(0.5, 0.01), SimdNormal::new(0.0, 0.1), Unseeded);
  a_fractional_noise_refuses_a_nan_horizon: "t must satisfy `t.is_none_or(|t| t > T::zero() && t.is_finite())`, got t = Some(NaN)"
    => Fgn::<f64>::new(0.7, 64, Some(f64::NAN), Unseeded);
  a_subordinator_refuses_a_nan_rate: "rate must satisfy `rate.is_finite()`, got rate = NaN"
    => GammaSubordinator::<f64>::new(1.0, f64::NAN, 64, None, None, Unseeded);
  a_rough_process_refuses_a_nan_hurst: "hurst must satisfy `hurst.is_finite()`, got hurst = NaN"
    => RlFBm::<f64>::new(f64::NAN, 64, None, None, Unseeded);
  a_fractional_sheet_refuses_a_nan_hurst: "hurst must satisfy `hurst.is_finite()`, got hurst = NaN"
    => Fbs::<f64>::new(f64::NAN, 16, 16, 1.0, Unseeded);
  a_volatility_model_refuses_a_nan_correlation: "rho must satisfy `rho.is_finite()`, got rho = NaN"
    => Heston::<f64>::new(Some(100.0), Some(0.04), 1.5, 0.04, 0.3, f64::NAN, 0.05, 64, None, HestonPow::Sqrt, None, Unseeded);
  a_volterra_kernel_refuses_a_nan_weight: "c must satisfy `c.is_finite()`, got c = NaN"
    => ExponentialKernel::<f64>::new(1.0, f64::NAN);
  a_volterra_volatility_refuses_a_nan_level: "theta must satisfy `theta.is_finite()`, got theta = NaN"
    => VolterraSquareRoot::<f64, _>::new(ExponentialKernel::new(1.0, 1.0), 1.0, f64::NAN, 0.3, 64, None, None, Unseeded);

  an_inverse_gaussian_process_refuses_one_point: "n must satisfy `n >= 2`, got n = 1"
    => Ig::<f64>::new(1.0, 1, Some(3.5), Some(1.0), Unseeded);
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
    ARp::<f64, Deterministic>::new(array![0.6], sigma, 64, None, Deterministic::new(3)).sample()
  });
}

#[test]
fn a_zero_volatility_moving_average_is_flat() {
  scales_with_sigma(|sigma| {
    MAq::<f64, Deterministic>::new(array![0.4, -0.2], sigma, 64, Deterministic::new(3)).sample()
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
  let mut p = MAq::<f64, Deterministic>::new(array![0.4, -0.2], 1.0, 64, Deterministic::new(3));
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
  expected = "std_dev must satisfy `std_dev.is_finite() && std_dev >= T::zero()`, got std_dev = -1.0"
)]
fn a_negative_deviation_written_to_white_noise_is_refused() {
  let mut p = Wn::<f64, Deterministic>::new(64, None, Some(1.0), Deterministic::new(3));
  p.std_dev = Some(-1.0);
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
  let p = Bates1996::new(
    Some(0.05),
    None,
    None,
    None,
    0.0,
    0.0,
    0.04,
    1.5,
    0.3,
    -0.6,
    jump_sizes(),
    64,
    Some(100.0),
    Some(0.04),
    Some(1.0),
    Some(false),
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
