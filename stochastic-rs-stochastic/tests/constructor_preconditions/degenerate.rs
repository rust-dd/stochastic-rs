use ndarray::Array1;
use ndarray::array;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_distributions::uniform::SimdUniform;
use stochastic_rs_stochastic::autoregressive::ar::Ar;
use stochastic_rs_stochastic::autoregressive::arima::Arima;
use stochastic_rs_stochastic::autoregressive::ma::Ma;
use stochastic_rs_stochastic::autoregressive::sarima::Sarima;
use stochastic_rs_stochastic::euler::EulerCoefficients;
use stochastic_rs_stochastic::euler::EulerSystem;
use stochastic_rs_stochastic::jump::bates::Bates;
use stochastic_rs_stochastic::jump::jump_fou::JumpFou;
use stochastic_rs_stochastic::jump::kou::Kou;
use stochastic_rs_stochastic::jump::levy_diffusion::LevyDiffusion;
use stochastic_rs_stochastic::jump::merton::Merton;
use stochastic_rs_stochastic::noise::wn::Wn;
use stochastic_rs_stochastic::params::Drift;
use stochastic_rs_stochastic::params::VarianceBoundary;
use stochastic_rs_stochastic::traits::ProcessExt;

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
