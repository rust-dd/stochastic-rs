use stochastic_rs_distributions::normal::SimdNormal;
use stochastic_rs_stochastic::diffusion::cfou::Cfou;
use stochastic_rs_stochastic::diffusion::fcir::Fcir;
use stochastic_rs_stochastic::diffusion::fgbm::Fgbm;
use stochastic_rs_stochastic::diffusion::fjacobi::Fjacobi;
use stochastic_rs_stochastic::diffusion::fou::Fou;
use stochastic_rs_stochastic::jump::jump_fou::JumpFou;
use stochastic_rs_stochastic::jump::jump_fou_custom::JumpFouCustom;
use stochastic_rs_stochastic::noise::cfgns::Cfgns;
use stochastic_rs_stochastic::params::VarianceBoundary;
use stochastic_rs_stochastic::process::cfbms::Cfbms;

use super::cir_2f::cir;
use super::cir_2f::cir_2f;
use super::common::jump_law;

macro_rules! rejects {
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

fn fou() -> Fou<f64> {
  Fou::new(0.7, 1.0, 0.0, 0.2, 16, None, None, Default::default())
}

fn fgbm() -> Fgbm<f64> {
  Fgbm::new(0.7, 0.05, 0.2, 16, None, None, Default::default())
}

fn fcir() -> Fcir<f64> {
  Fcir::new(
    0.7,
    1.0,
    0.04,
    0.1,
    16,
    None,
    None,
    VarianceBoundary::Floor,
    Default::default(),
  )
}

fn fjacobi() -> Fjacobi<f64> {
  Fjacobi::new(0.7, 1.0, 2.0, 0.2, 16, None, None, Default::default())
}

fn cfbms() -> Cfbms<f64> {
  Cfbms::new(0.7, 0.3, 16, None, Default::default())
}

fn cfgns() -> Cfgns<f64> {
  Cfgns::new(0.7, 0.3, 16, None, Default::default())
}

fn cfou() -> Cfou<f64> {
  Cfou::new(0.7, 1.0, 0.5, 0.2, 16, None, None, None, Default::default())
}

fn jump_fou() -> JumpFou<f64, SimdNormal<f64>> {
  JumpFou::new(
    0.7,
    1.0,
    0.0,
    0.2,
    2.0,
    jump_law(0.1),
    16,
    None,
    None,
    Default::default(),
  )
}

fn jump_fou_custom() -> JumpFouCustom<f64, SimdNormal<f64>, SimdNormal<f64>> {
  JumpFouCustom::new(
    0.7,
    1.0,
    0.0,
    0.2,
    16,
    None,
    None,
    SimdNormal::new(0.5, 0.01),
    jump_law(0.1),
    Default::default(),
  )
}

rejects! {
  fou_with_points_rejects_one_point: "n must satisfy `n >= 2`, got n = 1" => fou().with_points(1);
  fgbm_with_points_rejects_one_point: "n must satisfy `n >= 2`, got n = 1" => fgbm().with_points(1);
  fcir_with_points_rejects_one_point: "n must satisfy `n >= 2`, got n = 1" => fcir().with_points(1);
  fjacobi_with_points_rejects_one_point: "n must satisfy `n >= 2`, got n = 1" => fjacobi().with_points(1);
  cfbms_with_points_rejects_one_point: "n must satisfy `n >= 2`, got n = 1" => cfbms().with_points(1);
  cfou_with_points_rejects_one_point: "n must satisfy `n >= 2`, got n = 1" => cfou().with_points(1);
  jump_fou_with_points_rejects_one_point: "n must satisfy `n >= 2`, got n = 1" => jump_fou().with_points(1);
  jump_fou_custom_with_points_rejects_one_point: "n must satisfy `n >= 2`, got n = 1" => jump_fou_custom().with_points(1);

  fjacobi_with_alpha_rejects_zero: "alpha must satisfy `alpha > T::zero()`, got alpha = 0.0" => fjacobi().with_alpha(0.0);
  fjacobi_with_alpha_rejects_the_value_of_beta: "alpha must satisfy `alpha < self.beta`, got alpha = 2.0, self.beta = 2.0" => fjacobi().with_alpha(2.0);
  fjacobi_with_beta_rejects_zero: "beta must satisfy `beta > T::zero()`, got beta = 0.0" => fjacobi().with_beta(0.0);
  fjacobi_with_beta_rejects_the_value_of_alpha: "beta must satisfy `self.alpha < beta`, got beta = 1.0, self.alpha = 1.0" => fjacobi().with_beta(1.0);
  fjacobi_with_sigma_rejects_zero: "sigma must satisfy `sigma > T::zero()`, got sigma = 0.0" => fjacobi().with_sigma(0.0);

  cfou_with_lambda_rejects_zero: "lambda must satisfy `lambda > T::zero()`, got lambda = 0.0" => cfou().with_lambda(0.0);
  cfou_with_a_rejects_zero: "a must satisfy `a > T::zero()`, got a = 0.0" => cfou().with_a(0.0);

  cfbms_with_hurst_rejects_above_one: "hurst must satisfy `(T::zero()..=T::one()).contains(&hurst)`, got hurst = 1.5" => cfbms().with_hurst(1.5);
  cfbms_with_rho_rejects_above_one: "rho must satisfy `" => cfbms().with_rho(1.5);
  cfgns_with_hurst_rejects_below_zero: "hurst must satisfy `(T::zero()..=T::one()).contains(&hurst)`, got hurst = -0.1" => cfgns().with_hurst(-0.1);
  cfgns_with_rho_rejects_below_minus_one: "rho must satisfy `" => cfgns().with_rho(-1.5);

  cir_2f_with_x_rejects_another_n: "x.n must satisfy `x.n == y.n`, got x.n = 33, y.n = 32" => cir_2f().with_x(cir(33, 1.0));
  cir_2f_with_y_rejects_another_n: "x.n must satisfy `x.n == y.n`, got x.n = 32, y.n = 33" => cir_2f().with_y(cir(33, 1.0));
  cir_2f_with_x_rejects_another_horizon: "x.t must satisfy `(x.t - y.t).abs() <= 1e-12`, got x.t = 2.0, y.t = 1.0" => cir_2f().with_x(cir(32, 2.0));
  cir_2f_with_y_rejects_another_horizon: "x.t must satisfy `(x.t - y.t).abs() <= 1e-12`, got x.t = 1.0, y.t = 2.0" => cir_2f().with_y(cir(32, 2.0));
}
