use stochastic_rs_distributions::scalar::ScalarNormal;
use stochastic_rs_stochastic::diffusion::cfou::Cfou;
use stochastic_rs_stochastic::diffusion::fcir::Fcir;
use stochastic_rs_stochastic::diffusion::fgbm::Fgbm;
use stochastic_rs_stochastic::diffusion::fjacobi::FJacobi;
use stochastic_rs_stochastic::diffusion::fou::Fou;
use stochastic_rs_stochastic::jump::jump_fou::JumpFou;
use stochastic_rs_stochastic::jump::jump_fou_custom::JumpFOUCustom;
use stochastic_rs_stochastic::noise::cfgns::Cfgns;
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
    None,
    Default::default(),
  )
}

fn fjacobi() -> FJacobi<f64> {
  FJacobi::new(0.7, 1.0, 2.0, 0.2, 16, None, None, Default::default())
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

fn jump_fou() -> JumpFou<f64, ScalarNormal<f64>> {
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

fn jump_fou_custom() -> JumpFOUCustom<f64, ScalarNormal<f64>> {
  JumpFOUCustom::new(
    0.7,
    1.0,
    0.0,
    0.2,
    16,
    None,
    None,
    ScalarNormal::new(0.5, 0.01),
    jump_law(0.1),
    Default::default(),
  )
}

rejects! {
  fou_with_steps_rejects_one_point: "n must be at least 2" => fou().with_steps(1);
  fgbm_with_steps_rejects_one_point: "n must be at least 2" => fgbm().with_steps(1);
  fcir_with_steps_rejects_one_point: "n must be at least 2" => fcir().with_steps(1);
  fjacobi_with_steps_rejects_one_point: "n must be at least 2" => fjacobi().with_steps(1);
  cfbms_with_steps_rejects_one_point: "n must be at least 2" => cfbms().with_steps(1);
  cfou_with_steps_rejects_one_point: "n must be at least 2" => cfou().with_steps(1);
  jump_fou_with_steps_rejects_one_point: "n must be at least 2" => jump_fou().with_steps(1);
  jump_fou_custom_with_steps_rejects_one_point: "n must be at least 2" => jump_fou_custom().with_steps(1);

  fjacobi_with_alpha_rejects_zero: "alpha must be positive" => fjacobi().with_alpha(0.0);
  fjacobi_with_alpha_rejects_the_value_of_beta: "alpha must be less than beta" => fjacobi().with_alpha(2.0);
  fjacobi_with_beta_rejects_zero: "beta must be positive" => fjacobi().with_beta(0.0);
  fjacobi_with_beta_rejects_the_value_of_alpha: "alpha must be less than beta" => fjacobi().with_beta(1.0);
  fjacobi_with_sigma_rejects_zero: "sigma must be positive" => fjacobi().with_sigma(0.0);

  cfou_with_lambda_rejects_zero: "lambda must be positive" => cfou().with_lambda(0.0);
  cfou_with_a_rejects_zero: "a must be positive" => cfou().with_a(0.0);

  cfbms_with_hurst_rejects_above_one: "Hurst parameter must be in (0, 1)" => cfbms().with_hurst(1.5);
  cfbms_with_rho_rejects_above_one: "Correlation coefficient must be in [-1, 1]" => cfbms().with_rho(1.5);
  cfgns_with_hurst_rejects_below_zero: "Hurst parameter must be in (0, 1)" => cfgns().with_hurst(-0.1);
  cfgns_with_rho_rejects_below_minus_one: "Correlation coefficient must be in [-1, 1]" => cfgns().with_rho(-1.5);

  cir_2f_with_x_rejects_another_n: "x and y Cir factors must use the same n" => cir_2f().with_x(cir(33, 1.0));
  cir_2f_with_y_rejects_another_n: "x and y Cir factors must use the same n" => cir_2f().with_y(cir(33, 1.0));
  cir_2f_with_x_rejects_another_horizon: "x and y Cir factors must use the same time horizon" => cir_2f().with_x(cir(32, 2.0));
  cir_2f_with_y_rejects_another_horizon: "x and y Cir factors must use the same time horizon" => cir_2f().with_y(cir(32, 2.0));
}
