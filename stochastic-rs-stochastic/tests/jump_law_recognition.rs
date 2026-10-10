use rand::Rng;
use rand::distr::Distribution;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::exp::SimdExp;
use stochastic_rs_distributions::normal::SimdNormal;
use stochastic_rs_stochastic::jump::bates::Bates1996;
use stochastic_rs_stochastic::jump::jump_fou::JumpFou;
use stochastic_rs_stochastic::jump::jump_fou_custom::JumpFOUCustom;
use stochastic_rs_stochastic::jump::kou::Kou;
use stochastic_rs_stochastic::jump::levy_diffusion::LevyDiffusion;
use stochastic_rs_stochastic::jump::merton::Merton;
use stochastic_rs_stochastic::process::ccustom::CompoundCustom;
use stochastic_rs_stochastic::process::cpoisson::CompoundPoisson;
use stochastic_rs_stochastic::process::customjt::CustomJt;
use stochastic_rs_stochastic::process::poisson::Poisson;
use stochastic_rs_stochastic::traits::ProcessExt;

const N: usize = 24;
const LAMBDA: f64 = 5.0;

struct Const(f64);

impl Distribution<f64> for Const {
  fn sample<G: Rng + ?Sized>(&self, _rng: &mut G) -> f64 {
    self.0
  }
}

fn reaches_the_kernels(name: &str, why: Option<&'static str>) {
  assert_eq!(
    why, None,
    "{name}: a stateless Simd law must reach the device kernels"
  );
}

fn stays_on_the_host(name: &str, why: Option<&'static str>) {
  assert!(
    why.is_some(),
    "{name}: a law the kernels do not draw must name its fallback"
  );
}

#[test]
fn stateless_simd_laws_reach_the_kernels_of_every_law_taking_process() {
  let normal = || SimdNormal::<f64>::new(0.0, 0.1);
  let exp = || SimdExp::<f64>::new(20.0);

  reaches_the_kernels(
    "Merton",
    Merton::new(
      0.03,
      0.2,
      LAMBDA,
      0.0,
      normal(),
      N,
      Some(0.0),
      Some(1.0),
      Unseeded,
    )
    .device_fallback(),
  );
  reaches_the_kernels(
    "Kou",
    Kou::new(
      0.03,
      0.2,
      LAMBDA,
      0.0,
      normal(),
      N,
      Some(0.0),
      Some(1.0),
      Unseeded,
    )
    .device_fallback(),
  );
  reaches_the_kernels(
    "LevyDiffusion",
    LevyDiffusion::new(
      0.01,
      0.2,
      LAMBDA,
      normal(),
      N,
      Some(0.0),
      Some(1.0),
      Unseeded,
    )
    .device_fallback(),
  );
  reaches_the_kernels(
    "JumpFou",
    JumpFou::new(
      0.65,
      1.5,
      0.0,
      0.2,
      LAMBDA,
      normal(),
      N,
      Some(0.0),
      Some(1.0),
      Unseeded,
    )
    .device_fallback(),
  );
  reaches_the_kernels(
    "JumpFOUCustom",
    JumpFOUCustom::new(
      0.65,
      1.5,
      0.0,
      0.2,
      N,
      Some(0.0),
      Some(1.0),
      exp(),
      exp(),
      Unseeded,
    )
    .device_fallback(),
  );
  reaches_the_kernels(
    "Bates1996",
    Bates1996::new(
      Some(0.05),
      None,
      None,
      None,
      LAMBDA,
      0.0,
      0.04,
      1.5,
      0.3,
      -0.6,
      normal(),
      N,
      Some(100.0),
      Some(0.04),
      Some(1.0),
      Some(false),
      Unseeded,
    )
    .device_fallback(),
  );
  reaches_the_kernels(
    "CompoundPoisson",
    CompoundPoisson::new(
      normal(),
      Poisson::new(LAMBDA, Some(N), Some(1.0), Unseeded),
      Unseeded,
    )
    .device_fallback(),
  );
  reaches_the_kernels(
    "CustomJt",
    CustomJt::new(Some(N), None, exp(), Unseeded).device_fallback(),
  );
  reaches_the_kernels(
    "CompoundCustom",
    CompoundCustom::new(
      Some(N),
      None,
      normal(),
      exp(),
      CustomJt::new(Some(N), None, exp(), Unseeded),
      Unseeded,
    )
    .device_fallback(),
  );
}

#[test]
fn a_law_the_kernels_do_not_draw_stays_on_the_host() {
  let exp = || SimdExp::<f64>::new(20.0);

  stays_on_the_host(
    "Merton",
    Merton::new(
      0.03,
      0.2,
      LAMBDA,
      0.0,
      Const(0.1),
      N,
      Some(0.0),
      Some(1.0),
      Unseeded,
    )
    .device_fallback(),
  );
  stays_on_the_host(
    "Kou",
    Kou::new(
      0.03,
      0.2,
      LAMBDA,
      0.0,
      Const(0.1),
      N,
      Some(0.0),
      Some(1.0),
      Unseeded,
    )
    .device_fallback(),
  );
  stays_on_the_host(
    "LevyDiffusion",
    LevyDiffusion::new(
      0.01,
      0.2,
      LAMBDA,
      Const(0.1),
      N,
      Some(0.0),
      Some(1.0),
      Unseeded,
    )
    .device_fallback(),
  );
  stays_on_the_host(
    "JumpFou",
    JumpFou::new(
      0.65,
      1.5,
      0.0,
      0.2,
      LAMBDA,
      Const(0.1),
      N,
      Some(0.0),
      Some(1.0),
      Unseeded,
    )
    .device_fallback(),
  );
  stays_on_the_host(
    "JumpFOUCustom",
    JumpFOUCustom::new(
      0.65,
      1.5,
      0.0,
      0.2,
      N,
      Some(0.0),
      Some(1.0),
      Const(0.05),
      Const(0.1),
      Unseeded,
    )
    .device_fallback(),
  );
  stays_on_the_host(
    "Bates1996",
    Bates1996::new(
      Some(0.05),
      None,
      None,
      None,
      LAMBDA,
      0.0,
      0.04,
      1.5,
      0.3,
      -0.6,
      Const(0.1),
      N,
      Some(100.0),
      Some(0.04),
      Some(1.0),
      Some(false),
      Unseeded,
    )
    .device_fallback(),
  );
  stays_on_the_host(
    "CompoundPoisson",
    CompoundPoisson::new(
      Const(0.1),
      Poisson::new(LAMBDA, Some(N), Some(1.0), Unseeded),
      Unseeded,
    )
    .device_fallback(),
  );
  stays_on_the_host(
    "CustomJt",
    CustomJt::new(Some(N), None, Const(0.05), Unseeded).device_fallback(),
  );
  stays_on_the_host(
    "CompoundCustom",
    CompoundCustom::new(
      Some(N),
      None,
      Const(0.1),
      exp(),
      CustomJt::new(Some(N), None, exp(), Unseeded),
      Unseeded,
    )
    .device_fallback(),
  );
}

#[test]
fn single_precision_laws_reach_the_kernels_too() {
  let merton = Merton::<f32, _, _>::new(
    0.03,
    0.2,
    5.0,
    0.0,
    SimdNormal::<f32>::new(0.0, 0.1),
    N,
    Some(0.0),
    Some(1.0),
    Unseeded,
  );
  reaches_the_kernels("Merton f32", merton.device_fallback());
}
