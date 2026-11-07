use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_distributions::normal::SimdNormal;
use stochastic_rs_stochastic::ProcessExt;
use stochastic_rs_stochastic::diffusion::cfou::Cfou;
use stochastic_rs_stochastic::diffusion::fcir::Fcir;
use stochastic_rs_stochastic::diffusion::fgbm::Fgbm;
use stochastic_rs_stochastic::diffusion::fjacobi::FJacobi;
use stochastic_rs_stochastic::diffusion::fou::Fou;
use stochastic_rs_stochastic::jump::jump_fou::JumpFou;
use stochastic_rs_stochastic::jump::jump_fou_custom::JumpFOUCustom;
use stochastic_rs_stochastic::noise::cfgns::Cfgns;
use stochastic_rs_stochastic::process::cfbms::Cfbms;
use stochastic_rs_stochastic::process::fbm::Fbm;

use super::common::jump_law;
use super::common::seed;

#[test]
fn fbm_setters_match_fresh_construction() {
  let chained = Fbm::<f64, _>::new(0.3, 16, Some(2.0), seed(1))
    .with_hurst(0.7)
    .with_steps(64)
    .with_horizon(Some(1.0))
    .with_seed(seed(7));
  let fresh = Fbm::<f64, _>::new(0.7, 64, Some(1.0), seed(7));

  assert_eq!(
    (chained.hurst(), chained.n(), chained.t()),
    (0.7, 64, Some(1.0))
  );
  assert_eq!(chained.seed().current(), 7);
  assert_eq!(chained.sample(), fresh.sample());
}

#[test]
fn fou_setters_match_fresh_construction() {
  let chained = Fou::<f64, _>::new(0.3, 2.0, 0.5, 0.4, 16, Some(0.1), Some(2.0), seed(1))
    .with_hurst(0.7)
    .with_theta(1.0)
    .with_mu(0.0)
    .with_sigma(0.2)
    .with_steps(64)
    .with_x0(Some(0.0))
    .with_horizon(Some(1.0))
    .with_seed(seed(7));
  let fresh = Fou::<f64, _>::new(0.7, 1.0, 0.0, 0.2, 64, Some(0.0), Some(1.0), seed(7));

  assert_eq!(
    (
      chained.hurst(),
      chained.theta(),
      chained.mu(),
      chained.sigma()
    ),
    (0.7, 1.0, 0.0, 0.2)
  );
  assert_eq!(
    (chained.n(), chained.x0(), chained.t()),
    (64, Some(0.0), Some(1.0))
  );
  assert_eq!(chained.seed().current(), 7);
  assert_eq!(chained.sample(), fresh.sample());
}

#[test]
fn fgbm_setters_match_fresh_construction() {
  let chained = Fgbm::<f64, _>::new(0.3, 0.2, 0.4, 16, Some(2.0), Some(2.0), seed(1))
    .with_hurst(0.7)
    .with_mu(0.05)
    .with_sigma(0.2)
    .with_steps(64)
    .with_x0(Some(1.0))
    .with_horizon(Some(1.0))
    .with_seed(seed(7));
  let fresh = Fgbm::<f64, _>::new(0.7, 0.05, 0.2, 64, Some(1.0), Some(1.0), seed(7));

  assert_eq!(
    (chained.hurst(), chained.mu(), chained.sigma()),
    (0.7, 0.05, 0.2)
  );
  assert_eq!(
    (chained.n(), chained.x0(), chained.t()),
    (64, Some(1.0), Some(1.0))
  );
  assert_eq!(chained.seed().current(), 7);
  assert_eq!(chained.sample(), fresh.sample());
}

#[test]
fn fcir_setters_match_fresh_construction() {
  let chained = Fcir::<f64, _>::new(
    0.3,
    2.0,
    0.09,
    0.3,
    16,
    Some(0.09),
    Some(2.0),
    Some(true),
    seed(1),
  )
  .with_hurst(0.7)
  .with_theta(1.0)
  .with_mu(0.04)
  .with_sigma(0.1)
  .with_steps(64)
  .with_x0(Some(0.04))
  .with_horizon(Some(1.0))
  .with_use_sym(Some(false))
  .with_seed(seed(7));
  let fresh = Fcir::<f64, _>::new(
    0.7,
    1.0,
    0.04,
    0.1,
    64,
    Some(0.04),
    Some(1.0),
    Some(false),
    seed(7),
  );

  assert_eq!(
    (
      chained.hurst(),
      chained.theta(),
      chained.mu(),
      chained.sigma()
    ),
    (0.7, 1.0, 0.04, 0.1)
  );
  assert_eq!(
    (chained.n(), chained.x0(), chained.t(), chained.use_sym()),
    (64, Some(0.04), Some(1.0), Some(false))
  );
  assert_eq!(chained.seed().current(), 7);
  assert_eq!(chained.sample(), fresh.sample());
}

#[test]
fn fjacobi_setters_match_fresh_construction() {
  let chained = FJacobi::<f64, _>::new(0.3, 0.5, 1.5, 0.4, 16, Some(0.3), Some(2.0), seed(1))
    .with_hurst(0.7)
    .with_beta(2.0)
    .with_alpha(1.0)
    .with_sigma(0.2)
    .with_steps(64)
    .with_x0(Some(0.5))
    .with_horizon(Some(1.0))
    .with_seed(seed(7));
  let fresh = FJacobi::<f64, _>::new(0.7, 1.0, 2.0, 0.2, 64, Some(0.5), Some(1.0), seed(7));

  assert_eq!(
    (
      chained.hurst(),
      chained.alpha(),
      chained.beta(),
      chained.sigma()
    ),
    (0.7, 1.0, 2.0, 0.2)
  );
  assert_eq!(
    (chained.n(), chained.x0(), chained.t()),
    (64, Some(0.5), Some(1.0))
  );
  assert_eq!(chained.seed().current(), 7);
  assert_eq!(chained.sample(), fresh.sample());
}

#[test]
fn cfbms_setters_match_fresh_construction() {
  let chained = Cfbms::<f64, _>::new(0.4, -0.5, 16, Some(2.0), seed(1))
    .with_hurst(0.7)
    .with_rho(0.3)
    .with_steps(64)
    .with_horizon(Some(1.0))
    .with_seed(seed(7));
  let fresh = Cfbms::<f64, _>::new(0.7, 0.3, 64, Some(1.0), seed(7));

  assert_eq!(
    (chained.hurst(), chained.rho(), chained.n(), chained.t()),
    (0.7, 0.3, 64, Some(1.0))
  );
  assert_eq!(chained.seed().current(), 7);
  assert_eq!(chained.sample(), fresh.sample());
}

#[test]
fn cfgns_setters_match_fresh_construction() {
  let chained = Cfgns::<f64, _>::new(0.4, -0.5, 16, Some(2.0), seed(1))
    .with_hurst(0.7)
    .with_rho(0.3)
    .with_steps(64)
    .with_horizon(Some(1.0))
    .with_seed(seed(7));
  let fresh = Cfgns::<f64, _>::new(0.7, 0.3, 64, Some(1.0), seed(7));

  assert_eq!(
    (chained.hurst(), chained.rho(), chained.n(), chained.t()),
    (0.7, 0.3, 64, Some(1.0))
  );
  assert_eq!(chained.seed().current(), 7);
  assert_eq!(chained.sample(), fresh.sample());
}

#[test]
fn cfou_setters_match_fresh_construction() {
  let chained = Cfou::<f64, _>::new(
    0.3,
    2.0,
    1.5,
    0.9,
    16,
    Some(0.2),
    Some(0.1),
    Some(2.0),
    seed(1),
  )
  .with_hurst(0.7)
  .with_lambda(1.0)
  .with_omega(0.5)
  .with_a(0.2)
  .with_steps(64)
  .with_x1_0(Some(0.0))
  .with_x2_0(Some(0.0))
  .with_horizon(Some(1.0))
  .with_seed(seed(7));
  let fresh = Cfou::<f64, _>::new(
    0.7,
    1.0,
    0.5,
    0.2,
    64,
    Some(0.0),
    Some(0.0),
    Some(1.0),
    seed(7),
  );

  assert_eq!(
    (
      chained.hurst(),
      chained.lambda(),
      chained.omega(),
      chained.a()
    ),
    (0.7, 1.0, 0.5, 0.2)
  );
  assert_eq!(
    (chained.n(), chained.x1_0(), chained.x2_0(), chained.t()),
    (64, Some(0.0), Some(0.0), Some(1.0))
  );
  assert_eq!(chained.seed().current(), 7);
  assert_eq!(chained.sample(), fresh.sample());
}

#[test]
fn jump_fou_setters_match_fresh_construction() {
  let chained = JumpFou::<f64, _, _>::new(
    0.3,
    2.0,
    0.5,
    0.4,
    1.0,
    jump_law(0.5),
    16,
    Some(0.1),
    Some(2.0),
    seed(1),
  )
  .with_seed(seed(7))
  .with_hurst(0.7)
  .with_theta(1.0)
  .with_mu(0.0)
  .with_sigma(0.2)
  .with_lambda(3.0)
  .with_jump_dist(jump_law(0.05))
  .with_steps(64)
  .with_x0(Some(0.0))
  .with_horizon(Some(1.0));
  let fresh = JumpFou::<f64, _, _>::new(
    0.7,
    1.0,
    0.0,
    0.2,
    3.0,
    jump_law(0.05),
    64,
    Some(0.0),
    Some(1.0),
    seed(7),
  );

  assert_eq!(
    (
      chained.hurst(),
      chained.theta(),
      chained.mu(),
      chained.sigma()
    ),
    (0.7, 1.0, 0.0, 0.2)
  );
  assert_eq!(
    (chained.lambda(), chained.n(), chained.x0(), chained.t()),
    (3.0, 64, Some(0.0), Some(1.0))
  );
  assert_eq!(chained.seed().current(), 7);
  assert_eq!(chained.sample(), fresh.sample());
}

#[test]
fn jump_fou_setters_keep_the_jump_driver_in_step() {
  let p = JumpFou::<f64, _, _>::new(
    0.7,
    1.0,
    0.0,
    0.2,
    2.0,
    jump_law(0.1),
    16,
    Some(0.0),
    Some(1.0),
    seed(1),
  );
  assert_eq!(p.cpoisson().poisson.lambda, 2.0);
  assert_eq!(p.cpoisson().poisson.n, Some(16));
  assert_eq!(p.cpoisson().poisson.t_max, Some(1.0));

  let p = p.with_lambda(3.0);
  assert_eq!(p.lambda(), 3.0);
  assert_eq!(p.cpoisson().poisson.lambda, 3.0);

  let p = p.with_steps(64);
  assert_eq!(p.cpoisson().poisson.n, Some(64));

  let p = p.with_horizon(Some(2.0));
  assert_eq!(p.cpoisson().poisson.t_max, Some(2.0));

  let p = p.with_seed(seed(7));
  assert_eq!(p.cpoisson().seed.current(), seed(7).derive().current());
}

#[test]
fn jump_fou_custom_setters_match_fresh_construction() {
  let chained = JumpFOUCustom::<f64, _, _>::new(
    0.3,
    2.0,
    0.5,
    0.4,
    16,
    Some(0.1),
    Some(2.0),
    SimdNormal::new(0.4, 0.01),
    jump_law(0.5),
    seed(1),
  )
  .with_hurst(0.7)
  .with_theta(1.0)
  .with_mu(0.0)
  .with_sigma(0.2)
  .with_steps(64)
  .with_x0(Some(0.0))
  .with_horizon(Some(1.0))
  .with_jump_times(SimdNormal::new(0.5, 0.01))
  .with_jump_sizes(jump_law(0.1))
  .with_seed(seed(7));
  let fresh = JumpFOUCustom::<f64, _, _>::new(
    0.7,
    1.0,
    0.0,
    0.2,
    64,
    Some(0.0),
    Some(1.0),
    SimdNormal::new(0.5, 0.01),
    jump_law(0.1),
    seed(7),
  );

  assert_eq!(
    (
      chained.hurst(),
      chained.theta(),
      chained.mu(),
      chained.sigma()
    ),
    (0.7, 1.0, 0.0, 0.2)
  );
  assert_eq!(
    (chained.n(), chained.x0(), chained.t()),
    (64, Some(0.0), Some(1.0))
  );
  assert_eq!(
    (chained.jump_times().mean(), chained.jump_sizes().std_dev()),
    (0.5, 0.1)
  );
  assert_eq!(chained.seed().current(), 7);
  assert_eq!(chained.sample(), fresh.sample());
}
