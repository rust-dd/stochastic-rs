//! Each cache-feeding setter alone, so a later setter's rebuild in a chain cannot hide one that
//! forgot its own.

use stochastic_rs_distributions::scalar::ScalarNormal;
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

use super::common::jump_law;
use super::common::seed;

#[test]
fn fou_cache_setters_match_fresh_construction() {
  let make = |hurst, n, t| Fou::<f64, _>::new(hurst, 1.0, 0.0, 0.2, n, Some(0.0), t, seed(7));
  let want = make(0.7, 64, Some(1.0)).sample();

  assert_eq!(make(0.3, 64, Some(1.0)).with_hurst(0.7).sample(), want);
  assert_eq!(make(0.7, 16, Some(1.0)).with_steps(64).sample(), want);
  assert_eq!(
    make(0.7, 64, Some(2.0)).with_horizon(Some(1.0)).sample(),
    want
  );
}

#[test]
fn fgbm_cache_setters_match_fresh_construction() {
  let make = |hurst, n, t| Fgbm::<f64, _>::new(hurst, 0.05, 0.2, n, Some(1.0), t, seed(7));
  let want = make(0.7, 64, Some(1.0)).sample();

  assert_eq!(make(0.3, 64, Some(1.0)).with_hurst(0.7).sample(), want);
  assert_eq!(make(0.7, 16, Some(1.0)).with_steps(64).sample(), want);
  assert_eq!(
    make(0.7, 64, Some(2.0)).with_horizon(Some(1.0)).sample(),
    want
  );
}

#[test]
fn fcir_cache_setters_match_fresh_construction() {
  let make = |hurst, n, t| {
    Fcir::<f64, _>::new(
      hurst,
      1.0,
      0.04,
      0.1,
      n,
      Some(0.04),
      t,
      Some(false),
      seed(7),
    )
  };
  let want = make(0.7, 64, Some(1.0)).sample();

  assert_eq!(make(0.3, 64, Some(1.0)).with_hurst(0.7).sample(), want);
  assert_eq!(make(0.7, 16, Some(1.0)).with_steps(64).sample(), want);
  assert_eq!(
    make(0.7, 64, Some(2.0)).with_horizon(Some(1.0)).sample(),
    want
  );
}

#[test]
fn fjacobi_cache_setters_match_fresh_construction() {
  let make = |hurst, n, t| FJacobi::<f64, _>::new(hurst, 1.0, 2.0, 0.2, n, Some(0.5), t, seed(7));
  let want = make(0.7, 64, Some(1.0)).sample();

  assert_eq!(make(0.3, 64, Some(1.0)).with_hurst(0.7).sample(), want);
  assert_eq!(make(0.7, 16, Some(1.0)).with_steps(64).sample(), want);
  assert_eq!(
    make(0.7, 64, Some(2.0)).with_horizon(Some(1.0)).sample(),
    want
  );
}

#[test]
fn cfbms_cache_setters_match_fresh_construction() {
  let make = |hurst, n, t| Cfbms::<f64, _>::new(hurst, 0.3, n, t, seed(7));
  let want = make(0.7, 64, Some(1.0)).sample();

  assert_eq!(make(0.3, 64, Some(1.0)).with_hurst(0.7).sample(), want);
  assert_eq!(make(0.7, 16, Some(1.0)).with_steps(64).sample(), want);
  assert_eq!(
    make(0.7, 64, Some(2.0)).with_horizon(Some(1.0)).sample(),
    want
  );
}

#[test]
fn cfgns_cache_setters_match_fresh_construction() {
  let make = |hurst, n, t| Cfgns::<f64, _>::new(hurst, 0.3, n, t, seed(7));
  let want = make(0.7, 64, Some(1.0)).sample();

  assert_eq!(make(0.3, 64, Some(1.0)).with_hurst(0.7).sample(), want);
  assert_eq!(make(0.7, 16, Some(1.0)).with_steps(64).sample(), want);
  assert_eq!(
    make(0.7, 64, Some(2.0)).with_horizon(Some(1.0)).sample(),
    want
  );
}

#[test]
fn cfou_cache_setters_match_fresh_construction() {
  let make =
    |hurst, n, t| Cfou::<f64, _>::new(hurst, 1.0, 0.5, 0.2, n, Some(0.0), Some(0.0), t, seed(7));
  let want = make(0.7, 64, Some(1.0)).sample();

  assert_eq!(make(0.3, 64, Some(1.0)).with_hurst(0.7).sample(), want);
  assert_eq!(make(0.7, 16, Some(1.0)).with_steps(64).sample(), want);
  assert_eq!(
    make(0.7, 64, Some(2.0)).with_horizon(Some(1.0)).sample(),
    want
  );
}

#[test]
fn jump_fou_cache_setters_match_fresh_construction() {
  let make = |hurst, n, t, seed_value| {
    JumpFou::<f64, _, _>::new(
      hurst,
      1.0,
      0.0,
      0.2,
      3.0,
      jump_law(0.05),
      n,
      Some(0.0),
      t,
      seed(seed_value),
    )
  };
  let want = make(0.7, 64, Some(1.0), 7).sample();

  assert_eq!(make(0.3, 64, Some(1.0), 7).with_hurst(0.7).sample(), want);
  assert_eq!(make(0.7, 16, Some(1.0), 7).with_steps(64).sample(), want);
  assert_eq!(
    make(0.7, 64, Some(2.0), 7).with_horizon(Some(1.0)).sample(),
    want
  );
  assert_eq!(
    make(0.7, 64, Some(1.0), 1).with_seed(seed(7)).sample(),
    want
  );
}

#[test]
fn jump_fou_custom_cache_setters_match_fresh_construction() {
  let make = |hurst, n, t| {
    JumpFOUCustom::<f64, _, _>::new(
      hurst,
      1.0,
      0.0,
      0.2,
      n,
      Some(0.0),
      t,
      ScalarNormal::new(0.5, 0.01),
      jump_law(0.1),
      seed(7),
    )
  };
  let want = make(0.7, 64, Some(1.0)).sample();

  assert_eq!(make(0.3, 64, Some(1.0)).with_hurst(0.7).sample(), want);
  assert_eq!(make(0.7, 16, Some(1.0)).with_steps(64).sample(), want);
  assert_eq!(
    make(0.7, 64, Some(2.0)).with_horizon(Some(1.0)).sample(),
    want
  );
}
