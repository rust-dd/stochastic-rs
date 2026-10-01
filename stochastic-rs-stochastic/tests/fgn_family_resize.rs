use ndarray::Array1;
use stochastic_rs_core::simd_rng::Deterministic;
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
use stochastic_rs_stochastic::process::fbm::Fbm;

fn assert_resized(path: &Array1<f64>, n: usize) {
  assert_eq!(path.len(), n);
  assert!(
    path.iter().all(|v| v.is_finite()),
    "uninitialized or non-finite values after resize"
  );
}

fn seed() -> Deterministic {
  Deterministic::new(7)
}

#[test]
fn fbm_resizes_through_with_steps() {
  let p = Fbm::<f64, _>::new(0.7, 10, Some(1.0), seed()).with_steps(1000);
  assert_eq!(p.n(), 1000);
  assert_resized(&p.sample(), 1000);
}

#[test]
fn fou_resizes_through_with_steps() {
  let p = Fou::<f64, _>::new(0.7, 1.0, 0.0, 0.2, 10, Some(0.0), Some(1.0), seed()).with_steps(1000);
  assert_resized(&p.sample(), 1000);
}

#[test]
fn fgbm_resizes_through_with_steps() {
  let p = Fgbm::<f64, _>::new(0.7, 0.05, 0.2, 10, Some(1.0), Some(1.0), seed()).with_steps(1000);
  assert_resized(&p.sample(), 1000);
}

#[test]
fn fcir_resizes_through_with_steps() {
  let p = Fcir::<f64, _>::new(
    0.7,
    1.0,
    0.04,
    0.1,
    10,
    Some(0.04),
    Some(1.0),
    Some(false),
    seed(),
  )
  .with_steps(1000);
  assert_resized(&p.sample(), 1000);
}

#[test]
fn fjacobi_resizes_through_with_steps() {
  let p =
    FJacobi::<f64, _>::new(0.7, 1.0, 2.0, 0.2, 10, Some(0.5), Some(1.0), seed()).with_steps(1000);
  assert_resized(&p.sample(), 1000);
}

#[test]
fn cfbms_resizes_through_with_steps() {
  let p = Cfbms::<f64, _>::new(0.7, 0.3, 10, Some(1.0), seed()).with_steps(1000);
  let [a, b] = p.sample();
  assert_resized(&a, 1000);
  assert_resized(&b, 1000);
}

#[test]
fn cfgns_resizes_through_with_steps() {
  let p = Cfgns::<f64, _>::new(0.7, 0.3, 10, Some(1.0), seed()).with_steps(1000);
  let [a, b] = p.sample();
  assert_resized(&a, 1000);
  assert_resized(&b, 1000);
}

#[test]
fn cfou_resizes_through_with_steps() {
  let p = Cfou::<f64, _>::new(
    0.7,
    1.0,
    0.5,
    0.2,
    10,
    Some(0.0),
    Some(0.0),
    Some(1.0),
    seed(),
  )
  .with_steps(1000);
  let path = p.sample();
  assert_eq!(path.len(), 1000);
  assert!(path.iter().all(|z| z.re.is_finite() && z.im.is_finite()));
}

#[test]
fn jump_fou_resizes_through_with_steps() {
  let law = ScalarNormal::<f64>::new(0.0, 0.1);
  let p = JumpFou::<f64, _, _>::new(
    0.7,
    1.0,
    0.0,
    0.2,
    2.0,
    law,
    10,
    Some(0.0),
    Some(1.0),
    seed(),
  )
  .with_steps(1000);
  assert_resized(&p.sample(), 1000);
}

#[test]
fn jump_fou_custom_resizes_through_with_steps() {
  let times = ScalarNormal::<f64>::new(0.5, 0.01);
  let sizes = ScalarNormal::<f64>::new(0.0, 0.1);
  let p = JumpFOUCustom::<f64, _, _>::new(
    0.7,
    1.0,
    0.0,
    0.2,
    10,
    Some(0.0),
    Some(1.0),
    times,
    sizes,
    seed(),
  )
  .with_steps(1000);
  assert_resized(&p.sample(), 1000);
}

#[test]
fn a_new_hurst_changes_the_paths() {
  let a = Fbm::<f64, _>::new(0.3, 256, Some(1.0), seed());
  let b = Fbm::<f64, _>::new(0.3, 256, Some(1.0), seed()).with_hurst(0.8);
  assert_ne!(
    a.sample(),
    b.sample(),
    "with_hurst must rebuild the fGN eigenvalues"
  );
}
