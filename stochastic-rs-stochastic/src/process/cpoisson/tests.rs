use rand::distr::Distribution;

use super::*;

#[derive(Clone, Copy)]
struct ConstJump<T: FloatExt>(T);

impl<T: FloatExt> Distribution<T> for ConstJump<T> {
  fn sample<R: rand::Rng + ?Sized>(&self, _rng: &mut R) -> T {
    self.0
  }
}

#[test]
fn the_output_names_its_slots() {
  let CompoundPaths {
    arrivals,
    cumulative,
    jumps,
    ..
  } = CompoundPoisson::new(
    ConstJump(1.0f64),
    Poisson::new(2.0, Horizon::Events(16), Unseeded),
    Unseeded,
  )
  .sample();
  assert_eq!(
    (arrivals.len(), cumulative.len(), jumps.len()),
    (17, 17, 17)
  );
}

#[test]
fn grid_increments_zero_for_zero_intensity() {
  let cp = CompoundPoisson::new(
    ConstJump(1.0f64),
    Poisson::new(0.0, Horizon::Until(1.0), Unseeded),
    Unseeded,
  );
  let inc = cp.sample_grid_increments(16, 1.0 / 15.0);
  assert_eq!(inc.len(), 16);
  assert!(inc.iter().all(|&x| x == 0.0));
}

#[test]
fn a_zero_rate_compounds_nothing_before_its_horizon() {
  let cp = CompoundPoisson::new(
    ConstJump(1.0f64),
    Poisson::new(0.0, Horizon::Until(1.0), Unseeded),
    Unseeded,
  );
  let origin = Array1::from(vec![0.0]);
  assert_eq!(
    cp.sample(),
    CompoundPaths::from([origin.clone(), origin.clone(), origin])
  );
}

#[test]
fn grid_increments_start_at_zero() {
  let cp = CompoundPoisson::new(
    ConstJump(1.0f64),
    Poisson::new(2.0, Horizon::Events(16), Unseeded),
    Unseeded,
  );
  let inc = cp.sample_grid_increments(16, 1.0 / 15.0);
  assert_eq!(inc[0], 0.0);
}

#[test]
fn relative_increment_compounds_multiple_jumps() {
  let mut rng = crate::simd_rng::rng();
  let rel = relative_jump_from_count(&ConstJump(0.1f64), 3, &mut rng);
  let expected = 1.1f64.powi(3) - 1.0;
  assert!((rel - expected).abs() < 1e-12);
}
