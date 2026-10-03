use ndarray::array;
use stochastic_rs_core::simd_rng::SimdRng;
use stochastic_rs_stats::filtering::particle::gaussian_random_walk_transition;

const PIN: [u64; 2] = [4607775976925510098, 4610527332073300406];

fn step(seed: u64) -> Vec<u64> {
  let transition = gaussian_random_walk_transition(array![0.1, 0.2]);
  let mut rng = SimdRng::from_seed(seed);
  transition(array![1.0, 2.0].view(), &mut rng)
    .iter()
    .map(|x| x.to_bits())
    .collect()
}

#[test]
fn the_step_replays_and_depends_on_the_rng() {
  assert_eq!(step(5), step(5));
  assert_ne!(step(5), step(6));
}

#[test]
fn the_step_is_pinned() {
  assert_eq!(step(5), PIN);
}

#[test]
#[should_panic(expected = "scales[1] must satisfy `scales[1] > 0`, got scales[1] = 0")]
fn a_non_positive_scale_is_named_when_the_transition_is_built() {
  let _ = gaussian_random_walk_transition(array![0.1, 0.0]);
}
