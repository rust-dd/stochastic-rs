use ndarray::array;
use stochastic_rs_core::simd_rng::SimdRng;
use stochastic_rs_stats::filtering::particle::gaussian_random_walk_transition;

const PIN: [u64; 2] = [4607382133634470021, 4612166311110089933];

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
