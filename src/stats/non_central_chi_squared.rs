use rand::Rng;
use rand_distr::ChiSquared;
use rand_distr::Distribution;
use rand_distr::Normal;

use crate::stochastic::Float;

pub fn sample<T: Float>(df: T, lambda: T, rng: &mut impl Rng) -> T {
  let chi_squared = ChiSquared::new(df).unwrap();
  let y = chi_squared.sample(rng);

  let normal = Normal::new(lambda.sqrt(), T::one()).unwrap();
  let z = normal.sample(rng);

  y + z * z
}
