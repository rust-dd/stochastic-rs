use rand::Rng;
use rand_distr::Distribution;

use crate::distributions::chi_square::SimdChiSquared;
use crate::distributions::normal::SimdNormal;
use crate::stochastic::Float;

pub fn sample<T: Float>(df: T, lambda: T, rng: &mut impl Rng) -> T {
  let chi_squared = SimdChiSquared::new(df);
  let y = chi_squared.sample(rng);

  let normal = SimdNormal::<T, 64>::new(lambda.sqrt(), T::one());
  let z = normal.sample(rng);

  y + z * z
}
