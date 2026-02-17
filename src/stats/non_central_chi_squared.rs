//! # Non Central Chi Squared
//!
//! $$
//! X\sim\chi^2_\nu(\lambda),\quad f_X(x)=\tfrac12 e^{-(x+\lambda)/2}(x/\lambda)^{\nu/4-1/2}I_{\nu/2-1}(\sqrt{\lambda x})
//! $$
//!
use rand::Rng;
use rand_distr::Distribution;

use crate::distributions::chi_square::SimdChiSquared;
use crate::distributions::normal::SimdNormal;
use crate::traits::FloatExt;

pub fn sample<T: FloatExt>(df: T, lambda: T, rng: &mut impl Rng) -> T {
  let chi_squared = SimdChiSquared::new(df);
  let y = chi_squared.sample(rng);

  let normal = SimdNormal::<T, 64>::new(lambda.sqrt(), T::one());
  let z = normal.sample(rng);

  y + z * z
}