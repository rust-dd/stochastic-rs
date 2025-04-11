use rand::Rng;
use rand_distr::Distribution;

use super::gamma::SimdGamma;

pub struct SimdChiSquared {
  df: f32,
  gamma: SimdGamma, // chi-sq(k) = gamma(k/2, 2)
}

impl SimdChiSquared {
  pub fn new(k: f32) -> Self {
    Self {
      df: k,
      gamma: SimdGamma::new(k * 0.5, 2.0),
    }
  }
}

impl Distribution<f32> for SimdChiSquared {
  fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f32 {
    self.gamma.sample(rng)
  }
}
