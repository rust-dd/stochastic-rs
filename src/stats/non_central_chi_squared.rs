use rand::Rng;
use rand_distr::ChiSquared;
use rand_distr::Distribution;
use rand_distr::Normal;

pub fn sample(df: f64, lambda: f64, rng: &mut impl Rng) -> f64 {
  let chi_squared = ChiSquared::new(df).unwrap();
  let y = chi_squared.sample(rng);

  let normal = Normal::new(lambda.sqrt(), 1.0).unwrap();
  let z = normal.sample(rng);

  y + z * z
}
