use impl_new_derive::ImplNew;
use rand::Rng;
use rand_distr::Distribution;

#[derive(ImplNew)]
pub struct DoubleExp {
  pub p: Option<f64>,
  pub lambda_plus: f64,
  pub lambda_minus: f64,
}

impl Distribution<f64> for DoubleExp {
  fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
    let u = rng.random::<f64>();

    if u < self.p.unwrap_or(0.5) {
      -rng.random::<f64>().ln() / self.lambda_plus
    } else {
      rng.random::<f64>().ln() / self.lambda_minus
    }
  }
}
