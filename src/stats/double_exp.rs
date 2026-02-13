use rand::Rng;
use rand_distr::Distribution;

pub struct DoubleExp {
  pub p: Option<f64>,
  pub lambda_plus: f64,
  pub lambda_minus: f64,
}

impl DoubleExp {
  pub fn new(p: Option<f64>, lambda_plus: f64, lambda_minus: f64) -> Self {
    Self { p, lambda_plus, lambda_minus }
  }
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
