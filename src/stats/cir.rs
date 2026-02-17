//! # CIR
//!
//! $$
//! dX_t=\kappa(\theta-X_t)\,dt+\sigma\sqrt{X_t}\,dW_t
//! $$
//!
use num_complex::Complex64;
use rand::rng;
use rand::Rng;
use rand_distr::StandardNormal;
use scilib::math::bessel::i_nu;
use statrs::function::gamma::gamma;

/// Cox-Ingersoll-Ross (CIR) process future value.
pub fn sample(theta: f64, mu: f64, sigma: f64, t: f64, r_t: f64) -> f64 {
  let c = (2.0 * theta) / ((1.0 - (-theta * t).exp()) * sigma.powi(2));

  let lambda = 2.0 * c * r_t * (-theta * t).exp();
  let df = ((4.0 * theta * mu) / sigma.powi(2)) as usize;

  let mut rng = rng();
  let chi2 = (0..df)
    .map(|_| {
      let z: f64 = rng.sample(StandardNormal);
      (z + lambda.sqrt()).powi(2)
    })
    .sum::<f64>();

  chi2 / (2.0 * c)
}

/// Cox-Ingersoll-Ross (CIR) process PDF.
pub fn pdf(theta: f64, mu: f64, sigma: f64, t: f64, r_t: f64, r_T: f64) -> f64 {
  let c = (2.0 * theta) / ((1.0 - (-theta * t).exp()) * sigma.powi(2));
  let q = (2.0 * theta * mu) / sigma.powi(2) - 1.0;
  let u = c * r_t * (-theta * t).exp();
  let v = c * r_T;
  let Iq = i_nu(q, Complex64::new(2.0 * (u * v).sqrt(), 0.0));

  c * (-u - v).exp() * (u / v).powf(q / 2.0) * Iq.re
}

/// Cox-Ingersoll-Ross (CIR) process Asymptotic PDF.
pub fn apdf(theta: f64, mu: f64, sigma: f64, r_t: f64) -> f64 {
  let beta = 2.0 * theta / sigma.powi(2);
  let alpha = 2.0 * theta * mu / sigma.powi(2);

  (beta.powf(alpha) / gamma(alpha)) * r_t.powf(alpha - 1.0) * (-beta * r_t).exp()
}
