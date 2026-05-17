//! # Cir
//!
//! $$
//! dX_t=\kappa(\theta-X_t)\,dt+\sigma\sqrt{X_t}\,dW_t
//! $$
//!
use num_complex::Complex64;
use scilib::math::bessel::i_nu;
use stochastic_rs_distributions::normal::SimdNormal;
use stochastic_rs_distributions::special::gamma;

/// Cox-Ingersoll-Ross (Cir) process future value.
pub fn sample(theta: f64, mu: f64, sigma: f64, t: f64, r_t: f64) -> f64 {
  let c = (2.0 * theta) / ((1.0 - (-theta * t).exp()) * sigma.powi(2));

  let lambda = 2.0 * c * r_t * (-theta * t).exp();
  let df = ((4.0 * theta * mu) / sigma.powi(2)) as usize;

  let normal = SimdNormal::<f64>::new(0.0, 1.0, &stochastic_rs_core::simd_rng::Unseeded);
  let chi2 = (0..df)
    .map(|_| {
      let z = normal.sample_fast();
      (z + lambda.sqrt()).powi(2)
    })
    .sum::<f64>();

  chi2 / (2.0 * c)
}

/// Cox-Ingersoll-Ross (Cir) process PDF.
pub fn pdf(theta: f64, mu: f64, sigma: f64, t: f64, r_t: f64, r_T: f64) -> f64 {
  let c = (2.0 * theta) / ((1.0 - (-theta * t).exp()) * sigma.powi(2));
  let q = (2.0 * theta * mu) / sigma.powi(2) - 1.0;
  let u = c * r_t * (-theta * t).exp();
  let v = c * r_T;
  let Iq = i_nu(q, Complex64::new(2.0 * (u * v).sqrt(), 0.0));

  c * (-u - v).exp() * (u / v).powf(q / 2.0) * Iq.re
}

/// Cox-Ingersoll-Ross (Cir) process Asymptotic PDF.
pub fn apdf(theta: f64, mu: f64, sigma: f64, r_t: f64) -> f64 {
  let beta = 2.0 * theta / sigma.powi(2);
  let alpha = 2.0 * theta * mu / sigma.powi(2);

  (beta.powf(alpha) / gamma(alpha)) * r_t.powf(alpha - 1.0) * (-beta * r_t).exp()
}
