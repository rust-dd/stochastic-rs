//! # Cir
//!
//! $$
//! dX_t=\theta(\mu-X_t)\,dt+\sigma\sqrt{X_t}\,dW_t
//! $$
//!
//! Exact transition sampling and density via the scaled noncentral chi-squared
//! law of the process.
//!
//! Reference: Cox, J.C., Ingersoll, J.E., Ross, S.A. (1985), "A Theory of the
//! Term Structure of Interest Rates", *Econometrica* 53(2), 385-407.
//! DOI: 10.2307/1911242
//!
use num_complex::Complex64;
use scilib::math::bessel::i_nu;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::non_central_chi_squared;
use stochastic_rs_distributions::special::gamma;

/// Samples one exact Cox-Ingersoll-Ross transition with an entropy-seeded RNG.
///
/// `theta` is the mean-reversion speed and `mu` is the long-run mean. Invalid
/// or non-finite inputs return `NaN`; at `t == 0` the current state is returned.
pub fn sample(theta: f64, mu: f64, sigma: f64, t: f64, r_t: f64) -> f64 {
  sample_with_seed(theta, mu, sigma, t, r_t, &Unseeded)
}

/// Samples one reproducible exact Cox-Ingersoll-Ross transition.
pub fn sample_seeded(theta: f64, mu: f64, sigma: f64, t: f64, r_t: f64, seed: u64) -> f64 {
  sample_with_seed(theta, mu, sigma, t, r_t, &Deterministic::new(seed))
}

/// Exact Cox-Ingersoll-Ross transition density.
pub fn pdf(theta: f64, mu: f64, sigma: f64, t: f64, r_t: f64, future_state: f64) -> f64 {
  if !valid_parameters(theta, mu, sigma, t, r_t)
    || t == 0.0
    || !future_state.is_finite()
    || future_state < 0.0
  {
    return f64::NAN;
  }
  let decay = (-theta * t).exp();
  let one_minus_decay = -(-theta * t).exp_m1();
  let c = 2.0 * theta / (one_minus_decay * sigma.powi(2));
  let q = (2.0 * theta * mu) / sigma.powi(2) - 1.0;
  let u = c * r_t * decay;
  let v = c * future_state;
  if u == 0.0 {
    return central_transition_pdf(c, q + 1.0, future_state);
  }
  if v == 0.0 {
    return match q.total_cmp(&0.0) {
      std::cmp::Ordering::Less => f64::INFINITY,
      std::cmp::Ordering::Equal => c * (-u).exp(),
      std::cmp::Ordering::Greater => 0.0,
    };
  }
  let bessel = i_nu(q, Complex64::new(2.0 * (u * v).sqrt(), 0.0));

  c * (-u - v).exp() * (v / u).powf(q / 2.0) * bessel.re
}

/// Cox-Ingersoll-Ross (Cir) process Asymptotic PDF.
pub fn apdf(theta: f64, mu: f64, sigma: f64, r_t: f64) -> f64 {
  let beta = 2.0 * theta / sigma.powi(2);
  let alpha = 2.0 * theta * mu / sigma.powi(2);

  (beta.powf(alpha) / gamma(alpha)) * r_t.powf(alpha - 1.0) * (-beta * r_t).exp()
}

fn sample_with_seed<S: SeedExt>(theta: f64, mu: f64, sigma: f64, t: f64, r_t: f64, seed: &S) -> f64 {
  if !valid_parameters(theta, mu, sigma, t, r_t) {
    return f64::NAN;
  }
  if t == 0.0 {
    return r_t;
  }
  let decay = (-theta * t).exp();
  let one_minus_decay = -(-theta * t).exp_m1();
  let sigma_squared = sigma * sigma;
  let scale = sigma_squared * one_minus_decay / (4.0 * theta);
  let degrees_of_freedom = 4.0 * theta * mu / sigma_squared;
  let noncentrality = r_t * decay / scale;
  scale * non_central_chi_squared::sample(degrees_of_freedom, noncentrality, seed)
}

fn valid_parameters(theta: f64, mu: f64, sigma: f64, t: f64, r_t: f64) -> bool {
  theta.is_finite()
    && theta > 0.0
    && mu.is_finite()
    && mu > 0.0
    && sigma.is_finite()
    && sigma > 0.0
    && t.is_finite()
    && t >= 0.0
    && r_t.is_finite()
    && r_t >= 0.0
}

fn central_transition_pdf(c: f64, shape: f64, value: f64) -> f64 {
  if value == 0.0 {
    return match shape.total_cmp(&1.0) {
      std::cmp::Ordering::Less => f64::INFINITY,
      std::cmp::Ordering::Equal => c,
      std::cmp::Ordering::Greater => 0.0,
    };
  }
  c.powf(shape) * value.powf(shape - 1.0) * (-c * value).exp() / gamma(shape)
}

#[cfg(test)]
#[path = "cir_tests.rs"]
mod tests;
