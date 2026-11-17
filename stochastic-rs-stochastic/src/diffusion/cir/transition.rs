//! The exact transition law of [`Cir`](super::Cir): a scaled noncentral χ² in the starting state.
//!
//! $$
//! X_{s+t}\mid X_s=r\ \sim\ \frac{\sigma^2(1-e^{-\theta t})}{4\theta}\,\chi'^2_{4\theta\mu/\sigma^2}\!\left(\frac{4\theta e^{-\theta t}\,r}{\sigma^2(1-e^{-\theta t})}\right)
//! $$
//!
//! Cox, Ingersoll, Ross (1985), Econometrica 53(2) 385–407, DOI 10.2307/1911242.

use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_distributions::non_central_chi_squared;
use stochastic_rs_distributions::special::gamma;
use stochastic_rs_distributions::special::ln_bessel_ie;

pub(crate) fn transition_pdf(
  theta: f64,
  mu: f64,
  sigma: f64,
  t: f64,
  r_t: f64,
  r_next: f64,
) -> f64 {
  if !valid_parameters(theta, mu, sigma, t, r_t) || t == 0.0 || !r_next.is_finite() || r_next < 0.0
  {
    return f64::NAN;
  }
  let decay = (-theta * t).exp();
  let one_minus_decay = -(-theta * t).exp_m1();
  let c = 2.0 * theta / (one_minus_decay * sigma.powi(2));
  let q = (2.0 * theta * mu) / sigma.powi(2) - 1.0;
  let u = c * r_t * decay;
  let v = c * r_next;
  if u == 0.0 {
    return central_transition_pdf(c, q + 1.0, r_next);
  }
  if v == 0.0 {
    return match q.total_cmp(&0.0) {
      std::cmp::Ordering::Less => f64::INFINITY,
      std::cmp::Ordering::Equal => c * (-u).exp(),
      std::cmp::Ordering::Greater => 0.0,
    };
  }
  let (root_u, root_v) = (u.sqrt(), v.sqrt());
  let gap = root_u - root_v;
  let (ratio, product) = (v / u, u * v);
  // Taken apart where `v / u` overflows or `u · v` loses digits, as `r_t` nears the subnormal range.
  let (log_ratio, root) = if ratio.is_normal() && product.is_normal() {
    (ratio.ln(), product.sqrt())
  } else {
    (v.ln() - u.ln(), root_u * root_v)
  };
  (c.ln() - gap * gap + 0.5 * q * log_ratio + ln_bessel_ie(q, 2.0 * root)).exp()
}

pub(crate) fn stationary_pdf(theta: f64, mu: f64, sigma: f64, r: f64) -> f64 {
  if !valid_parameters(theta, mu, sigma, 0.0, r) {
    return f64::NAN;
  }
  let beta = 2.0 * theta / sigma.powi(2);
  let alpha = 2.0 * theta * mu / sigma.powi(2);

  (beta.powf(alpha) / gamma(alpha)) * r.powf(alpha - 1.0) * (-beta * r).exp()
}

pub(crate) fn sample_transition<S: SeedExt>(
  theta: f64,
  mu: f64,
  sigma: f64,
  t: f64,
  r_t: f64,
  seed: &S,
) -> f64 {
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
  if !(degrees_of_freedom + noncentrality).is_finite() {
    // Past the double range the relative spread `2/√(df + ncp)` is below 1.5e-154: draw the mean.
    return r_t * decay + mu * one_minus_decay;
  }
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
