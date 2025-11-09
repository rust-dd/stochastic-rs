use statrs::distribution::{ContinuousCDF, Normal};

use crate::quant::{
  pricing::bsm::{BSMCoc, BSMPricer},
  r#trait::PricerExt,
  OptionType,
};

/// Forward FX F = S * exp((r_d - r_f) T)
pub fn forward_fx(s: f64, tau: f64, r_d: f64, r_f: f64) -> f64 {
  s * ((r_d - r_f) * tau).exp()
}

/// Hagan et al. (2002) implied vol approximation for beta = 1 (lognormal SABR)
/// Matches the Python reference provided by the user.
pub fn hagan_implied_vol_beta1(k: f64, f: f64, tau: f64, alpha: f64, nu: f64, rho: f64) -> f64 {
  if (k - f).abs() < 1e-12 {
    return alpha * (1.0 + tau * (rho * nu * alpha / 4.0 + (2.0 - 3.0 * rho * rho) * nu * nu / 24.0));
  }

  let z = (nu / alpha) * (f / k).ln();
  let sqrt_term = (1.0 - 2.0 * rho * z + z * z).sqrt();
  let x_z = ((sqrt_term + z - rho) / (1.0 - rho)).ln();

  if x_z.abs() < 1e-14 {
    return alpha;
  }

  let base = alpha * z / x_z;
  base * (1.0 + tau * (rho * nu * alpha / 4.0 + (2.0 - 3.0 * rho * rho) * nu * nu / 24.0))
}

/// Black-Scholes(-Garman-Kohlhagen) price for FX under GK using BSMPricer.
pub fn bs_price_fx(s: f64, k: f64, r_d: f64, r_f: f64, tau: f64, sigma: f64) -> (f64, f64) {
  let pricer = BSMPricer::new(
    s,
    sigma,
    k,
    r_d,
    Some(r_d),
    Some(r_f),
    None,
    Some(tau),
    None,
    None,
    OptionType::Call,
    BSMCoc::GARMAN1983,
  );
  pricer.calculate_call_put()
}

/// Delta on forward with premium included, as in the Python code
pub fn fx_delta_from_forward(k: f64, f: f64, sigma: f64, tau: f64, r_f: f64, phi: f64) -> f64 {
  let d2 = (f / k).ln() / (sigma * tau.sqrt()) - 0.5 * sigma * tau.sqrt();
  let n = Normal::new(0.0, 1.0).unwrap();
  let nd2 = n.cdf(phi * d2);
  phi * (-r_f * tau).exp() * (k / f) * nd2
}

/// Model price computed by plugging Hagan approx vol into BS(GK)
pub fn model_price_hagan(
  s: f64,
  k: f64,
  r_d: f64,
  r_f: f64,
  tau: f64,
  alpha: f64,
  nu: f64,
  rho: f64,
) -> (f64, f64) {
  let fwd = forward_fx(s, tau, r_d, r_f);
  let sigma = hagan_implied_vol_beta1(k, fwd, tau, alpha, nu, rho);
  bs_price_fx(s, k, r_d, r_f, tau, sigma)
}