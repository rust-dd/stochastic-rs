use stochastic_rs_distributions::special::norm_cdf;

use crate::OptionType;
use crate::pricing::bsm::BSMCoc;
use crate::pricing::bsm::BSMPricer;
use crate::traits::PricerExt;

/// Forward FX F = S * exp((r_d - r_f) T)
pub fn forward_fx(s: f64, tau: f64, r_d: f64, r_f: f64) -> f64 {
  s * ((r_d - r_f) * tau).exp()
}

/// Hagan et al. (2002) Sabr implied-vol approximation for general β.
///
/// Implements Eq. A.69a from:
/// P. S. Hagan, D. Kumar, A. S. Lesniewski, D. E. Woodward,
/// *Managing Smile Risk*, Wilmott Magazine, pp. 84–108, 2002.
///
/// $$
/// \sigma(K)\approx
/// \frac{\alpha}{(FK)^{(1-\beta)/2}\!\bigl[1+\tfrac{(1-\beta)^2}{24}\ln^2\!\tfrac FK
/// +\tfrac{(1-\beta)^4}{1920}\ln^4\!\tfrac FK\bigr]}
/// \;\frac{z}{x(z)}
/// \;\Bigl[1+\Bigl(\tfrac{(1-\beta)^2\alpha^2}{24(FK)^{1-\beta}}
/// +\tfrac{\rho\beta\nu\alpha}{4(FK)^{(1-\beta)/2}}
/// +\tfrac{(2-3\rho^2)\nu^2}{24}\Bigr)\tau\Bigr]
/// $$
pub fn hagan_implied_vol(
  k: f64,
  f: f64,
  tau: f64,
  alpha: f64,
  beta: f64,
  nu: f64,
  rho: f64,
) -> f64 {
  if k <= 0.0 || f <= 0.0 || alpha <= 0.0 {
    return 0.0;
  }

  let eps = 1e-07;
  let logfk = (f / k).ln();
  let fkbeta = (f * k).powf(1.0 - beta);
  let a = (1.0 - beta).powi(2) * alpha * alpha / (24.0 * fkbeta);
  let b = 0.25 * rho * beta * nu * alpha / fkbeta.sqrt();
  let c = (2.0 - 3.0 * rho * rho) * nu * nu / 24.0;
  let d = fkbeta.sqrt();
  let v = (1.0 - beta).powi(2) * logfk * logfk / 24.0;
  let w = (1.0 - beta).powi(4) * logfk.powi(4) / 1920.0;
  let z = nu * fkbeta.sqrt() * logfk / alpha;

  if z.abs() > eps {
    let arg = (1.0 - 2.0 * rho * z + z * z).sqrt() + z - rho;
    let xz = (arg / (1.0 - rho)).ln();
    if xz.abs() < 1e-14 {
      return alpha * (1.0 + (a + b + c) * tau) / (d * (1.0 + v + w));
    }
    alpha * z * (1.0 + (a + b + c) * tau) / (d * (1.0 + v + w) * xz)
  } else {
    alpha * (1.0 + (a + b + c) * tau) / (d * (1.0 + v + w))
  }
}

/// Convenience wrapper: Hagan (2002) with β = 1.
#[inline]
pub fn hagan_implied_vol_beta1(k: f64, f: f64, tau: f64, alpha: f64, nu: f64, rho: f64) -> f64 {
  hagan_implied_vol(k, f, tau, alpha, 1.0, nu, rho)
}

/// Compute Sabr α from an ATM lognormal vol by solving the Hagan (2002) ATM
/// condition (Eq. A.69b) as a cubic polynomial in α (quadratic when β = 1).
///
/// $$
/// \frac{(1-\beta)^2 f_*^3}{24}\,\tau\;\alpha^3
/// +\frac{\rho\beta\nu\,f_*^2}{4}\,\tau\;\alpha^2
/// +\bigl(1+\tfrac{(2-3\rho^2)\nu^2}{24}\,\tau\bigr)\,f_*\;\alpha
/// -\sigma_{ATM}=0, \quad f_*=F^{\beta-1}
/// $$
pub fn alpha_from_atm_vol(v_atm: f64, f: f64, tau: f64, beta: f64, rho: f64, nu: f64) -> f64 {
  let f_ = f.powf(beta - 1.0);
  let p3 = tau * f_.powi(3) * (1.0 - beta).powi(2) / 24.0;
  let p2 = tau * f_.powi(2) * rho * beta * nu / 4.0;
  let p1 = (1.0 + tau * nu * nu * (2.0 - 3.0 * rho * rho) / 24.0) * f_;
  let p0 = -v_atm;

  // Newton's method from initial guess α ≈ σ_ATM · F^{1−β}
  let mut x = v_atm * f.powf(1.0 - beta);
  for _ in 0..100 {
    let fx = p3 * x * x * x + p2 * x * x + p1 * x + p0;
    let dfx = 3.0 * p3 * x * x + 2.0 * p2 * x + p1;
    if dfx.abs() < 1e-15 {
      break;
    }
    let dx = fx / dfx;
    x -= dx;
    if dx.abs() < 1e-12 * x.abs().max(1e-12) {
      break;
    }
  }
  x.max(1e-8)
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
    BSMCoc::GarmanKohlhagen1983,
  );
  pricer.calculate_call_put()
}

/// Delta on forward with premium included
pub fn fx_delta_from_forward(k: f64, f: f64, sigma: f64, tau: f64, r_f: f64, phi: f64) -> f64 {
  let d2 = (f / k).ln() / (sigma * tau.sqrt()) - 0.5 * sigma * tau.sqrt();
  let nd2 = norm_cdf(phi * d2);
  phi * (-r_f * tau).exp() * (k / f) * nd2
}

/// Model price computed by plugging Hagan approx vol (β = 1) into BS(GK)
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

/// Model price computed by plugging the general-β Hagan vol into BS(GK)
pub fn model_price_hagan_general(
  s: f64,
  k: f64,
  r_d: f64,
  r_f: f64,
  tau: f64,
  alpha: f64,
  beta: f64,
  nu: f64,
  rho: f64,
) -> (f64, f64) {
  let fwd = forward_fx(s, tau, r_d, r_f);
  let sigma = hagan_implied_vol(k, fwd, tau, alpha, beta, nu, rho);
  bs_price_fx(s, k, r_d, r_f, tau, sigma)
}
