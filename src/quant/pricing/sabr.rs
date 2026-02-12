use implied_vol::implied_black_volatility;
use statrs::distribution::ContinuousCDF;
use statrs::distribution::Normal;

use crate::quant::pricing::bsm::BSMCoc;
use crate::quant::pricing::bsm::BSMPricer;
use crate::traits::PricerExt;
use crate::traits::TimeExt;
use crate::quant::OptionType;

/// Forward FX F = S * exp((r_d - r_f) T)
pub fn forward_fx(s: f64, tau: f64, r_d: f64, r_f: f64) -> f64 {
  s * ((r_d - r_f) * tau).exp()
}

/// Hagan et al. (2002) implied vol approximation for beta = 1 (lognormal SABR)
pub fn hagan_implied_vol_beta1(k: f64, f: f64, tau: f64, alpha: f64, nu: f64, rho: f64) -> f64 {
  if (k - f).abs() < 1e-12 {
    return alpha
      * (1.0 + tau * (rho * nu * alpha / 4.0 + (2.0 - 3.0 * rho * rho) * nu * nu / 24.0));
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
    BSMCoc::GarmanKohlhagen1983,
  );
  pricer.calculate_call_put()
}

/// Delta on forward with premium included
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

/// Pricer that uses SABR(Hagan beta=1) to produce an implied vol, then prices via Black-GK.
#[derive(Clone, Copy, Debug)]
pub struct SabrPricer {
  pub s: f64,
  pub k: f64,
  /// Risk-free rate (domestic)
  pub r: f64,
  /// Dividend/foreign rate (q)
  pub q: Option<f64>,
  pub alpha: f64,
  pub nu: f64,
  pub rho: f64,
  pub tau: Option<f64>,
  pub eval: Option<chrono::NaiveDate>,
  pub expiration: Option<chrono::NaiveDate>,
}

impl SabrPricer {
  pub fn new(
    s: f64, k: f64, r: f64, q: Option<f64>, alpha: f64, nu: f64, rho: f64,
    tau: Option<f64>, eval: Option<chrono::NaiveDate>, expiration: Option<chrono::NaiveDate>,
  ) -> Self {
    Self { s, k, r, q, alpha, nu, rho, tau, eval, expiration }
  }
}

impl TimeExt for SabrPricer {
  fn tau(&self) -> Option<f64> {
    self.tau
  }
  fn eval(&self) -> Option<chrono::NaiveDate> {
    self.eval
  }
  fn expiration(&self) -> Option<chrono::NaiveDate> {
    self.expiration
  }
}

impl SabrPricer {
  pub fn forward(&self) -> f64 {
    forward_fx(self.s, self.tau().unwrap(), self.r, self.q.unwrap_or(0.0))
  }
  pub fn sigma(&self) -> f64 {
    hagan_implied_vol_beta1(
      self.k,
      self.forward(),
      self.tau().unwrap(),
      self.alpha,
      self.nu,
      self.rho,
    )
  }
  /// Forward-based (premium-included) delta with r_f := q
  pub fn sabr_fx_forward_delta(&self, phi: f64) -> f64 {
    fx_delta_from_forward(
      self.k,
      self.forward(),
      self.sigma(),
      self.tau().unwrap(),
      self.q.unwrap_or(0.0),
      phi,
    )
  }
}

impl PricerExt for SabrPricer {
  fn calculate_call_put(&self) -> (f64, f64) {
    let sigma = self.sigma();
    let pricer = BSMPricer::new(
      self.s,
      sigma,
      self.k,
      self.r,
      None,
      None,
      self.q,
      Some(self.tau().unwrap()),
      self.eval,
      self.expiration,
      OptionType::Call,
      BSMCoc::Merton1973,
    );
    pricer.calculate_call_put()
  }

  fn calculate_price(&self) -> f64 {
    self.calculate_call_put().0
  }

  fn implied_volatility(&self, c_price: f64, option_type: OptionType) -> f64 {
    implied_black_volatility(
      c_price,
      self.s,
      self.k,
      self.calculate_tau_in_days(),
      option_type == OptionType::Call,
    )
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn sabr_pricer_basic() {
    let s = 3.724;
    let k = 3.8;
    let r = 0.065;
    let q = Some(0.022);
    let tau = 30.0 / 365.0;
    let pr = SabrPricer::new(s, k, r, q, 0.11, 0.6, 0.5, Some(tau), None, None);
    let (c, p) = pr.calculate_call_put();
    println!("Call: {}, Put: {}", c, p);
    assert!(c >= 0.0 && p >= 0.0);
    let d = pr.sabr_fx_forward_delta(1.0);
    assert!(d.is_finite());
  }
}
