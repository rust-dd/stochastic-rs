//! # SABR
//!
//! Implied-volatility approximation and pricing under the SABR stochastic
//! volatility model.
//!
//! **Reference:** P. S. Hagan, D. Kumar, A. S. Lesniewski, D. E. Woodward,
//! *Managing Smile Risk*, Wilmott Magazine, pp. 84–108, 2002.
//! (Eq. A.69a–A.69c for the general-β lognormal vol expansion.)
//!
//! $$
//! dF_t=\alpha_t F_t^\beta dW_t^1,\quad d\alpha_t=\nu\alpha_t dW_t^2,\ d\langle W^1,W^2\rangle_t=\rho dt
//! $$
//!
use implied_vol::DefaultSpecialFn;
use implied_vol::ImpliedBlackVolatility;
use statrs::distribution::ContinuousCDF;
use statrs::distribution::Normal;

use crate::quant::OptionType;
use crate::quant::pricing::bsm::BSMCoc;
use crate::quant::pricing::bsm::BSMPricer;
use crate::traits::PricerExt;
use crate::traits::TimeExt;

/// Forward FX F = S * exp((r_d - r_f) T)
pub fn forward_fx(s: f64, tau: f64, r_d: f64, r_f: f64) -> f64 {
  s * ((r_d - r_f) * tau).exp()
}

/// Hagan et al. (2002) SABR implied-vol approximation for general β.
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

/// Compute SABR α from an ATM lognormal vol by solving the Hagan (2002) ATM
/// condition (Eq. A.69b) as a cubic polynomial in α (quadratic when β = 1).
///
/// $$
/// \frac{(1-\beta)^2 f_*^3}{24}\,\tau\;\alpha^3
/// +\frac{\rho\beta\nu\,f_*^2}{4}\,\tau\;\alpha^2
/// +\bigl(1+\tfrac{(2-3\rho^2)\nu^2}{24}\,\tau\bigr)\,f_*\;\alpha
/// -\sigma_{ATM}=0, \quad f_*=F^{\beta-1}
/// $$
pub fn alpha_from_atm_vol(
  v_atm: f64,
  f: f64,
  tau: f64,
  beta: f64,
  rho: f64,
  nu: f64,
) -> f64 {
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
  let n = Normal::new(0.0, 1.0).unwrap();
  let nd2 = n.cdf(phi * d2);
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

/// Pricer that uses SABR (Hagan 2002, general β) to produce an implied vol,
/// then prices via Black-GK.
#[derive(Clone, Copy, Debug)]
pub struct SabrPricer {
  /// Underlying spot/forward level.
  pub s: f64,
  /// Strike level.
  pub k: f64,
  /// Risk-free rate (domestic)
  pub r: f64,
  /// Dividend/foreign rate (q)
  pub q: Option<f64>,
  /// Model shape/loading parameter.
  pub alpha: f64,
  /// CEV exponent (0 = normal, 1 = lognormal).
  pub beta: f64,
  /// Volatility-of-volatility parameter.
  pub nu: f64,
  /// Correlation parameter.
  pub rho: f64,
  /// Time-to-maturity in years.
  pub tau: Option<f64>,
  /// Valuation date.
  pub eval: Option<chrono::NaiveDate>,
  /// Expiration date.
  pub expiration: Option<chrono::NaiveDate>,
}

impl SabrPricer {
  pub fn new(
    s: f64,
    k: f64,
    r: f64,
    q: Option<f64>,
    alpha: f64,
    beta: f64,
    nu: f64,
    rho: f64,
    tau: Option<f64>,
    eval: Option<chrono::NaiveDate>,
    expiration: Option<chrono::NaiveDate>,
  ) -> Self {
    Self {
      s,
      k,
      r,
      q,
      alpha,
      beta,
      nu,
      rho,
      tau,
      eval,
      expiration,
    }
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
    hagan_implied_vol(
      self.k,
      self.forward(),
      self.tau().unwrap(),
      self.alpha,
      self.beta,
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
    ImpliedBlackVolatility::builder()
      .option_price(c_price)
      .forward(self.s)
      .strike(self.k)
      .expiry(self.calculate_tau_in_days())
      .is_call(option_type == OptionType::Call)
      .build()
      .and_then(|iv| iv.calculate::<DefaultSpecialFn>())
      .unwrap_or(f64::NAN)
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
    let pr = SabrPricer::new(s, k, r, q, 0.11, 1.0, 0.6, 0.5, Some(tau), None, None);
    let (c, p) = pr.calculate_call_put();
    println!("Call: {}, Put: {}", c, p);
    assert!(c >= 0.0 && p >= 0.0);
    let d = pr.sabr_fx_forward_delta(1.0);
    assert!(d.is_finite());
  }

  /// Hagan (2002, Eq. A.69a) general-β implied vol — reference values.
  #[test]
  fn hagan_implied_vol_reference() {
    // (k, f, t, alpha, beta, rho, nu, expected_vol)
    let cases: &[(f64, f64, f64, f64, f64, f64, f64, f64)] = &[
      // β=1 ATM
      (100.0, 100.0, 1.0, 0.2, 1.0, -0.3, 0.5, 2.021041666666667e-01),
      // β=1 OTM call
      (110.0, 100.0, 1.0, 0.2, 1.0, -0.3, 0.5, 1.966695601513802e-01),
      // β=1 OTM put
      (90.0, 100.0, 1.0, 0.2, 1.0, -0.3, 0.5, 2.118933616456034e-01),
      // β=0.5 ATM
      (100.0, 100.0, 0.5, 0.15, 0.5, -0.2, 0.8, 1.537376757812500e-02),
      // β=0.5 OTM call
      (110.0, 100.0, 0.5, 0.15, 0.5, -0.2, 0.8, 3.080869461133284e-02),
      // β=0.5 OTM put
      (90.0, 100.0, 0.5, 0.15, 0.5, -0.2, 0.8, 3.832319931343581e-02),
      // FX-like β=1 OTM call
      (3.8, 3.724, 30.0 / 365.0, 0.14, 1.0, 0.33, 1.6, 1.486360336704149e-01),
      // FX-like β=1 OTM put
      (3.6, 3.724, 30.0 / 365.0, 0.14, 1.0, 0.33, 1.6, 1.365590050371177e-01),
      // β=0.7
      (105.0, 100.0, 0.25, 0.3, 0.7, 0.1, 0.4, 7.683910485737674e-02),
    ];

    for (i, &(k, f, t, alpha, beta, rho, nu, expected)) in cases.iter().enumerate() {
      let got = hagan_implied_vol(k, f, t, alpha, beta, nu, rho);
      let err = (got - expected).abs();
      assert!(
        err < 1e-12,
        "case {}: got {:.15e}, expected {:.15e}, err={:.2e}",
        i, got, expected, err
      );
    }
  }

  /// α-from-ATM-vol cubic solver (Hagan 2002, Eq. A.69b) — reference values + round-trip.
  #[test]
  fn alpha_from_atm_vol_reference() {
    // (v_atm, f, t, beta, rho, nu, expected_alpha)
    let cases: &[(f64, f64, f64, f64, f64, f64, f64)] = &[
      // β=1 equity-like
      (0.20, 100.0, 1.0, 1.0, -0.3, 0.5, 1.979023350370119e-01),
      // β=0.5
      (0.15, 100.0, 0.5, 0.5, -0.2, 0.8, 1.465254087095464e+00),
      // β=1 FX-like
      (0.1424, 3.724, 30.0 / 365.0, 1.0, 0.33, 1.6, 1.401312256794535e-01),
      // β=0.7 long-dated
      (0.30, 50.0, 2.0, 0.7, 0.1, 0.4, 9.409442654142710e-01),
      // β=0 (normal SABR)
      (0.25, 100.0, 1.0, 0.0, 0.0, 0.3, 2.475118650781591e+01),
    ];

    for (i, &(v_atm, f, t, beta, rho, nu, expected)) in cases.iter().enumerate() {
      let got = alpha_from_atm_vol(v_atm, f, t, beta, rho, nu);
      let err = (got - expected).abs() / expected.abs();
      assert!(
        err < 1e-8,
        "case {}: got {:.15e}, expected {:.15e}, rel_err={:.2e}",
        i, got, expected, err
      );
      // Round-trip: α → ATM vol
      let vol_rt = hagan_implied_vol(f, f, t, got, beta, nu, rho);
      let rt_err = (vol_rt - v_atm).abs();
      assert!(
        rt_err < 1e-10,
        "round-trip case {}: vol={:.15e}, target={:.15e}, err={:.2e}",
        i, vol_rt, v_atm, rt_err
      );
    }
  }
}
