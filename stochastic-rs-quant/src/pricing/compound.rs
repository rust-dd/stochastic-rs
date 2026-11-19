//! # Compound
//!
//! $$
//! C^c = Se^{-qT_2}M(z_1,y_1;\sqrt{T_1/T_2})
//!     - K_2 e^{-rT_2}M(z_2,y_2;\sqrt{T_1/T_2})
//!     - K_1 e^{-rT_1}N(z_2)
//! $$
//!
//! Source:
//! - Geske, R. (1979), "The Valuation of Compound Options", J. Financial Economics 7
//! - Rubinstein, M. (1991), "Double Trouble", *RISK Magazine* 5(1)
//! - Haug, E. G. (2007), "The Complete Guide to Option Pricing Formulas", 2nd ed., Ch. 4.3
//!
use owens_t::biv_norm;
use stochastic_rs_distributions::special::norm_cdf;

use crate::OptionType;
use crate::pricing::exotic::CompoundOption;
use crate::pricing::exotic::assert_sigma;

/// Compound option type. The outer option has strike $K_1$, maturity $T_1$;
/// the inner option has strike $K_2$, maturity $T_2 > T_1$.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompoundType {
  /// An outer call on an inner call.
  CallOnCall,
  /// An outer call on an inner put.
  CallOnPut,
  /// An outer put on an inner call.
  PutOnCall,
  /// An outer put on an inner put.
  PutOnPut,
}

impl CompoundType {
  /// Inner option type (call or put).
  pub fn inner(&self) -> OptionType {
    match self {
      Self::CallOnCall | Self::PutOnCall => OptionType::Call,
      Self::CallOnPut | Self::PutOnPut => OptionType::Put,
    }
  }

  /// Outer option type (call or put).
  pub fn outer(&self) -> OptionType {
    match self {
      Self::CallOnCall | Self::CallOnPut => OptionType::Call,
      Self::PutOnCall | Self::PutOnPut => OptionType::Put,
    }
  }
}

/// Geske (1979) compound option: an option on an option, both on the spot.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CompoundPricer {
  /// Annualised Black–Scholes volatility.
  pub sigma: f64,
}

impl CompoundPricer {
  /// Panics unless `sigma` is finite and positive.
  pub fn new(sigma: f64) -> Self {
    assert_sigma(sigma);
    Self { sigma }
  }

  /// Geske's price of `option` at the query `(s, r, q, tau)`, `tau` being the inner maturity.
  pub fn price(&self, option: &CompoundOption, s: f64, r: f64, q: f64, tau: f64) -> f64 {
    let CompoundOption {
      outer_strike: k1,
      outer_tau: t1,
      inner_strike: k2,
      compound_type,
    } = *option;
    let t2 = tau;
    let v = self.sigma;
    let v2 = v * v;
    let b = r - q;
    let s_star = self.critical_inner_price(option, s, r, q, tau);
    let cdf2 = |x: f64, y: f64, rho: f64| -> f64 { biv_norm(-x, -y, rho) };

    let z1 = ((s / s_star).ln() + (b + 0.5 * v2) * t1) / (v * t1.sqrt());
    let z2 = z1 - v * t1.sqrt();
    let y1 = ((s / k2).ln() + (b + 0.5 * v2) * t2) / (v * t2.sqrt());
    let y2 = y1 - v * t2.sqrt();
    let rho = (t1 / t2).sqrt();

    let coc_t2 = ((b - r) * t2).exp();
    let disc_t1 = (-r * t1).exp();
    let disc_t2 = (-r * t2).exp();

    match compound_type {
      CompoundType::CallOnCall => {
        s * coc_t2 * cdf2(z1, y1, rho)
          - k2 * disc_t2 * cdf2(z2, y2, rho)
          - k1 * disc_t1 * norm_cdf(z2)
      }
      CompoundType::PutOnCall => {
        k2 * disc_t2 * cdf2(-z2, y2, -rho) - s * coc_t2 * cdf2(-z1, y1, -rho)
          + k1 * disc_t1 * norm_cdf(-z2)
      }
      CompoundType::CallOnPut => {
        k2 * disc_t2 * cdf2(-z2, -y2, rho)
          - s * coc_t2 * cdf2(-z1, -y1, rho)
          - k1 * disc_t1 * norm_cdf(-z2)
      }
      CompoundType::PutOnPut => {
        s * coc_t2 * cdf2(z1, -y1, -rho) - k2 * disc_t2 * cdf2(z2, -y2, -rho)
          + k1 * disc_t1 * norm_cdf(z2)
      }
    }
  }

  /// Bisection for the spot at the outer maturity where the inner option is worth the outer
  /// strike.
  fn critical_inner_price(&self, option: &CompoundOption, s: f64, r: f64, q: f64, tau: f64) -> f64 {
    let mut lo = 1e-6;
    let mut hi = (option.inner_strike + option.outer_strike).max(s) * 100.0;
    for _ in 0..200 {
      let mid = 0.5 * (lo + hi);
      let inner = self.inner_price(option, mid, r, q, tau);
      let diff = inner - option.outer_strike;
      if diff.abs() < 1e-10 {
        return mid;
      }
      match option.compound_type.inner() {
        OptionType::Call => {
          if diff > 0.0 {
            hi = mid;
          } else {
            lo = mid;
          }
        }
        OptionType::Put => {
          if diff > 0.0 {
            lo = mid;
          } else {
            hi = mid;
          }
        }
      }
    }
    0.5 * (lo + hi)
  }

  /// The inner option at the outer maturity, at a trial spot `s_t1`.
  fn inner_price(&self, option: &CompoundOption, s_t1: f64, r: f64, q: f64, tau: f64) -> f64 {
    let s = s_t1;
    let k2 = option.inner_strike;
    let v = self.sigma;
    let v2 = v * v;
    let b = r - q;
    let tau = tau - option.outer_tau;
    let d1 = ((s / k2).ln() + (b + 0.5 * v2) * tau) / (v * tau.sqrt());
    let d2 = d1 - v * tau.sqrt();
    match option.compound_type.inner() {
      OptionType::Call => {
        s * ((b - r) * tau).exp() * norm_cdf(d1) - k2 * (-r * tau).exp() * norm_cdf(d2)
      }
      OptionType::Put => {
        k2 * (-r * tau).exp() * norm_cdf(-d2) - s * ((b - r) * tau).exp() * norm_cdf(-d1)
      }
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  fn compound(outer_strike: f64, outer_tau: f64, compound_type: CompoundType) -> CompoundOption {
    CompoundOption {
      outer_strike,
      outer_tau,
      inner_strike: 100.0,
      compound_type,
    }
  }

  /// Geske (1979) classic example: S=50, K1=10, K2=50, T1=0.5, T2=1.0,
  /// r=0.05, b=0.05, sigma=0.40 — Call on Call should be positive and bounded
  /// above by the inner call $C(S, K_2, T_2)$.
  #[test]
  fn call_on_call_basic() {
    use crate::pricing::bsm::BsmCoc;
    use crate::pricing::bsm::BsmPricer;
    use crate::traits::ModelPricer;
    let option = CompoundOption {
      outer_strike: 10.0,
      outer_tau: 0.5,
      inner_strike: 50.0,
      compound_type: CompoundType::CallOnCall,
    };
    let coc_price = CompoundPricer::new(0.40).price(&option, 50.0, 0.05, 0.0, 1.0);
    let inner = BsmPricer::new(0.40, BsmCoc::Bsm1973).price_call(50.0, 50.0, 0.05, 0.0, 1.0);
    assert!(coc_price > 0.0, "CoC={coc_price}");
    assert!(
      coc_price < inner,
      "CoC={coc_price} should be < inner call={inner}"
    );
  }

  /// A call on a call is worth less than the call it buys.
  #[test]
  fn coc_bounded_below_by_call_minus_strike() {
    use crate::pricing::bsm::BsmCoc;
    use crate::pricing::bsm::BsmPricer;
    use crate::traits::ModelPricer;

    let s = 100.0;
    let r = 0.05;
    let sigma = 0.25;

    let option = compound(5.0, 0.25, CompoundType::CallOnCall);
    let coc = CompoundPricer::new(sigma).price(&option, s, r, 0.0, 1.0);

    let outer = BsmPricer::new(sigma, BsmCoc::Bsm1973).price_call(s, 100.0, r, 0.0, 1.0);

    assert!(coc < outer);
    assert!(coc > 0.0);
  }

  /// Put-on-call with deep OTM outer strike approaches zero.
  #[test]
  fn put_on_call_otm() {
    let option = compound(0.001, 0.25, CompoundType::PutOnCall);
    let price = CompoundPricer::new(0.25).price(&option, 100.0, 0.05, 0.0, 1.0);
    assert!(price < 0.05, "PoC OTM={price}");
  }

  /// Compound put-call parity (Geske):
  /// CoC - PoC = C(S, K2, T2) - K1 * exp(-r * T1)
  #[test]
  fn compound_put_call_parity() {
    use crate::pricing::bsm::BsmCoc;
    use crate::pricing::bsm::BsmPricer;
    use crate::traits::ModelPricer;

    let s = 100.0;
    let k1 = 5.0;
    let k2 = 100.0;
    let t1 = 0.25;
    let t2 = 1.0;
    let r = 0.05;
    let sigma = 0.25;

    let pricer = CompoundPricer::new(sigma);
    let coc = pricer.price(&compound(k1, t1, CompoundType::CallOnCall), s, r, 0.0, t2);
    let poc = pricer.price(&compound(k1, t1, CompoundType::PutOnCall), s, r, 0.0, t2);

    let inner_call = BsmPricer::new(sigma, BsmCoc::Bsm1973).price_call(s, k2, r, 0.0, t2);

    let lhs = coc - poc;
    let rhs = inner_call - k1 * (-r * t1).exp();
    assert!(
      (lhs - rhs).abs() < 0.05,
      "compound parity violated: lhs={lhs}, rhs={rhs}"
    );
  }
}
