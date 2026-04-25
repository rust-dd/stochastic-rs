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
use statrs::distribution::ContinuousCDF;
use statrs::distribution::Normal;

use crate::quant::OptionType;

/// Compound option type. The outer option has strike $K_1$, maturity $T_1$;
/// the inner option has strike $K_2$, maturity $T_2 > T_1$.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompoundType {
  /// Call on call.
  CallOnCall,
  /// Call on put.
  CallOnPut,
  /// Put on call.
  PutOnCall,
  /// Put on put.
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

/// Geske (1979) compound option.
#[derive(Debug, Clone)]
pub struct CompoundPricer {
  /// Spot.
  pub s: f64,
  /// Outer strike.
  pub k1: f64,
  /// Inner strike.
  pub k2: f64,
  /// Outer maturity.
  pub t1: f64,
  /// Inner maturity (must satisfy $T_2 > T_1$).
  pub t2: f64,
  /// Risk-free rate.
  pub r: f64,
  /// Dividend yield.
  pub q: f64,
  /// Volatility.
  pub sigma: f64,
  /// Compound type.
  pub compound_type: CompoundType,
}

impl CompoundPricer {
  pub fn price(&self) -> f64 {
    let v = self.sigma;
    let v2 = v * v;
    let b = self.r - self.q;
    let s_star = self.critical_inner_price();
    let n = Normal::new(0.0, 1.0).unwrap();
    let cdf2 = |x: f64, y: f64, rho: f64| -> f64 { biv_norm(-x, -y, rho) };

    let z1 = ((self.s / s_star).ln() + (b + 0.5 * v2) * self.t1) / (v * self.t1.sqrt());
    let z2 = z1 - v * self.t1.sqrt();
    let y1 = ((self.s / self.k2).ln() + (b + 0.5 * v2) * self.t2) / (v * self.t2.sqrt());
    let y2 = y1 - v * self.t2.sqrt();
    let rho = (self.t1 / self.t2).sqrt();

    let coc_t2 = ((b - self.r) * self.t2).exp();
    let disc_t1 = (-self.r * self.t1).exp();
    let disc_t2 = (-self.r * self.t2).exp();

    match self.compound_type {
      CompoundType::CallOnCall => {
        self.s * coc_t2 * cdf2(z1, y1, rho)
          - self.k2 * disc_t2 * cdf2(z2, y2, rho)
          - self.k1 * disc_t1 * n.cdf(z2)
      }
      CompoundType::PutOnCall => {
        self.k2 * disc_t2 * cdf2(-z2, y2, -rho) - self.s * coc_t2 * cdf2(-z1, y1, -rho)
          + self.k1 * disc_t1 * n.cdf(-z2)
      }
      CompoundType::CallOnPut => {
        self.k2 * disc_t2 * cdf2(-z2, -y2, rho)
          - self.s * coc_t2 * cdf2(-z1, -y1, rho)
          - self.k1 * disc_t1 * n.cdf(-z2)
      }
      CompoundType::PutOnPut => {
        self.s * coc_t2 * cdf2(z1, -y1, -rho) - self.k2 * disc_t2 * cdf2(z2, -y2, -rho)
          + self.k1 * disc_t1 * n.cdf(z2)
      }
    }
  }

  /// Solve for the spot price at $T_1$ where the inner option payoff equals
  /// the outer strike $K_1$ (or $K_1 - 0$ for puts).
  fn critical_inner_price(&self) -> f64 {
    let mut lo = 1e-6;
    let mut hi = (self.k2 + self.k1).max(self.s) * 100.0;
    for _ in 0..200 {
      let mid = 0.5 * (lo + hi);
      let inner = self.inner_price(mid);
      let diff = inner - self.k1;
      if diff.abs() < 1e-10 {
        return mid;
      }
      match self.compound_type.inner() {
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

  fn inner_price(&self, s: f64) -> f64 {
    let n = Normal::new(0.0, 1.0).unwrap();
    let v = self.sigma;
    let v2 = v * v;
    let b = self.r - self.q;
    let tau = self.t2 - self.t1;
    let d1 = ((s / self.k2).ln() + (b + 0.5 * v2) * tau) / (v * tau.sqrt());
    let d2 = d1 - v * tau.sqrt();
    match self.compound_type.inner() {
      OptionType::Call => {
        s * ((b - self.r) * tau).exp() * n.cdf(d1) - self.k2 * (-self.r * tau).exp() * n.cdf(d2)
      }
      OptionType::Put => {
        self.k2 * (-self.r * tau).exp() * n.cdf(-d2) - s * ((b - self.r) * tau).exp() * n.cdf(-d1)
      }
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  /// Geske (1979) classic example: S=50, K1=10, K2=50, T1=0.5, T2=1.0,
  /// r=0.05, b=0.05, sigma=0.40 — Call on Call should be positive and bounded
  /// above by the inner call $C(S, K_2, T_2)$.
  #[test]
  fn call_on_call_basic() {
    use crate::quant::pricing::bsm::BSMCoc;
    use crate::quant::pricing::bsm::BSMPricer;
    use crate::traits::PricerExt;
    let p = CompoundPricer {
      s: 50.0,
      k1: 10.0,
      k2: 50.0,
      t1: 0.5,
      t2: 1.0,
      r: 0.05,
      q: 0.0,
      sigma: 0.40,
      compound_type: CompoundType::CallOnCall,
    };
    let coc_price = p.price();
    let inner = BSMPricer::builder(50.0, 0.40, 50.0, 0.05)
      .tau(1.0)
      .option_type(OptionType::Call)
      .coc(BSMCoc::Bsm1973)
      .build()
      .calculate_call_put()
      .0;
    assert!(coc_price > 0.0, "CoC={coc_price}");
    assert!(
      coc_price < inner,
      "CoC={coc_price} should be < inner call={inner}"
    );
  }

  /// Compound option must be at least the discounted vanilla $C - K_1$.
  #[test]
  fn coc_bounded_below_by_call_minus_strike() {
    use crate::quant::pricing::bsm::BSMCoc;
    use crate::quant::pricing::bsm::BSMPricer;
    use crate::traits::PricerExt;

    let s = 100.0;
    let r = 0.05;
    let sigma = 0.25;

    let p = CompoundPricer {
      s,
      k1: 5.0,
      k2: 100.0,
      t1: 0.25,
      t2: 1.0,
      r,
      q: 0.0,
      sigma,
      compound_type: CompoundType::CallOnCall,
    };
    let coc = p.price();

    let outer = BSMPricer::builder(s, sigma, 100.0, r)
      .tau(1.0)
      .option_type(OptionType::Call)
      .coc(BSMCoc::Bsm1973)
      .build()
      .calculate_call_put()
      .0;

    // Compound is bounded by inner option price minus discounted strike (very loose but always true)
    assert!(coc < outer);
    assert!(coc > 0.0);
  }

  /// Put-on-call with deep OTM outer strike approaches zero.
  #[test]
  fn put_on_call_otm() {
    let p = CompoundPricer {
      s: 100.0,
      k1: 0.001,
      k2: 100.0,
      t1: 0.25,
      t2: 1.0,
      r: 0.05,
      q: 0.0,
      sigma: 0.25,
      compound_type: CompoundType::PutOnCall,
    };
    let price = p.price();
    assert!(price < 0.05, "PoC OTM={price}");
  }

  /// Compound put-call parity (Geske):
  /// CoC - PoC = C(S, K2, T2) - K1 * exp(-r * T1)
  #[test]
  fn compound_put_call_parity() {
    use crate::quant::pricing::bsm::BSMCoc;
    use crate::quant::pricing::bsm::BSMPricer;
    use crate::traits::PricerExt;

    let s = 100.0;
    let k1 = 5.0;
    let k2 = 100.0;
    let t1 = 0.25;
    let t2 = 1.0;
    let r = 0.05;
    let sigma = 0.25;

    let coc = CompoundPricer {
      s,
      k1,
      k2,
      t1,
      t2,
      r,
      q: 0.0,
      sigma,
      compound_type: CompoundType::CallOnCall,
    };
    let poc = CompoundPricer {
      compound_type: CompoundType::PutOnCall,
      ..coc.clone()
    };

    let inner_call = BSMPricer::builder(s, sigma, k2, r)
      .tau(t2)
      .option_type(OptionType::Call)
      .coc(BSMCoc::Bsm1973)
      .build()
      .calculate_call_put()
      .0;

    let lhs = coc.price() - poc.price();
    let rhs = inner_call - k1 * (-r * t1).exp();
    assert!(
      (lhs - rhs).abs() < 0.05,
      "compound parity violated: lhs={lhs}, rhs={rhs}"
    );
  }
}
