//! # Chooser
//!
//! $$
//! W = Se^{-qT}N(d) - Ke^{-rT}N(d - \sigma\sqrt T)
//!     -Se^{-qT}N(-y) + Ke^{-rT}N(-y + \sigma\sqrt{t_1})
//! $$
//!
//! Source:
//! - Rubinstein, M. (1991), "Options for the Undecided", *RISK Magazine* 4(4)
//! - Haug, E. G. (2007), "The Complete Guide to Option Pricing Formulas", 2nd ed., Ch. 4.2
//!
use owens_t::biv_norm;
use statrs::distribution::ContinuousCDF;
use statrs::distribution::Normal;

use crate::quant::OptionType;

/// Simple chooser: at time $t_1 < T$ the holder picks between a call and a
/// put, both with strike $K$ and maturity $T$.
///
/// $$
/// W = Se^{-qT}N(d) - Ke^{-rT}N(d-\sigma\sqrt T)
///   - Se^{-qT}N(-y) + Ke^{-rT}N(-y+\sigma\sqrt{t_1})
/// $$
#[derive(Debug, Clone)]
pub struct SimpleChooserPricer {
  /// Spot.
  pub s: f64,
  /// Strike of both call and put.
  pub k: f64,
  /// Risk-free rate.
  pub r: f64,
  /// Dividend yield.
  pub q: f64,
  /// Volatility.
  pub sigma: f64,
  /// Choice time (must be strictly less than `t`).
  pub t1: f64,
  /// Common maturity.
  pub t: f64,
}

impl SimpleChooserPricer {
  pub fn price(&self) -> f64 {
    let n = Normal::new(0.0, 1.0).unwrap();
    let v = self.sigma;
    let v2 = v * v;
    let b = self.r - self.q;
    let sqrt_t = self.t.sqrt();
    let sqrt_t1 = self.t1.sqrt();
    let d = ((self.s / self.k).ln() + (b + 0.5 * v2) * self.t) / (v * sqrt_t);
    let y = ((self.s / self.k).ln() + b * self.t + 0.5 * v2 * self.t1) / (v * sqrt_t1);

    self.s * (-self.q * self.t).exp() * n.cdf(d)
      - self.k * (-self.r * self.t).exp() * n.cdf(d - v * sqrt_t)
      - self.s * (-self.q * self.t).exp() * n.cdf(-y)
      + self.k * (-self.r * self.t).exp() * n.cdf(-y + v * sqrt_t1)
  }
}

/// Complex chooser: at time $t_1$ pick between a call $(K_c, T_c)$ and a put
/// $(K_p, T_p)$ where both maturities exceed $t_1$.
///
/// Requires solving for the critical underlying price $S^\*$ at the choice
/// time where the call value equals the put value, then a bivariate normal
/// integral.
#[derive(Debug, Clone)]
pub struct ComplexChooserPricer {
  /// Spot.
  pub s: f64,
  /// Risk-free rate.
  pub r: f64,
  /// Dividend yield.
  pub q: f64,
  /// Volatility.
  pub sigma: f64,
  /// Choice time.
  pub t1: f64,
  /// Call strike.
  pub k_call: f64,
  /// Call maturity.
  pub t_call: f64,
  /// Put strike.
  pub k_put: f64,
  /// Put maturity.
  pub t_put: f64,
}

impl ComplexChooserPricer {
  pub fn price(&self) -> f64 {
    let v = self.sigma;
    let v2 = v * v;
    let b = self.r - self.q;
    let s_star = self.critical_price();

    let cdf = |x: f64, y: f64, rho: f64| -> f64 { biv_norm(-x, -y, rho) };

    let d1 = ((self.s / s_star).ln() + (b + 0.5 * v2) * self.t1) / (v * self.t1.sqrt());
    let d2 = d1 - v * self.t1.sqrt();
    let y1 =
      ((self.s / self.k_call).ln() + (b + 0.5 * v2) * self.t_call) / (v * self.t_call.sqrt());
    let y2 = ((self.s / self.k_put).ln() + (b + 0.5 * v2) * self.t_put) / (v * self.t_put.sqrt());
    let rho1 = (self.t1 / self.t_call).sqrt();
    let rho2 = (self.t1 / self.t_put).sqrt();

    self.s * ((b - self.r) * self.t_call).exp() * cdf(d1, y1, rho1)
      - self.k_call * (-self.r * self.t_call).exp() * cdf(d2, y1 - v * self.t_call.sqrt(), rho1)
      - self.s * ((b - self.r) * self.t_put).exp() * cdf(-d1, -y2, rho2)
      + self.k_put * (-self.r * self.t_put).exp() * cdf(-d2, -y2 + v * self.t_put.sqrt(), rho2)
  }

  /// Critical price: solve $C(S^*, t_1) = P(S^*, t_1)$ via bisection.
  fn critical_price(&self) -> f64 {
    let mut lo = 1e-6;
    let mut hi = 1e6_f64.max(self.s * 100.0);
    for _ in 0..200 {
      let mid = 0.5 * (lo + hi);
      let diff = self.put_call_diff(mid);
      if diff.abs() < 1e-10 {
        return mid;
      }
      if diff > 0.0 {
        hi = mid;
      } else {
        lo = mid;
      }
    }
    0.5 * (lo + hi)
  }

  fn put_call_diff(&self, s: f64) -> f64 {
    let n = Normal::new(0.0, 1.0).unwrap();
    let v = self.sigma;
    let v2 = v * v;
    let b = self.r - self.q;
    let tau_c = self.t_call - self.t1;
    let tau_p = self.t_put - self.t1;
    let d1c = ((s / self.k_call).ln() + (b + 0.5 * v2) * tau_c) / (v * tau_c.sqrt());
    let d2c = d1c - v * tau_c.sqrt();
    let d1p = ((s / self.k_put).ln() + (b + 0.5 * v2) * tau_p) / (v * tau_p.sqrt());
    let d2p = d1p - v * tau_p.sqrt();
    let call = s * ((b - self.r) * tau_c).exp() * n.cdf(d1c)
      - self.k_call * (-self.r * tau_c).exp() * n.cdf(d2c);
    let put = self.k_put * (-self.r * tau_p).exp() * n.cdf(-d2p)
      - s * ((b - self.r) * tau_p).exp() * n.cdf(-d1p);
    call - put
  }
}

/// Forward-start option (Rubinstein 1990). Strike fixes at time $t_1$ as
/// $K = \alpha S_{t_1}$.
#[derive(Debug, Clone)]
pub struct ForwardStartPricer {
  /// Spot.
  pub s: f64,
  /// Strike multiplier.
  pub alpha: f64,
  /// Risk-free rate.
  pub r: f64,
  /// Dividend yield.
  pub q: f64,
  /// Volatility (assumed constant from $t_1$ to $T$).
  pub sigma: f64,
  /// Strike-fixing time.
  pub t1: f64,
  /// Maturity.
  pub t: f64,
  /// Option type.
  pub option_type: OptionType,
}

impl ForwardStartPricer {
  pub fn price(&self) -> f64 {
    let n = Normal::new(0.0, 1.0).unwrap();
    let tau = self.t - self.t1;
    let v = self.sigma;
    let v2 = v * v;
    let b = self.r - self.q;
    let d1 = ((-self.alpha.ln()) + (b + 0.5 * v2) * tau) / (v * tau.sqrt());
    let d2 = d1 - v * tau.sqrt();
    let coc1 = ((b - self.r) * self.t1).exp();
    let coc_tau = ((b - self.r) * tau).exp();
    let disc = (-self.r * tau).exp();
    match self.option_type {
      OptionType::Call => self.s * coc1 * (coc_tau * n.cdf(d1) - self.alpha * disc * n.cdf(d2)),
      OptionType::Put => {
        self.s * coc1 * (self.alpha * disc * n.cdf(-d2) - coc_tau * n.cdf(-d1))
      }
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  /// Haug 2007 example, p. 174: simple chooser with S=50, K=50, t1=0.25,
  /// T=0.5, r=0.08, b=0.08, sigma=0.25 → 6.1071
  #[test]
  fn simple_chooser_haug() {
    let p = SimpleChooserPricer {
      s: 50.0,
      k: 50.0,
      r: 0.08,
      q: 0.0,
      sigma: 0.25,
      t1: 0.25,
      t: 0.5,
    };
    let price = p.price();
    assert!((price - 6.1071).abs() < 0.01, "chooser={price}");
  }

  /// Simple chooser with $t_1 \to 0$ should equal a long straddle (call+put)
  /// only at maturity decision; with $t_1 \to T$ the holder waits and gets
  /// $\max(C_T, P_T)$ which equals the straddle. With $t_1=0$ the holder
  /// commits immediately so picks the better one ATM, payoff equals
  /// max(call, put) ≈ at-the-money put or call.
  #[test]
  fn simple_chooser_bounded_by_straddle() {
    use crate::quant::pricing::bsm::BSMCoc;
    use crate::quant::pricing::bsm::BSMPricer;
    use crate::traits::PricerExt;

    let s = 100.0;
    let k = 100.0;
    let r = 0.05;
    let sigma = 0.25;
    let t = 1.0;

    let chooser = SimpleChooserPricer {
      s,
      k,
      r,
      q: 0.0,
      sigma,
      t1: 0.5,
      t,
    };
    let chooser_price = chooser.price();

    let call = BSMPricer::builder(s, sigma, k, r)
      .tau(t)
      .option_type(OptionType::Call)
      .coc(BSMCoc::Bsm1973)
      .build()
      .calculate_call_put();

    // Chooser must be bounded between max(call,put) and call+put.
    assert!(chooser_price > call.0.max(call.1));
    assert!(chooser_price < call.0 + call.1);
  }

  /// Complex chooser with `K_call = K_put` and `T_call = T_put` should equal
  /// the simple chooser.
  #[test]
  fn complex_chooser_matches_simple() {
    let s = 100.0;
    let r = 0.05;
    let q = 0.02;
    let sigma = 0.3;
    let t1 = 0.4;
    let t = 1.0;
    let k = 95.0;

    let simple = SimpleChooserPricer {
      s,
      k,
      r,
      q,
      sigma,
      t1,
      t,
    };
    let complex = ComplexChooserPricer {
      s,
      r,
      q,
      sigma,
      t1,
      k_call: k,
      t_call: t,
      k_put: k,
      t_put: t,
    };
    let diff = (simple.price() - complex.price()).abs();
    assert!(diff < 0.01, "diff={diff}");
  }

  /// Forward-start at-the-money ($\alpha=1$) is positive and increases with
  /// volatility.
  #[test]
  fn forward_start_atm_positive() {
    let base = ForwardStartPricer {
      s: 100.0,
      alpha: 1.0,
      r: 0.05,
      q: 0.0,
      sigma: 0.20,
      t1: 0.25,
      t: 1.0,
      option_type: OptionType::Call,
    };
    let high = ForwardStartPricer { sigma: 0.40, ..base.clone() };
    assert!(base.price() > 0.0);
    assert!(high.price() > base.price());
  }
}
