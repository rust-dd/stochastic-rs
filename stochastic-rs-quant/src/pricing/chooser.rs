//! The simple and complex choosers and the forward start under Black–Scholes, in closed form.
//!
//! $$
//! W = Se^{-qT}N(d) - Ke^{-rT}N(d - \sigma\sqrt T)
//!     -Se^{-qT}N(-y) + Ke^{-rT}N(-y + \sigma\sqrt{t_1})
//! $$
//!
//! Reference: Rubinstein, M. (1991), "Options for the Undecided", *Risk* 4(4), 43
//! Reference: Rubinstein, M. (1991), "Pay Now, Choose Later", *RISK Magazine*
//! Reference: Vector Risk (2017), "Forward Start Option", v8.0.7905, p. 2: Rubinstein's 4(2), 13
//! Reference: Haug, E. G. (2007), "The Complete Guide to Option Pricing Formulas", ISBN 0071389970
//! Reference: Vector Risk (2017), "Simple Chooser Option", v8.0.7905, p. 1: Haug's §4.12.1, p. 128
//! Reference: Vector Risk (2017), "Complex Chooser Option", v8.0.7905, p. 2: Haug's §4.12.2, p. 129
//!
use owens_t::biv_norm;
use stochastic_rs_distributions::RealExt;
use stochastic_rs_distributions::special::norm_cdf;

use crate::OptionType;
use crate::pricing::bsm::BsmCoc;
use crate::pricing::bsm::BsmPricer;
use crate::pricing::digital::vanilla_payoff;
use crate::pricing::exotic::ChooserOption;
use crate::pricing::exotic::ComplexChooserOption;
use crate::pricing::exotic::ForwardStartOption;
use crate::pricing::exotic::assert_sigma;
use crate::traits::ModelPricer;

/// Black–Scholes prices of Rubinstein's (1991) simple and complex choosers and forward start.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChooserPricer {
  /// Annualised Black–Scholes volatility.
  pub sigma: f64,
}

impl ChooserPricer {
  /// Panics unless `sigma` is finite and `>= 0`; at `0` each price is its deterministic limit, NaN
  /// at the money forward, but the complex chooser is NaN unless `choice_time = 0`.
  pub fn new(sigma: f64) -> Self {
    assert_sigma(sigma);
    Self { sigma }
  }

  /// The simple chooser $W$, the better leg at `choice_time = 0` and `|s - strike|` at `tau = 0`;
  /// NaN for a NaN rate, `choice_time` off `[0, tau]` or a spot that is not positive before expiry.
  pub fn price_simple(&self, option: &ChooserOption, s: f64, r: f64, q: f64, tau: f64) -> f64 {
    let ChooserOption {
      strike: k,
      choice_time: t1,
    } = *option;
    if !(0.0 <= t1 && t1 <= tau) || r.is_nan() || q.is_nan() {
      return f64::NAN;
    }
    if tau == 0.0 {
      return (s - k).abs();
    }
    if s.is_nan() || s <= 0.0 {
      return f64::NAN;
    }
    if t1 == 0.0 {
      let legs = BsmPricer::new(self.sigma, BsmCoc::Merton1973).price_call_put(s, k, r, q, tau);
      return legs.call.max_or_nan(legs.put);
    }
    let v = self.sigma;
    let v2 = v * v;
    let b = r - q;
    let sqrt_t = tau.sqrt();
    let sqrt_t1 = t1.sqrt();
    let d = ((s / k).ln() + (b + 0.5 * v2) * tau) / (v * sqrt_t);
    let y = ((s / k).ln() + b * tau + 0.5 * v2 * t1) / (v * sqrt_t1);

    s * (-q * tau).exp() * norm_cdf(d)
      - k * (-r * tau).exp() * norm_cdf(d - v * sqrt_t)
      - s * (-q * tau).exp() * norm_cdf(-y)
      + k * (-r * tau).exp() * norm_cdf(-y + v * sqrt_t1)
  }

  /// The complex chooser through $S^*$, where its legs are equal, and the better leg deciding at 0;
  /// NaN for `choice_time` off `[0, min(call_tau, put_tau)]` or a spot not positive before expiry.
  pub fn price_complex(&self, option: &ComplexChooserOption, s: f64, r: f64, q: f64) -> f64 {
    let ComplexChooserOption {
      call_strike: k_call,
      call_tau: t_call,
      put_strike: k_put,
      put_tau: t_put,
      choice_time: t1,
    } = *option;
    if !(0.0 <= t1 && t1 <= t_call && t1 <= t_put) || t1.is_infinite() {
      return f64::NAN;
    }
    if (s.is_nan() || s <= 0.0) && (t_call > 0.0 || t_put > 0.0) {
      return f64::NAN;
    }
    if t1 == 0.0 {
      let vanilla = BsmPricer::new(self.sigma, BsmCoc::Merton1973);
      let call = vanilla.price_call(s, k_call, r, q, t_call);
      return call.max_or_nan(vanilla.price_put(s, k_put, r, q, t_put));
    }
    let v = self.sigma;
    let v2 = v * v;
    let b = r - q;
    let s_star = self.critical_price(option, s, r, q);

    let cdf = |x: f64, y: f64, rho: f64| -> f64 { biv_norm(-x, -y, rho) };

    let d1 = ((s / s_star).ln() + (b + 0.5 * v2) * t1) / (v * t1.sqrt());
    let d2 = d1 - v * t1.sqrt();
    let y1 = ((s / k_call).ln() + (b + 0.5 * v2) * t_call) / (v * t_call.sqrt());
    let y2 = ((s / k_put).ln() + (b + 0.5 * v2) * t_put) / (v * t_put.sqrt());
    let rho1 = (t1 / t_call).sqrt();
    let rho2 = (t1 / t_put).sqrt();

    s * ((b - r) * t_call).exp() * cdf(d1, y1, rho1)
      - k_call * (-r * t_call).exp() * cdf(d2, y1 - v * t_call.sqrt(), rho1)
      - s * ((b - r) * t_put).exp() * cdf(-d1, -y2, rho2)
      + k_put * (-r * t_put).exp() * cdf(-d2, -y2 + v * t_put.sqrt(), rho2)
  }

  /// Bisection for $C(S^*, t_1) = P(S^*, t_1)$.
  fn critical_price(&self, option: &ComplexChooserOption, s: f64, r: f64, q: f64) -> f64 {
    let mut lo = 1e-6;
    let mut hi = 1e6_f64.max(s * 100.0);
    for _ in 0..200 {
      let mid = 0.5 * (lo + hi);
      let diff = self.put_call_diff(option, mid, r, q);
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

  /// Call less put at the choice time, at a trial spot `s_t1`.
  fn put_call_diff(&self, option: &ComplexChooserOption, s_t1: f64, r: f64, q: f64) -> f64 {
    let s = s_t1;
    let v = self.sigma;
    let v2 = v * v;
    let b = r - q;
    let tau_c = option.call_tau - option.choice_time;
    let tau_p = option.put_tau - option.choice_time;
    let d1c = ((s / option.call_strike).ln() + (b + 0.5 * v2) * tau_c) / (v * tau_c.sqrt());
    let d2c = d1c - v * tau_c.sqrt();
    let d1p = ((s / option.put_strike).ln() + (b + 0.5 * v2) * tau_p) / (v * tau_p.sqrt());
    let d2p = d1p - v * tau_p.sqrt();
    let call = s * ((b - r) * tau_c).exp() * norm_cdf(d1c)
      - option.call_strike * (-r * tau_c).exp() * norm_cdf(d2c);
    let put = option.put_strike * (-r * tau_p).exp() * norm_cdf(-d2p)
      - s * ((b - r) * tau_p).exp() * norm_cdf(-d1p);
    call - put
  }

  /// The forward start struck at `moneyness` times the spot at `start_time`, its payoff at expiry;
  /// NaN for a NaN rate, `start_time` off `[0, tau]` or a spot that is not positive before expiry.
  pub fn price_forward_start(
    &self,
    option: &ForwardStartOption,
    s: f64,
    r: f64,
    q: f64,
    tau: f64,
  ) -> f64 {
    let ForwardStartOption {
      moneyness: alpha,
      start_time: t1,
      option_type,
    } = *option;
    if !(0.0 <= t1 && t1 <= tau) || r.is_nan() || q.is_nan() {
      return f64::NAN;
    }
    if tau == 0.0 {
      return vanilla_payoff(option_type, s, alpha * s);
    }
    if s.is_nan() || s <= 0.0 {
      return f64::NAN;
    }
    let remaining_tau = tau - t1;
    let v = self.sigma;
    let v2 = v * v;
    let b = r - q;
    if remaining_tau == 0.0 {
      let carried = s * ((b - r) * t1).exp();
      return vanilla_payoff(option_type, carried, alpha * carried);
    }
    let d1 = ((-alpha.ln()) + (b + 0.5 * v2) * remaining_tau) / (v * remaining_tau.sqrt());
    let d2 = d1 - v * remaining_tau.sqrt();
    let coc1 = ((b - r) * t1).exp();
    let coc_tau = ((b - r) * remaining_tau).exp();
    let disc = (-r * remaining_tau).exp();
    match option_type {
      OptionType::Call => s * coc1 * (coc_tau * norm_cdf(d1) - alpha * disc * norm_cdf(d2)),
      OptionType::Put => s * coc1 * (alpha * disc * norm_cdf(-d2) - coc_tau * norm_cdf(-d1)),
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
    let option = ChooserOption {
      strike: 50.0,
      choice_time: 0.25,
    };
    let price = ChooserPricer::new(0.25).price_simple(&option, 50.0, 0.08, 0.0, 0.5);
    assert!((price - 6.1071).abs() < 0.01, "chooser={price}");
  }

  /// The chooser lies between the better of the call and the put and the straddle.
  #[test]
  fn simple_chooser_bounded_by_straddle() {
    use crate::pricing::bsm::BsmCoc;
    use crate::pricing::bsm::BsmPricer;
    use crate::traits::ModelPricer;

    let s = 100.0;
    let k = 100.0;
    let r = 0.05;
    let sigma = 0.25;
    let tau = 1.0;

    let option = ChooserOption {
      strike: k,
      choice_time: 0.5,
    };
    let chooser_price = ChooserPricer::new(sigma).price_simple(&option, s, r, 0.0, tau);

    let bsm = BsmPricer::new(sigma, BsmCoc::Bsm1973);
    let (call, put) = (
      bsm.price_call(s, k, r, 0.0, tau),
      bsm.price_put(s, k, r, 0.0, tau),
    );

    assert!(chooser_price > call.max(put));
    assert!(chooser_price < call + put);
  }

  /// Complex chooser with `K_call = K_put` and `T_call = T_put` should equal
  /// the simple chooser.
  #[test]
  fn complex_chooser_matches_simple() {
    let s = 100.0;
    let r = 0.05;
    let q = 0.02;
    let t1 = 0.4;
    let tau = 1.0;
    let k = 95.0;

    let pricer = ChooserPricer::new(0.3);
    let simple = ChooserOption {
      strike: k,
      choice_time: t1,
    };
    let complex = ComplexChooserOption {
      call_strike: k,
      call_tau: tau,
      put_strike: k,
      put_tau: tau,
      choice_time: t1,
    };
    let diff =
      (pricer.price_simple(&simple, s, r, q, tau) - pricer.price_complex(&complex, s, r, q)).abs();
    assert!(diff < 0.01, "diff={diff}");
  }

  /// Forward-start at-the-money ($\alpha=1$) is positive and increases with
  /// volatility.
  #[test]
  fn forward_start_atm_positive() {
    let option = ForwardStartOption {
      moneyness: 1.0,
      start_time: 0.25,
      option_type: OptionType::Call,
    };
    let base = ChooserPricer::new(0.20).price_forward_start(&option, 100.0, 0.05, 0.0, 1.0);
    let high = ChooserPricer::new(0.40).price_forward_start(&option, 100.0, 0.05, 0.0, 1.0);
    assert!(base > 0.0);
    assert!(high > base);
  }
}
