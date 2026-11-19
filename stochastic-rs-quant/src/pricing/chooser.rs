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
use stochastic_rs_distributions::special::norm_cdf;

use crate::OptionType;
use crate::pricing::exotic::ChooserOption;
use crate::pricing::exotic::ComplexChooserOption;
use crate::pricing::exotic::ForwardStartOption;
use crate::pricing::exotic::assert_sigma;

/// Black–Scholes prices of the simple and complex choosers (Rubinstein 1991) and of the forward
/// start (Rubinstein 1990).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChooserPricer {
  /// Annualised Black–Scholes volatility.
  pub sigma: f64,
}

impl ChooserPricer {
  /// Panics unless `sigma` is finite and `>= 0`; at `0` the simple chooser and the forward start
  /// price their deterministic limits, NaN at the money forward, and the complex chooser NaN.
  pub fn new(sigma: f64) -> Self {
    assert_sigma(sigma);
    Self { sigma }
  }

  /// The simple chooser $W$ above at the query `(s, r, q, tau)`; at `tau = 0` the better leg's
  /// payoff `|s - strike|`, and NaN for a negative or NaN `tau`.
  pub fn price_simple(&self, option: &ChooserOption, s: f64, r: f64, q: f64, tau: f64) -> f64 {
    let ChooserOption {
      strike: k,
      choice_time: t1,
    } = *option;
    if tau.is_nan() || tau < 0.0 {
      return f64::NAN;
    }
    if tau == 0.0 {
      return (s - k).abs();
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

  /// The complex chooser at `(s, r, q)`, its maturities the contract's, through the spot $S^*$ at
  /// which the call and put are equal; NaN unless `0 < choice_time <= min(call_tau, put_tau)`.
  pub fn price_complex(&self, option: &ComplexChooserOption, s: f64, r: f64, q: f64) -> f64 {
    let ComplexChooserOption {
      call_strike: k_call,
      call_tau: t_call,
      put_strike: k_put,
      put_tau: t_put,
      choice_time: t1,
    } = *option;
    if !(0.0 < t1 && t1 <= t_call && t1 <= t_put) {
      return f64::NAN;
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

  /// The forward start at the query `(s, r, q, tau)`, its strike fixed at `start_time` as
  /// `moneyness` times the spot then.
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
    let remaining_tau = tau - t1;
    let v = self.sigma;
    let v2 = v * v;
    let b = r - q;
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
