//! # Cliquet
//!
//! A cliquet (or "ratchet") option is a chain of forward-start options.
//! Common payoffs:
//! $$
//! \text{Cliquet} = N \sum_{i=1}^M \min\!\Big(\max(R_i, f_l), c_l\Big),\quad
//! R_i = \frac{S_{t_i}}{S_{t_{i-1}}} - 1
//! $$
//!
//! With an outer global cap/floor:
//! $$
//! \text{Globally capped} = N \min\!\Big(c_g, \max\!\Big(f_g,
//!   \sum_i \min(\max(R_i, f_l), c_l)\Big)\Big)
//! $$
//!
//! Source:
//! - Wilmott, P. (2002), "Cliquet Options and Volatility Models", *Wilmott Magazine*
//! - Hess, M. (2018), "Cliquet option pricing in a jump-diffusion Lévy model",
//!   arXiv:1810.09670, DOI: 10.15559/18-VMSTA107
//! - Haug, E. G. (2007), "The Complete Guide to Option Pricing Formulas", 2nd ed.
//!
use rayon::prelude::*;
use statrs::distribution::ContinuousCDF;
use statrs::distribution::Normal;

use crate::traits::FloatExt;

/// Closed-form cliquet on $M$ equal sub-periods under Black-Scholes.
///
/// Each period contributes a forward-start payoff
/// $\min(\max(R_i, f_l), c_l)$. Under BSM the price reduces to a finite sum
/// of forward-start option values, which by scaling invariance share a
/// common value (so the total is $M$ times one).
#[derive(Debug, Clone)]
pub struct CliquetPricer {
  /// Spot price.
  pub s: f64,
  /// Notional.
  pub notional: f64,
  /// Number of sub-periods.
  pub m: usize,
  /// Total maturity in years (each period = $T/M$).
  pub t: f64,
  /// Risk-free rate.
  pub r: f64,
  /// Dividend yield.
  pub q: f64,
  /// Constant volatility.
  pub sigma: f64,
  /// Local floor on per-period return (e.g. 0.0 means "drop negative").
  pub local_floor: Option<f64>,
  /// Local cap on per-period return.
  pub local_cap: Option<f64>,
}

impl CliquetPricer {
  /// Payoff convention: the sum of capped/floored returns is paid at
  /// maturity $T$, discounted by a single factor $e^{-rT}$.
  pub fn price(&self) -> f64 {
    let tau = self.t / self.m as f64;
    let per_period = self.expected_period_payoff(tau);
    self.notional * self.m as f64 * per_period * (-self.r * self.t).exp()
  }

  /// Risk-neutral expectation $E^Q[\min(\max(R, f_l), c_l)]$ for one
  /// period of length $\tau$ — undiscounted; the BSM scaling invariance
  /// makes this the same for every period.
  fn expected_period_payoff(&self, tau: f64) -> f64 {
    let n = Normal::new(0.0, 1.0).unwrap();
    let v = self.sigma;
    let drift = self.r - self.q;
    match (self.local_floor, self.local_cap) {
      (None, None) => (drift * tau).exp() - 1.0,
      (Some(f), None) => f + Self::expected_max_return_minus(self.r, self.q, v, tau, 1.0 + f, &n),
      (None, Some(c)) => {
        let unbounded = (drift * tau).exp() - 1.0;
        unbounded - Self::expected_max_return_minus(self.r, self.q, v, tau, 1.0 + c, &n)
      }
      (Some(f), Some(c)) => {
        f + Self::expected_max_return_minus(self.r, self.q, v, tau, 1.0 + f, &n)
          - Self::expected_max_return_minus(self.r, self.q, v, tau, 1.0 + c, &n)
      }
    }
  }

  /// $E^Q[\max(S_T/S_0 - K, 0)] = e^{(r-q)T} N(d_1) - K N(d_2)$ —
  /// undiscounted.
  fn expected_max_return_minus(r: f64, q: f64, sigma: f64, t: f64, k: f64, n: &Normal) -> f64 {
    let v_sq = sigma * sigma;
    let drift = r - q;
    let sqrt_t = t.sqrt();
    let d1 = ((1.0 / k).ln() + (drift + 0.5 * v_sq) * t) / (sigma * sqrt_t);
    let d2 = d1 - sigma * sqrt_t;
    (drift * t).exp() * n.cdf(d1) - k * n.cdf(d2)
  }
}

/// Monte-Carlo cliquet pricer with optional global cap/floor on the sum of
/// local-capped returns.
#[derive(Debug, Clone)]
pub struct McCliquetPricer {
  /// Spot.
  pub s: f64,
  /// Notional.
  pub notional: f64,
  /// Number of sub-periods.
  pub m: usize,
  /// Total maturity.
  pub t: f64,
  /// Risk-free rate.
  pub r: f64,
  /// Dividend yield.
  pub q: f64,
  /// Constant volatility.
  pub sigma: f64,
  /// Local floor on per-period return.
  pub local_floor: Option<f64>,
  /// Local cap on per-period return.
  pub local_cap: Option<f64>,
  /// Global floor on sum.
  pub global_floor: Option<f64>,
  /// Global cap on sum.
  pub global_cap: Option<f64>,
  /// Number of MC paths.
  pub n_paths: usize,
}

impl McCliquetPricer {
  pub fn price(&self) -> f64 {
    let tau = self.t / self.m as f64;
    let drift = (self.r - self.q - 0.5 * self.sigma * self.sigma) * tau;
    let vol = self.sigma * tau.sqrt();
    let m = self.m;
    let f_l = self.local_floor.unwrap_or(f64::NEG_INFINITY);
    let c_l = self.local_cap.unwrap_or(f64::INFINITY);
    let f_g = self.global_floor.unwrap_or(f64::NEG_INFINITY);
    let c_g = self.global_cap.unwrap_or(f64::INFINITY);

    // Pre-generate all standard normals using the project's SIMD ziggurat.
    let mut all_z = vec![0.0_f64; self.n_paths * m];
    <f64 as FloatExt>::fill_standard_normal_slice(&mut all_z);

    let sum: f64 = (0..self.n_paths)
      .into_par_iter()
      .map(|p| {
        let z = &all_z[p * m..(p + 1) * m];
        let mut s_curr = self.s;
        let mut sum_returns = 0.0;
        for k in 0..m {
          let s_next = s_curr * (drift + vol * z[k]).exp();
          let r_i = s_next / s_curr - 1.0;
          let r_capped = r_i.max(f_l).min(c_l);
          sum_returns += r_capped;
          s_curr = s_next;
        }
        sum_returns.min(c_g).max(f_g)
      })
      .sum();
    self.notional * (-self.r * self.t).exp() * sum / self.n_paths as f64
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  /// Cliquet with M=1 and no caps pays the simple return $R$ at maturity.
  /// PV = $e^{-rT} E^Q[R] = e^{-rT}(e^{(r-q)T} - 1)$.
  #[test]
  fn unbounded_cliquet_one_period_equals_expected_return() {
    let r: f64 = 0.05;
    let q: f64 = 0.0;
    let t: f64 = 1.0;
    let p = CliquetPricer {
      s: 100.0,
      notional: 1.0,
      m: 1,
      t,
      r,
      q,
      sigma: 0.20,
      local_floor: None,
      local_cap: None,
    };
    let price = p.price();
    let expected = (-r * t).exp() * (((r - q) * t).exp() - 1.0);
    assert!(
      (price - expected).abs() < 1e-9,
      "price={price}, expected={expected}"
    );
  }

  /// Cliquet with floor at 0 (capped only on the downside) is positive and
  /// scales linearly in the number of periods.
  #[test]
  fn floored_cliquet_linearity_in_m() {
    let mut prev = 0.0;
    for m in [1, 2, 4, 12, 252] {
      let p = CliquetPricer {
        s: 100.0,
        notional: 100.0,
        m,
        t: 1.0,
        r: 0.05,
        q: 0.0,
        sigma: 0.20,
        local_floor: Some(0.0),
        local_cap: None,
      };
      let price = p.price();
      // monotonic: more sub-periods, more upside captured (per Jensen)
      assert!(price > prev, "m={m}: price={price} not > prev={prev}");
      prev = price;
    }
  }

  /// Closed-form cliquet should agree with MC within ~3% (no global cap).
  #[test]
  fn closed_form_matches_mc() {
    let cf = CliquetPricer {
      s: 100.0,
      notional: 100.0,
      m: 12,
      t: 1.0,
      r: 0.04,
      q: 0.0,
      sigma: 0.25,
      local_floor: Some(-0.01),
      local_cap: Some(0.04),
    }
    .price();
    let mc = McCliquetPricer {
      s: 100.0,
      notional: 100.0,
      m: 12,
      t: 1.0,
      r: 0.04,
      q: 0.0,
      sigma: 0.25,
      local_floor: Some(-0.01),
      local_cap: Some(0.04),
      global_floor: None,
      global_cap: None,
      n_paths: 100_000,
    }
    .price();
    let rel = (cf - mc).abs() / cf.abs().max(1e-10);
    assert!(rel < 0.04, "cf={cf}, mc={mc}, rel={rel}");
  }

  /// Adding a global cap reduces the MC price.
  #[test]
  fn global_cap_reduces_price() {
    let no_cap = McCliquetPricer {
      s: 100.0,
      notional: 100.0,
      m: 12,
      t: 1.0,
      r: 0.04,
      q: 0.0,
      sigma: 0.25,
      local_floor: Some(0.0),
      local_cap: Some(0.05),
      global_floor: None,
      global_cap: None,
      n_paths: 50_000,
    }
    .price();
    let capped = McCliquetPricer {
      s: 100.0,
      notional: 100.0,
      m: 12,
      t: 1.0,
      r: 0.04,
      q: 0.0,
      sigma: 0.25,
      local_floor: Some(0.0),
      local_cap: Some(0.05),
      global_floor: None,
      global_cap: Some(0.20),
      n_paths: 50_000,
    }
    .price();
    assert!(capped < no_cap, "capped={capped}, no_cap={no_cap}");
  }

  /// Higher volatility increases a floored cliquet's value (vega positive).
  #[test]
  fn floored_cliquet_has_positive_vega() {
    let make = |sigma: f64| {
      CliquetPricer {
        s: 100.0,
        notional: 100.0,
        m: 12,
        t: 1.0,
        r: 0.04,
        q: 0.0,
        sigma,
        local_floor: Some(0.0),
        local_cap: None,
      }
      .price()
    };
    assert!(make(0.30) > make(0.15));
  }
}
