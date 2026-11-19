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
use stochastic_rs_distributions::RealExt;
use stochastic_rs_distributions::special::norm_cdf;
use stochastic_rs_stochastic::mc::McEstimate;

use crate::pricing::exotic::CliquetOption;
use crate::pricing::exotic::assert_sigma;
use crate::pricing::mc_stats::std_err_from_sums;
use crate::traits::FloatExt;

/// Closed-form cliquet on $M$ equal sub-periods under Black-Scholes.
///
/// Each period contributes a forward-start payoff
/// $\min(\max(R_i, f_l), c_l)$. Under BSM the price reduces to a finite sum
/// of forward-start option values, which by scaling invariance share a
/// common value (so the total is $M$ times one).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CliquetPricer {
  /// Constant volatility.
  pub sigma: f64,
}

impl CliquetPricer {
  /// Panics unless `sigma` is finite and `>= 0`; at `0` the deterministic limit, NaN where a local
  /// bound equals the period's forward return.
  pub fn new(sigma: f64) -> Self {
    assert_sigma(sigma);
    Self { sigma }
  }

  /// The summed period payoffs paid at `tau`, the zero-length periods' at `tau = 0`; NaN for a NaN
  /// rate, a negative or NaN `tau` or a global bound, which only [`McCliquetPricer`] prices.
  pub fn price(&self, option: &CliquetOption, r: f64, q: f64, tau: f64) -> f64 {
    let global = option.global_floor.is_some() || option.global_cap.is_some();
    if global || tau.is_nan() || tau < 0.0 || r.is_nan() || q.is_nan() {
      return f64::NAN;
    }
    if tau == 0.0 {
      let floor = option.local_floor.unwrap_or(f64::NEG_INFINITY);
      let cap = option.local_cap.unwrap_or(f64::INFINITY);
      return option.notional * option.periods as f64 * 0.0_f64.max_or_nan(floor).min_or_nan(cap);
    }
    let period_tau = tau / option.periods as f64;
    let per_period = self.expected_period_payoff(option, r, q, period_tau);
    option.notional * option.periods as f64 * per_period * (-r * tau).exp()
  }

  /// Risk-neutral expectation $E^Q[\min(\max(R, f_l), c_l)]$ for one
  /// period of length $\tau$ — undiscounted; the BSM scaling invariance
  /// makes this the same for every period.
  fn expected_period_payoff(&self, option: &CliquetOption, r: f64, q: f64, tau: f64) -> f64 {
    let v = self.sigma;
    let drift = r - q;
    match (option.local_floor, option.local_cap) {
      (None, None) => (drift * tau).exp() - 1.0,
      (Some(f), None) => f + Self::expected_max_return_minus(r, q, v, tau, 1.0 + f),
      (None, Some(c)) => {
        let unbounded = (drift * tau).exp() - 1.0;
        unbounded - Self::expected_max_return_minus(r, q, v, tau, 1.0 + c)
      }
      (Some(f), Some(c)) => {
        f + Self::expected_max_return_minus(r, q, v, tau, 1.0 + f)
          - Self::expected_max_return_minus(r, q, v, tau, 1.0 + c)
      }
    }
  }

  /// $E^Q[\max(S_T/S_0 - K, 0)] = e^{(r-q)T} N(d_1) - K N(d_2)$ —
  /// undiscounted.
  fn expected_max_return_minus(r: f64, q: f64, sigma: f64, t: f64, k: f64) -> f64 {
    let v_sq = sigma * sigma;
    let drift = r - q;
    let sqrt_t = t.sqrt();
    let d1 = ((1.0 / k).ln() + (drift + 0.5 * v_sq) * t) / (sigma * sqrt_t);
    let d2 = d1 - sigma * sqrt_t;
    (drift * t).exp() * norm_cdf(d1) - k * norm_cdf(d2)
  }
}

/// Monte-Carlo cliquet pricer with optional global cap/floor on the sum of
/// local-capped returns.
#[derive(Debug, Clone)]
pub struct McCliquetPricer {
  /// Spot price.
  pub s: f64,
  /// Notional the summed period payoffs are paid on.
  pub notional: f64,
  /// Number of sub-periods.
  pub m: usize,
  /// Time to maturity in years.
  pub tau: f64,
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
  /// Floor on the sum of the locally capped and floored period returns.
  pub global_floor: Option<f64>,
  /// Cap on that sum.
  pub global_cap: Option<f64>,
  /// Number of MC paths.
  pub n_paths: usize,
}

impl McCliquetPricer {
  pub fn price(&self) -> McEstimate<f64> {
    let period_tau = self.tau / self.m as f64;
    let drift = (self.r - self.q - 0.5 * self.sigma * self.sigma) * period_tau;
    let vol = self.sigma * period_tau.sqrt();
    let m = self.m;
    let f_l = self.local_floor.unwrap_or(f64::NEG_INFINITY);
    let c_l = self.local_cap.unwrap_or(f64::INFINITY);
    let f_g = self.global_floor.unwrap_or(f64::NEG_INFINITY);
    let c_g = self.global_cap.unwrap_or(f64::INFINITY);

    // Pre-generate all standard normals using the project's SIMD ziggurat.
    let mut all_z = vec![0.0_f64; self.n_paths * m];
    <f64 as FloatExt>::fill_standard_normal_slice(&mut all_z);

    let payoff_of = |p: usize| -> f64 {
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
    };
    let sum: f64 = (0..self.n_paths).into_par_iter().map(&payoff_of).sum();
    let sum_sq: f64 = (0..self.n_paths)
      .into_par_iter()
      .map(|p| {
        let y = payoff_of(p);
        y * y
      })
      .sum();

    let scale = self.notional * (-self.r * self.tau).exp();
    McEstimate {
      mean: scale * sum / self.n_paths as f64,
      std_err: scale.abs() * std_err_from_sums(sum, sum_sq, self.n_paths),
      n_samples: self.n_paths,
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  fn cliquet(
    notional: f64,
    periods: usize,
    local_floor: Option<f64>,
    local_cap: Option<f64>,
  ) -> CliquetOption {
    CliquetOption {
      notional,
      periods,
      local_floor,
      local_cap,
      global_floor: None,
      global_cap: None,
    }
  }

  /// Cliquet with M=1 and no caps pays the simple return $R$ at maturity.
  /// PV = $e^{-rT} E^Q[R] = e^{-rT}(e^{(r-q)T} - 1)$.
  #[test]
  fn unbounded_cliquet_one_period_equals_expected_return() {
    let r: f64 = 0.05;
    let q: f64 = 0.0;
    let tau: f64 = 1.0;
    let option = cliquet(1.0, 1, None, None);
    let price = CliquetPricer::new(0.20).price(&option, r, q, tau);
    let expected = (-r * tau).exp() * (((r - q) * tau).exp() - 1.0);
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
      let option = cliquet(100.0, m, Some(0.0), None);
      let price = CliquetPricer::new(0.20).price(&option, 0.05, 0.0, 1.0);
      // monotonic: more sub-periods, more upside captured (per Jensen)
      assert!(price > prev, "m={m}: price={price} not > prev={prev}");
      prev = price;
    }
  }

  /// Closed-form cliquet should agree with MC within ~3% (no global cap).
  #[test]
  fn closed_form_matches_mc() {
    let option = cliquet(100.0, 12, Some(-0.01), Some(0.04));
    let cf = CliquetPricer::new(0.25).price(&option, 0.04, 0.0, 1.0);
    let mc = McCliquetPricer {
      s: 100.0,
      notional: 100.0,
      m: 12,
      tau: 1.0,
      r: 0.04,
      q: 0.0,
      sigma: 0.25,
      local_floor: Some(-0.01),
      local_cap: Some(0.04),
      global_floor: None,
      global_cap: None,
      n_paths: 100_000,
    }
    .price()
    .mean;
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
      tau: 1.0,
      r: 0.04,
      q: 0.0,
      sigma: 0.25,
      local_floor: Some(0.0),
      local_cap: Some(0.05),
      global_floor: None,
      global_cap: None,
      n_paths: 50_000,
    }
    .price()
    .mean;
    let capped = McCliquetPricer {
      s: 100.0,
      notional: 100.0,
      m: 12,
      tau: 1.0,
      r: 0.04,
      q: 0.0,
      sigma: 0.25,
      local_floor: Some(0.0),
      local_cap: Some(0.05),
      global_floor: None,
      global_cap: Some(0.20),
      n_paths: 50_000,
    }
    .price()
    .mean;
    assert!(capped < no_cap, "capped={capped}, no_cap={no_cap}");
  }

  /// Higher volatility increases a floored cliquet's value (vega positive).
  #[test]
  fn floored_cliquet_has_positive_vega() {
    let option = cliquet(100.0, 12, Some(0.0), None);
    let make = |sigma: f64| CliquetPricer::new(sigma).price(&option, 0.04, 0.0, 1.0);
    assert!(make(0.30) > make(0.15));
  }

  #[test]
  fn the_closed_form_answers_a_global_floor_or_cap_with_nan() {
    let pricer = CliquetPricer::new(0.25);
    let local = cliquet(100.0, 12, Some(0.0), Some(0.05));
    let floored = CliquetOption {
      global_floor: Some(0.0),
      ..local
    };
    let capped = CliquetOption {
      global_cap: Some(0.2),
      ..local
    };
    assert!(pricer.price(&local, 0.04, 0.0, 1.0).is_finite());
    assert!(pricer.price(&floored, 0.04, 0.0, 1.0).is_nan());
    assert!(pricer.price(&capped, 0.04, 0.0, 1.0).is_nan());
  }
}
