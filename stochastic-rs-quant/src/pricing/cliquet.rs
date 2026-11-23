//! Cliquet (ratchet) options, chains of forward-start options; the common payoffs:
//!
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
//! Reference: Wilmott (2002), Wilmott Mag.
//! Reference: Hess (2018), Mod. Stoch. Theory Appl., arXiv:1810.09670, DOI 10.15559/18-VMSTA107
//! Reference: Haug (2007), Option Pricing Formulas, 2nd ed., ISBN 978-0-07-138997-6
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::DistributionSampler;
use stochastic_rs_distributions::RealExt;
use stochastic_rs_distributions::SimdDistribution;
use stochastic_rs_distributions::normal::SimdNormal;
use stochastic_rs_distributions::special::norm_cdf;
use stochastic_rs_stochastic::mc::McEstimate;

use crate::pricing::exotic::CliquetOption;
use crate::pricing::exotic::assert_sigma;
use crate::pricing::mc_stats::par_estimate;

/// Closed-form cliquet on `M` equal sub-periods under Black-Scholes: each period's forward-start
/// `min(max(R_i, f_l), c_l)` has one value by scaling invariance, so the price is `M` times it.
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

  /// Undiscounted `E^Q[min(max(R, f_l), c_l)]` over one period of length `τ`, the same for every
  /// period by Black-Scholes scaling invariance.
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

/// Monte Carlo price of a [`CliquetOption`], global floor and cap included, which the closed form
/// answers with NaN.
#[derive(Debug, Clone, Copy)]
pub struct McCliquetPricer<S: SeedExt = Unseeded> {
  /// Annualised Black–Scholes volatility.
  sigma: f64,
  n_paths: usize,
  /// Each call draws from a clone, so a `Deterministic` seed replays.
  pub seed: S,
}

impl McCliquetPricer {
  /// Panics unless `sigma` is finite and `>= 0` and `n_paths >= 1`.
  pub fn new(sigma: f64, n_paths: usize) -> Self {
    assert_sigma(sigma);
    assert!(
      n_paths >= 1,
      "n_paths must satisfy `n_paths >= 1`, got n_paths = {n_paths:?}"
    );
    Self {
      sigma,
      n_paths,
      seed: Unseeded,
    }
  }
}

impl<S: SeedExt> McCliquetPricer<S> {
  pub fn with_seed<S2: SeedExt>(self, seed: S2) -> McCliquetPricer<S2> {
    McCliquetPricer {
      sigma: self.sigma,
      n_paths: self.n_paths,
      seed,
    }
  }

  /// Discounted estimate at [`CliquetPricer::price`]'s query, which reads no spot; NaN for a
  /// negative or NaN `tau` or a NaN rate.
  pub fn price(&self, option: &CliquetOption, r: f64, q: f64, tau: f64) -> McEstimate<f64> {
    if tau.is_nan() || tau < 0.0 || r.is_nan() || q.is_nan() {
      return McEstimate {
        mean: f64::NAN,
        std_err: f64::NAN,
        n_samples: 0,
      };
    }
    let CliquetOption {
      notional,
      periods: m,
      local_floor,
      local_cap,
      global_floor,
      global_cap,
    } = *option;
    let period_tau = tau / m as f64;
    let drift = (r - q - 0.5 * self.sigma * self.sigma) * period_tau;
    let vol = self.sigma * period_tau.sqrt();
    let f_l = local_floor.unwrap_or(f64::NEG_INFINITY);
    let c_l = local_cap.unwrap_or(f64::INFINITY);
    let f_g = global_floor.unwrap_or(f64::NEG_INFINITY);
    let c_g = global_cap.unwrap_or(f64::INFINITY);

    let mut all_z = vec![0.0_f64; self.n_paths * m];
    let seed = self.seed.clone();
    SimdNormal::<f64>::new(0.0, 1.0)
      .seeded(&seed)
      .fill_slice(&mut all_z);

    let payoff_of = |p: usize| -> f64 {
      let returns = all_z[p * m..(p + 1) * m]
        .iter()
        .map(|&z| (drift + vol * z).exp_m1().max_or_nan(f_l).min_or_nan(c_l));
      returns.sum::<f64>().min_or_nan(c_g).max_or_nan(f_g)
    };
    par_estimate(self.n_paths, notional * (-r * tau).exp(), payoff_of)
  }
}

#[cfg(test)]
mod tests {
  use stochastic_rs_core::simd_rng::Deterministic;

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
    let mc = McCliquetPricer::new(0.25, 100_000)
      .price(&option, 0.04, 0.0, 1.0)
      .mean;
    let rel = (cf - mc).abs() / cf.abs().max(1e-10);
    assert!(rel < 0.04, "cf={cf}, mc={mc}, rel={rel}");
  }

  /// Adding a global cap reduces the MC price.
  #[test]
  fn global_cap_reduces_price() {
    let mc = McCliquetPricer::new(0.25, 50_000);
    let no_cap = cliquet(100.0, 12, Some(0.0), Some(0.05));
    let capped = CliquetOption {
      global_cap: Some(0.20),
      ..no_cap
    };
    let no_cap = mc.price(&no_cap, 0.04, 0.0, 1.0).mean;
    let capped = mc.price(&capped, 0.04, 0.0, 1.0).mean;
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

  #[test]
  fn a_nan_bound_prices_nan_in_both_pricers() {
    let mc = McCliquetPricer::new(0.2, 2_000).with_seed(Deterministic::new(1));
    let bounded = cliquet(1.0, 4, Some(0.0), Some(0.05));
    for option in [
      cliquet(1.0, 4, Some(f64::NAN), Some(0.05)),
      cliquet(1.0, 4, Some(0.0), Some(f64::NAN)),
      CliquetOption {
        global_floor: Some(f64::NAN),
        ..bounded
      },
      CliquetOption {
        global_cap: Some(f64::NAN),
        ..bounded
      },
    ] {
      assert!(
        CliquetPricer::new(0.2)
          .price(&option, 0.03, 0.0, 1.0)
          .is_nan()
      );
      let est = mc.price(&option, 0.03, 0.0, 1.0);
      assert!(
        est.mean.is_nan() && est.std_err.is_nan(),
        "{option:?}: {est}"
      );
    }
  }
}
