//! # Autocallable
//!
//! Auto-callable / phoenix / athena structured note. Observation grid
//! $t_1 < t_2 < \cdots < t_M = T$. Each observation date checks an upper
//! "autocall" barrier; if breached the note redeems early at par plus the
//! accrued coupon. Otherwise a coupon may be paid (phoenix) or skipped, and
//! at maturity the principal is at risk via a knock-in barrier on a worst-of
//! basket (here single underlying).
//!
//! Source:
//! - Bouzoubaa, M. & Osseiran, A. (2010), "Exotic Options and Hybrids", §11
//! - Kuklinski, J., Papaioannou, P. & Tyloo, K. (2016), "Pricing Weakly
//!   Model Dependent Barrier Products", arXiv:1608.00280
//! - Cibrario, F. et al. (2025), "Autocallable Options Pricing with
//!   Integration-Based Exponential Amplitude Loading", arXiv:2507.19039
//!
use ndarray::Array1;
use rayon::prelude::*;

use crate::traits::FloatExt;

/// Barrier observation style for the knock-in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KnockInStyle {
  /// Knock-in barrier checked only at maturity.
  EuropeanAtMaturity,
  /// Knock-in barrier checked on each observation date.
  Discrete,
  /// Knock-in barrier monitored continuously (approximated by a fine grid).
  Continuous,
}

/// Autocallable / phoenix / athena structured note. All barriers are
/// expressed as a fraction of the initial spot $S_0$.
#[derive(Debug, Clone)]
pub struct AutocallablePricer {
  /// Spot price.
  pub s: f64,
  /// Notional.
  pub notional: f64,
  /// Observation times $t_1, \ldots, t_M$ (years from now).
  pub observation_times: Array1<f64>,
  /// Autocall barrier (e.g. 1.0 = par level).
  pub autocall_barrier: f64,
  /// Coupon barrier (lower than autocall, e.g. 0.7). Pays the coupon when
  /// the underlying is at or above this level on the observation date.
  pub coupon_barrier: f64,
  /// Per-period coupon (already scaled to the period length, e.g. 0.04 for
  /// a 4% periodic coupon).
  pub coupon: f64,
  /// Knock-in barrier on principal (e.g. 0.6 — below this at maturity the
  /// holder takes a linear loss).
  pub knock_in_barrier: f64,
  /// Knock-in observation style.
  pub knock_in_style: KnockInStyle,
  /// Memory effect (athena): missed coupons accumulate and are paid when
  /// the next coupon barrier is met. False = phoenix without memory.
  pub memory: bool,
  /// Risk-free rate.
  pub r: f64,
  /// Dividend yield.
  pub q: f64,
  /// Volatility.
  pub sigma: f64,
  /// Number of MC paths.
  pub n_paths: usize,
  /// Time-steps per observation period when monitoring continuous KI.
  pub steps_per_period: usize,
}

impl AutocallablePricer {
  pub fn price(&self) -> f64 {
    let n_obs = self.observation_times.len();
    assert!(n_obs > 0, "need at least one observation date");
    let auto_b = self.autocall_barrier * self.s;
    let coup_b = self.coupon_barrier * self.s;
    let ki_b = self.knock_in_barrier * self.s;
    let times = self.observation_times.clone();
    let style = self.knock_in_style;
    let memory = self.memory;
    let coupon = self.coupon;
    let s0 = self.s;
    let notional = self.notional;
    let n_paths = self.n_paths;
    let steps_per_period = self.steps_per_period.max(1);

    // Worst-case normals required per path: every observation period uses
    // either one big step or `steps_per_period` substeps (continuous KI).
    let max_normals_per_path = match style {
      KnockInStyle::Continuous => n_obs * steps_per_period,
      _ => n_obs,
    };
    let mut all_z = vec![0.0_f64; n_paths * max_normals_per_path];
    <f64 as FloatExt>::fill_standard_normal_slice(&mut all_z);

    let sum: f64 = (0..n_paths)
      .into_par_iter()
      .map(|p| {
        let z_path = &all_z[p * max_normals_per_path..(p + 1) * max_normals_per_path];
        let mut z_idx = 0_usize;
        let mut s_prev = s0;
        let mut t_prev = 0.0;
        let mut pv: f64 = 0.0;
        let mut missed_coupons: f64 = 0.0;
        let mut autocalled = false;
        let mut autocall_time = 0.0;
        let mut breached_ki = false;
        let mut s_at_t: f64 = 0.0;

        for i in 0..n_obs {
          let t_i = times[i];
          let dt = t_i - t_prev;
          // sub-grid for continuous KI monitoring; otherwise one big step
          let sub_steps = match style {
            KnockInStyle::Continuous => steps_per_period,
            _ => 1,
          };
          let dt_sub = dt / sub_steps as f64;
          let drift = (self.r - self.q - 0.5 * self.sigma * self.sigma) * dt_sub;
          let vol = self.sigma * dt_sub.sqrt();
          let mut s_curr = s_prev;
          for _ in 0..sub_steps {
            let z = z_path[z_idx];
            z_idx += 1;
            s_curr *= (drift + vol * z).exp();
            if matches!(style, KnockInStyle::Continuous) && s_curr <= ki_b {
              breached_ki = true;
            }
          }
          if matches!(style, KnockInStyle::Discrete) && s_curr <= ki_b {
            breached_ki = true;
          }
          s_at_t = s_curr;
          // Autocall check
          if s_curr >= auto_b {
            let cf = notional * (1.0 + coupon + missed_coupons);
            pv += cf * (-self.r * t_i).exp();
            autocalled = true;
            autocall_time = t_i;
            break;
          }
          // Coupon barrier
          if s_curr >= coup_b {
            let cf = notional * (coupon + if memory { missed_coupons } else { 0.0 });
            pv += cf * (-self.r * t_i).exp();
            missed_coupons = 0.0;
          } else if memory {
            missed_coupons += coupon;
          }
          s_prev = s_curr;
          t_prev = t_i;
        }
        if !autocalled {
          if matches!(style, KnockInStyle::EuropeanAtMaturity) && s_at_t <= ki_b {
            breached_ki = true;
          }
          let t_mat = times[n_obs - 1];
          let principal = if breached_ki {
            notional * (s_at_t / s0)
          } else {
            notional
          };
          pv += principal * (-self.r * t_mat).exp();
        }
        let _ = autocall_time;
        pv
      })
      .sum();
    sum / n_paths as f64
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use ndarray::array;

  fn quarterly_obs(years: f64, n: usize) -> Array1<f64> {
    let dt = years / n as f64;
    Array1::from_iter((1..=n).map(|i| i as f64 * dt))
  }

  /// With autocall barrier far below spot the note redeems at the first
  /// observation. Price approximately equals $\notional \cdot (1 + c) \cdot
  /// e^{-r t_1}$.
  #[test]
  fn deep_in_autocall_redeems_first_period() {
    let p = AutocallablePricer {
      s: 100.0,
      notional: 100.0,
      observation_times: array![0.25, 0.5, 0.75, 1.0],
      autocall_barrier: 0.5, // 50% of spot
      coupon_barrier: 0.7,
      coupon: 0.04,
      knock_in_barrier: 0.6,
      knock_in_style: KnockInStyle::EuropeanAtMaturity,
      memory: false,
      r: 0.03,
      q: 0.0,
      sigma: 0.20,
      n_paths: 20_000,
      steps_per_period: 4,
    };
    let price = p.price();
    let expected = 100.0 * 1.04 * (-0.03_f64 * 0.25).exp();
    let rel = (price - expected).abs() / expected;
    assert!(rel < 0.01, "price={price}, expected={expected}, rel={rel}");
  }

  /// Memory (athena) version is at least as expensive as phoenix
  /// (no-memory).
  #[test]
  fn memory_premium_over_phoenix() {
    let base = AutocallablePricer {
      s: 100.0,
      notional: 100.0,
      observation_times: quarterly_obs(2.0, 8),
      autocall_barrier: 1.0,
      coupon_barrier: 0.75,
      coupon: 0.025,
      knock_in_barrier: 0.65,
      knock_in_style: KnockInStyle::EuropeanAtMaturity,
      memory: false,
      r: 0.03,
      q: 0.01,
      sigma: 0.30,
      n_paths: 30_000,
      steps_per_period: 4,
    };
    let phoenix = base.clone();
    let athena = AutocallablePricer { memory: true, ..base };
    assert!(athena.price() >= phoenix.price() - 1e-3);
  }

  /// Continuous knock-in monitoring should give a price below or equal to
  /// the European-at-maturity knock-in (more chances to breach).
  #[test]
  fn continuous_ki_below_european_ki() {
    let base = AutocallablePricer {
      s: 100.0,
      notional: 100.0,
      observation_times: quarterly_obs(1.0, 4),
      autocall_barrier: 1.05,
      coupon_barrier: 0.70,
      coupon: 0.03,
      knock_in_barrier: 0.65,
      knock_in_style: KnockInStyle::EuropeanAtMaturity,
      memory: false,
      r: 0.03,
      q: 0.0,
      sigma: 0.30,
      n_paths: 50_000,
      steps_per_period: 8,
    };
    let euro = base.clone();
    let cont = AutocallablePricer { knock_in_style: KnockInStyle::Continuous, ..base };
    assert!(cont.price() <= euro.price() + 1e-2);
  }

  /// At near-zero volatility, with spot strictly above the autocall barrier,
  /// all paths must redeem at the first observation date.
  #[test]
  fn zero_vol_spot_above_autocall() {
    let p = AutocallablePricer {
      s: 120.0,
      notional: 100.0,
      observation_times: array![0.5, 1.0],
      autocall_barrier: 1.0, // 1.0 * 120 = 120 — spot exactly at barrier
      coupon_barrier: 0.5,
      coupon: 0.05,
      knock_in_barrier: 0.4,
      knock_in_style: KnockInStyle::EuropeanAtMaturity,
      memory: false,
      r: 0.0,
      q: 0.0,
      sigma: 1e-6,
      n_paths: 5_000,
      steps_per_period: 1,
    };
    // Use a barrier slightly below spot so all paths trigger.
    let mut p = p;
    p.autocall_barrier = 0.95;
    let price = p.price();
    let expected = 100.0 * 1.05;
    let rel = (price - expected).abs() / expected;
    assert!(rel < 0.005, "price={price}");
  }
}
